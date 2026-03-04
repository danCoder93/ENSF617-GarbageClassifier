import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from sklearn.metrics import auc, confusion_matrix, precision_recall_fscore_support, roc_curve
from sklearn.preprocessing import label_binarize

from pipeline_logger import PipelineLogger


@dataclass(frozen=True)
class TrainConfig:
    device: str
    max_epochs: int = 10
    logger: Optional[PipelineLogger] = None
    grad_clip_norm: Optional[float] = None
    save_path: str = "best_model.pth"
    log_every_n_steps: int = 20
    log_histograms_every_n_steps: int = 100
    use_amp: bool = False


class Trainer:
    def __init__(self, cfg: TrainConfig, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.logger = cfg.logger or PipelineLogger()
        self.loss_fn = loss_fn

        self.global_step = 0
        self.best_val_acc = float("-inf")

        self.scaler = torch.amp.GradScaler(enabled=(cfg.use_amp and self.device.type == "cuda"))
        self.log_every_n_steps = max(1, int(cfg.log_every_n_steps))
        self.log_histograms_every_n_steps = max(1, int(cfg.log_histograms_every_n_steps))

    def fit(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: Optimizer) -> None:
        model.to(self.device)
        self.logger.log_run_start(
            cfg=self.cfg,
            model=model,
            optimizer=optimizer,
            train_batches=self._safe_len(train_loader),
            val_batches=self._safe_len(val_loader),
        )

        for epoch in range(1, self.cfg.max_epochs + 1):
            train_m = self._run_epoch(model, train_loader, optimizer, epoch, stage="train")
            val_m = self._run_epoch(model, val_loader, optimizer=None, epoch=epoch, stage="val")

            self._maybe_checkpoint(model, val_m)
            self._log_epoch_metrics(epoch, train_m, val_m)

            self.logger.flush()

    def _run_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: Optional[Optimizer],
        epoch: int,
        stage: str,
    ) -> Dict[str, float]:
        is_train = stage == "train"
        model.train(is_train)

        loss_sum = 0.0
        correct = 0
        total = 0

        t0 = time.time()
        epoch_start = time.time()
        num_batches = self._safe_len(loader)
        epoch_batches = num_batches if num_batches is not None else 1

        grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
        with grad_ctx:
            for batch_idx, batch in enumerate(loader):
                step_start = time.time()
                batch = self._to_device(batch)
                labels = batch["labels"]
                bs = int(labels.shape[0])
                grad_norm = 0.0
                param_norm = 0.0

                if is_train:
                    optimizer.zero_grad(set_to_none=True)

                # amp -> fp32 -> fp16 faster
                # requires scaling and then scaling down
                with torch.amp.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                    logits = model(batch)
                    loss = self.loss_fn(logits, labels)

                if is_train:
                    self.scaler.scale(loss).backward()

                    if self.scaler.is_enabled():
                        self.scaler.unscale_(optimizer)

                    if self.cfg.grad_clip_norm is not None:
                        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip_norm))
                    else:
                        grad_norm = self._grad_norm(model.parameters())

                    self.scaler.step(optimizer)
                    self.scaler.update()
                    self.global_step += 1
                    param_norm = self._param_norm(model.parameters())

                # stats
                preds = logits.argmax(dim=1)
                probs = logits.softmax(dim=1)
                confidences = probs.max(dim=1).values
                loss_sum += float(loss.item()) * bs
                correct += int((preds == labels).sum().item())
                total += bs

                step_loss = float(loss.item())
                step_acc = float((preds == labels).float().mean().item())
                step_time = max(time.time() - step_start, 1e-12)
                samples_per_sec = bs / step_time

                log_step = self.global_step if is_train else ((max(epoch, 1) - 1) * epoch_batches + batch_idx + 1)
                internals = {
                    "batch_size": float(bs),
                    "step_time_s": float(step_time),
                    "samples_per_sec": float(samples_per_sec),
                    "logits_mean": float(logits.detach().mean().item()),
                    "logits_std": float(logits.detach().std(unbiased=False).item()),
                    "confidence_mean": float(confidences.detach().mean().item()),
                    "confidence_std": float(confidences.detach().std(unbiased=False).item()),
                }
                if is_train:
                    internals["lr"] = float(optimizer.param_groups[0]["lr"])
                    internals["grad_norm"] = float(grad_norm)
                    internals["param_norm"] = float(param_norm)
                    internals["amp_loss_scale"] = float(self.scaler.get_scale() if self.scaler.is_enabled() else 1.0)
                if self.device.type == "cuda":
                    internals["cuda_mem_alloc_mb"] = float(torch.cuda.memory_allocated(self.device) / (1024 ** 2))
                    internals["cuda_mem_reserved_mb"] = float(torch.cuda.memory_reserved(self.device) / (1024 ** 2))

                self.logger.log_step(
                    stage=stage,
                    step=log_step,
                    metrics={"loss": step_loss, "acc": step_acc},
                    internals=internals,
                    logits=logits,
                    confidences=confidences,
                    log_histograms=(batch_idx % self.log_histograms_every_n_steps == 0),
                )

                if batch_idx % self.log_every_n_steps == 0:
                    dt = time.time() - t0
                    print(
                        f"  [{stage}] epoch={epoch:02d} step={log_step} "
                        f"loss={step_loss:.4f} acc={step_acc:.4f} dt={dt:.1f}s"
                    )
                    t0 = time.time()

        avg_loss = loss_sum / max(total, 1)
        avg_acc = correct / max(total, 1)
        epoch_time = max(time.time() - epoch_start, 1e-12)
        return {
            "loss": float(avg_loss),
            "acc": float(avg_acc),
            "epoch_time_s": float(epoch_time),
            "samples_per_sec": float(total / epoch_time),
            "num_samples": float(total),
        }

    @torch.no_grad()
    def evaluate(self, model: nn.Module, loader: DataLoader, stage: str = "test") -> Dict[str, float]:
        model.to(self.device)
        state = torch.load(self.cfg.save_path, map_location=self.device)
        model.load_state_dict(state)
        model.eval()
        eval_outputs = self._collect_eval_outputs(model, loader, stage=stage)
        class_names = self._infer_class_names(loader, eval_outputs["num_classes"])
        report = self._build_eval_report(eval_outputs, class_names)
        self._save_eval_artifacts(report, stage)

        metrics = {
            "loss": report["loss"],
            "acc": report["acc"],
            "epoch_time_s": report["epoch_time_s"],
            "samples_per_sec": report["samples_per_sec"],
            "num_samples": report["num_samples"],
            "macro_f1": report["macro_f1"],
            "weighted_f1": report["weighted_f1"],
            "macro_roc_auc": report["macro_roc_auc"],
            "misclassified_total": report["misclassified_total"],
        }
        self.logger.log_evaluation(stage=stage, metrics=metrics, step=self.global_step)
        self.logger.flush()
        return metrics

    def _maybe_checkpoint(self, model: nn.Module, val_m: Dict[str, float]) -> None:
        score = float(val_m["acc"])
        if score > self.best_val_acc:
            self.best_val_acc = score
            dir_path = Path(self.cfg.save_path)
            dir_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.cfg.save_path)
            print(f"Saved best checkpoint to {self.cfg.save_path} (val_acc={score:.4f})")
            self.logger.log_checkpoint(epoch=self.global_step, save_path=self.cfg.save_path, score=score)

    def _log_epoch_metrics(self, epoch: int, train_m: Dict[str, float], val_m: Dict[str, float]) -> None:
        print(
            f"Epoch {epoch:02d}/{self.cfg.max_epochs:02d} | "
            f"train_loss={train_m['loss']:.4f} train_acc={train_m['acc']:.4f} | "
            f"val_loss={val_m['loss']:.4f} val_acc={val_m['acc']:.4f} | "
            f"best_val_acc={self.best_val_acc:.4f}"
        )
        self.logger.log_epoch(epoch=epoch, train_metrics=train_m, val_metrics=val_m, best_val_acc=self.best_val_acc)

    def _to_device(self, batch):
        if isinstance(batch, dict):
            out = {}
            for k, v in batch.items():
                out[k] = v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
            if "labels" not in out:
                raise KeyError("Batch dict must include 'labels'.")
            return out

        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            x, y = batch
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            return {"x": x, "labels": y}

        raise TypeError(f"Unsupported batch type: {type(batch)}")

    @staticmethod
    def _safe_len(loader: DataLoader) -> Optional[int]:
        try:
            return len(loader)
        except TypeError:
            return None

    @staticmethod
    def _grad_norm(parameters: Iterable[torch.Tensor]) -> float:
        total_sq = 0.0
        for p in parameters:
            if p.grad is None:
                continue
            grad = p.grad.detach()
            total_sq += float(torch.sum(grad * grad).item())
        return total_sq ** 0.5

    @staticmethod
    def _param_norm(parameters: Iterable[torch.Tensor]) -> float:
        total_sq = 0.0
        for p in parameters:
            data = p.detach()
            total_sq += float(torch.sum(data * data).item())
        return total_sq ** 0.5

    def _collect_eval_outputs(self, model: nn.Module, loader: DataLoader, stage: str) -> Dict[str, Any]:
        loss_sum = 0.0
        correct = 0
        total = 0
        num_classes = 0

        y_true: List[torch.Tensor] = []
        y_pred: List[torch.Tensor] = []
        y_prob: List[torch.Tensor] = []
        paths: List[str] = []
        raw_texts: List[str] = []

        t0 = time.time()
        eval_start = time.time()

        for batch_idx, batch in enumerate(loader):
            batch = self._to_device(batch)
            labels = batch["labels"]
            bs = int(labels.shape[0])

            with torch.amp.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                logits = model(batch)
                loss = self.loss_fn(logits, labels)

            preds = logits.argmax(dim=1)
            probs = logits.softmax(dim=1)

            num_classes = max(num_classes, int(logits.shape[1]))
            loss_sum += float(loss.item()) * bs
            correct += int((preds == labels).sum().item())
            total += bs

            y_true.append(labels.detach().cpu())
            y_pred.append(preds.detach().cpu())
            y_prob.append(probs.detach().float().cpu())

            batch_paths = list(batch.get("path", [""] * bs))
            batch_raw_text = list(batch.get("raw_text", [""] * bs))
            if len(batch_paths) < bs:
                batch_paths.extend([""] * (bs - len(batch_paths)))
            if len(batch_raw_text) < bs:
                batch_raw_text.extend([""] * (bs - len(batch_raw_text)))
            paths.extend(str(path) for path in batch_paths[:bs])
            raw_texts.extend(str(text) for text in batch_raw_text[:bs])

            if batch_idx % self.log_every_n_steps == 0:
                step_loss = float(loss.item())
                step_acc = float((preds == labels).float().mean().item())
                dt = time.time() - t0
                print(
                    f"  [{stage}] batch={batch_idx + 1:03d} "
                    f"loss={step_loss:.4f} acc={step_acc:.4f} dt={dt:.1f}s"
                )
                t0 = time.time()

        epoch_time = max(time.time() - eval_start, 1e-12)
        true_array = torch.cat(y_true).numpy() if y_true else np.asarray([], dtype=np.int64)
        pred_array = torch.cat(y_pred).numpy() if y_pred else np.asarray([], dtype=np.int64)
        prob_array = (
            torch.cat(y_prob).numpy()
            if y_prob
            else np.zeros((0, num_classes), dtype=np.float32)
        )

        return {
            "loss": float(loss_sum / max(total, 1)),
            "acc": float(correct / max(total, 1)),
            "epoch_time_s": float(epoch_time),
            "samples_per_sec": float(total / epoch_time),
            "num_samples": float(total),
            "num_classes": num_classes,
            "y_true": true_array,
            "y_pred": pred_array,
            "y_prob": prob_array,
            "paths": paths,
            "raw_texts": raw_texts,
        }

    def _infer_class_names(self, loader: DataLoader, num_classes: int) -> List[str]:
        dataset = getattr(loader, "dataset", None)
        if dataset is not None:
            classes = getattr(dataset, "classes", None)
            if classes and (num_classes == 0 or len(classes) == num_classes):
                return [str(name) for name in classes]

            inner = getattr(dataset, "ds", None)
            inner_classes = getattr(inner, "classes", None)
            if inner_classes and (num_classes == 0 or len(inner_classes) == num_classes):
                return [str(name) for name in inner_classes]

        return [f"class_{idx}" for idx in range(num_classes)]

    def _build_eval_report(self, eval_outputs: Dict[str, Any], class_names: List[str]) -> Dict[str, Any]:
        y_true = eval_outputs["y_true"]
        y_pred = eval_outputs["y_pred"]
        y_prob = eval_outputs["y_prob"]
        num_classes = len(class_names) if class_names else int(eval_outputs["num_classes"])
        if not class_names:
            class_names = [f"class_{idx}" for idx in range(num_classes)]

        label_indices = list(range(num_classes))
        conf_mat = confusion_matrix(y_true, y_pred, labels=label_indices)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=label_indices,
            zero_division=0,
        )

        per_class = {}
        misclassified_by_true = {name: 0 for name in class_names}
        misclassified_rows = []

        for idx, class_name in enumerate(class_names):
            per_class[class_name] = {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }

        for sample_idx, (true_idx, pred_idx) in enumerate(zip(y_true.tolist(), y_pred.tolist())):
            if int(true_idx) == int(pred_idx):
                continue

            true_label = class_names[int(true_idx)]
            pred_label = class_names[int(pred_idx)]
            misclassified_by_true[true_label] += 1
            confidence = 0.0
            if sample_idx < len(y_prob) and int(pred_idx) < y_prob.shape[1]:
                confidence = float(y_prob[sample_idx, int(pred_idx)])

            misclassified_rows.append({
                "path": eval_outputs["paths"][sample_idx] if sample_idx < len(eval_outputs["paths"]) else "",
                "true_label": true_label,
                "pred_label": pred_label,
                "confidence": confidence,
                "raw_text": eval_outputs["raw_texts"][sample_idx] if sample_idx < len(eval_outputs["raw_texts"]) else "",
            })

        roc_curves = []
        roc_auc_by_class = {name: None for name in class_names}
        roc_auc_values = []

        if len(y_true) > 0 and y_prob.size > 0 and num_classes > 0:
            y_true_bin = label_binarize(y_true, classes=label_indices)
            if num_classes == 2 and y_true_bin.ndim == 2 and y_true_bin.shape[1] == 1:
                y_true_bin = np.concatenate([1 - y_true_bin, y_true_bin], axis=1)

            for idx, class_name in enumerate(class_names):
                class_targets = y_true_bin[:, idx]
                if len(np.unique(class_targets)) < 2:
                    continue

                fpr, tpr, _ = roc_curve(class_targets, y_prob[:, idx])
                auc_value = float(auc(fpr, tpr))
                roc_auc_by_class[class_name] = auc_value
                roc_auc_values.append(auc_value)
                roc_curves.append({
                    "label": class_name,
                    "auc": auc_value,
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                })

        macro_f1 = float(np.mean(f1)) if len(f1) > 0 else 0.0
        weighted_f1 = float(np.average(f1, weights=support)) if int(np.sum(support)) > 0 else 0.0
        macro_roc_auc = float(np.mean(roc_auc_values)) if roc_auc_values else 0.0

        return {
            "loss": float(eval_outputs["loss"]),
            "acc": float(eval_outputs["acc"]),
            "epoch_time_s": float(eval_outputs["epoch_time_s"]),
            "samples_per_sec": float(eval_outputs["samples_per_sec"]),
            "num_samples": float(eval_outputs["num_samples"]),
            "class_names": class_names,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "macro_roc_auc": macro_roc_auc,
            "misclassified_total": float(len(misclassified_rows)),
            "per_class": per_class,
            "confusion_matrix": conf_mat.tolist(),
            "roc_auc_by_class": roc_auc_by_class,
            "roc_curves": roc_curves,
            "misclassified_by_true_label": misclassified_by_true,
            "misclassified_rows": misclassified_rows,
        }

    def _save_eval_artifacts(self, report: Dict[str, Any], stage: str) -> None:
        artifact_dir = Path("artifacts") / Path(self.cfg.save_path).stem / stage
        artifact_dir.mkdir(parents=True, exist_ok=True)

        metrics_payload = {
            "loss": report["loss"],
            "acc": report["acc"],
            "epoch_time_s": report["epoch_time_s"],
            "samples_per_sec": report["samples_per_sec"],
            "num_samples": report["num_samples"],
            "class_names": report["class_names"],
            "macro_f1": report["macro_f1"],
            "weighted_f1": report["weighted_f1"],
            "macro_roc_auc": report["macro_roc_auc"],
            "misclassified_total": report["misclassified_total"],
            "per_class": report["per_class"],
            "confusion_matrix": report["confusion_matrix"],
            "roc_auc_by_class": report["roc_auc_by_class"],
            "misclassified_by_true_label": report["misclassified_by_true_label"],
        }

        (artifact_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

        with (artifact_dir / "misclassified.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["path", "true_label", "pred_label", "confidence", "raw_text"],
            )
            writer.writeheader()
            writer.writerows(report["misclassified_rows"])

        conf_mat = np.asarray(report["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(conf_mat, cmap="Blues")
        ax.figure.colorbar(image, ax=ax)
        ax.set(
            xticks=np.arange(len(report["class_names"])),
            yticks=np.arange(len(report["class_names"])),
            xticklabels=report["class_names"],
            yticklabels=report["class_names"],
            xlabel="Predicted label",
            ylabel="True label",
            title="Confusion Matrix",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for row_idx in range(conf_mat.shape[0]):
            for col_idx in range(conf_mat.shape[1]):
                ax.text(col_idx, row_idx, int(conf_mat[row_idx, col_idx]), ha="center", va="center")
        fig.tight_layout()
        fig.savefig(artifact_dir / "confusion_matrix.png", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 6))
        plotted = False
        for curve in report["roc_curves"]:
            plotted = True
            ax.plot(curve["fpr"], curve["tpr"], label=f"{curve['label']} (AUC={curve['auc']:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        if plotted:
            ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(artifact_dir / "roc_curves.png", bbox_inches="tight")
        plt.close(fig)
