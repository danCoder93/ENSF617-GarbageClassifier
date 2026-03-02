from dataclasses import dataclass
from typing import Dict, Optional, Callable, Iterable
import time

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

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
        metrics = self._run_epoch(model, loader, optimizer=None, epoch=0, stage=stage)
        self.logger.log_evaluation(stage=stage, metrics=metrics, step=self.global_step)
        self.logger.flush()
        return metrics

    def _maybe_checkpoint(self, model: nn.Module, val_m: Dict[str, float]) -> None:
        score = float(val_m["acc"])
        if score > self.best_val_acc:
            self.best_val_acc = score
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
