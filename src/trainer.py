from dataclasses import dataclass
from typing import Dict, Optional, Callable
import time

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


@dataclass(frozen=True)
class TrainConfig:
    device: str
    max_epochs: int = 10
    writer: Optional[SummaryWriter] = None
    grad_clip_norm: Optional[float] = None
    save_path: str = "best_model.pth"
    log_every_n_steps: int = 20
    use_amp: bool = False


class Trainer:
    def __init__(self, cfg: TrainConfig, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.writer = cfg.writer
        self.loss_fn = loss_fn

        self.global_step = 0
        self.best_val_acc = float("-inf")

        self.scaler = torch.amp.GradScaler(enabled=(cfg.use_amp and self.device.type == "cuda"))

    def fit(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: Optimizer) -> None:
        model.to(self.device)

        for epoch in range(1, self.cfg.max_epochs + 1):
            train_m = self._run_epoch(model, train_loader, optimizer, epoch, stage="train")
            val_m = self._run_epoch(model, val_loader, optimizer=None, epoch=epoch, stage="val")

            self._log_epoch_metrics(epoch, train_m, val_m)
            self._maybe_checkpoint(model, val_m)

            if self.writer:
                self.writer.flush()

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

        grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
        with grad_ctx:
            for batch_idx, batch in enumerate(loader):
                batch = self._to_device(batch)
                labels = batch["labels"]
                bs = int(labels.shape[0])

                if is_train:
                    optimizer.zero_grad(set_to_none=True)

                # amp -> fp32 -> fp16 faster
                # requires scaling and then scaling down
                with torch.amp.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                    logits = model(batch)
                    loss = self.loss_fn(logits, labels)

                if is_train:
                    self.scaler.scale(loss).backward()

                    if self.cfg.grad_clip_norm is not None:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip_norm)

                    self.scaler.step(optimizer)
                    self.scaler.update()
                    self.global_step += 1

                # stats
                preds = logits.argmax(dim=1)
                loss_sum += float(loss.item()) * bs
                correct += int((preds == labels).sum().item())
                total += bs

                # step logging (train only)
                if is_train:
                    step_loss = float(loss.item())
                    step_acc = float((preds == labels).float().mean().item())

                    if batch_idx % self.cfg.log_every_n_steps == 0:
                        dt = time.time() - t0
                        print(
                            f"  [{stage}] epoch={epoch:02d} step={self.global_step} "
                            f"loss={step_loss:.4f} acc={step_acc:.4f} dt={dt:.1f}s"
                        )
                        t0 = time.time()

                    if self.writer:
                        self.writer.add_scalar("step/train_loss", step_loss, self.global_step)
                        self.writer.add_scalar("step/train_acc", step_acc, self.global_step)

        avg_loss = loss_sum / max(total, 1)
        avg_acc = correct / max(total, 1)
        return {"loss": float(avg_loss), "acc": float(avg_acc)}

    @torch.no_grad()
    def evaluate(self, model: nn.Module, loader: DataLoader, stage: str = "test") -> Dict[str, float]:
        model.to(self.device)
        state = torch.load(self.cfg.save_path, map_location=self.device)
        model.load_state_dict(state)
        model.eval()
        return self._run_epoch(model, loader, optimizer=None, epoch=0, stage=stage)

    def _maybe_checkpoint(self, model: nn.Module, val_m: Dict[str, float]) -> None:
        score = float(val_m["acc"])
        if score > self.best_val_acc:
            self.best_val_acc = score
            torch.save(model.state_dict(), self.cfg.save_path)
            print(f"Saved best checkpoint to {self.cfg.save_path} (val_acc={score:.4f})")

    def _log_epoch_metrics(self, epoch: int, train_m: Dict[str, float], val_m: Dict[str, float]) -> None:
        print(
            f"Epoch {epoch:02d}/{self.cfg.max_epochs:02d} | "
            f"train_loss={train_m['loss']:.4f} train_acc={train_m['acc']:.4f} | "
            f"val_loss={val_m['loss']:.4f} val_acc={val_m['acc']:.4f} | "
            f"best_val_acc={self.best_val_acc:.4f}"
        )

        if self.writer:
            self.writer.add_scalar("epoch/train_loss", train_m["loss"], epoch)
            self.writer.add_scalar("epoch/train_acc", train_m["acc"], epoch)
            self.writer.add_scalar("epoch/val_loss", val_m["loss"], epoch)
            self.writer.add_scalar("epoch/val_acc", val_m["acc"], epoch)
            self.writer.add_scalar("epoch/best_val_acc", self.best_val_acc, epoch)

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