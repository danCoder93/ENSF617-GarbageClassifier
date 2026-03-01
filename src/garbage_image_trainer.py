from dataclasses import dataclass
from typing import Any, Dict, Optional

import time

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


@dataclass(frozen=True)
class TrainConfig:
    device: str
    max_epochs: int = 5
    writer: Optional[SummaryWriter] = None
    grad_clip_norm: Optional[float] = None
    save_path: str = "best_model.pth"
    monitor: str = "val_acc"  # which metric decides "best"


@dataclass(frozen=True)
class TrainStats:
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


class GarbageImageTrainer:
    def __init__(self, train_config: TrainConfig):
        self.max_epochs = train_config.max_epochs
        self.device = torch.device(train_config.device)
        self.writer = train_config.writer
        self.grad_clip_norm = train_config.grad_clip_norm
        self.save_path = train_config.save_path

        self.monitor = train_config.monitor
        self.best_score = float("-inf")
        self.global_step = 0

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
    ) -> None:
        model.to(self.device)

        for epoch in range(1, self.max_epochs + 1):
            train_stats = self._run_train_epoch(model, train_loader, optimizer, epoch)
            val_stats = self._run_val_epoch(model, val_loader, epoch)

            stats = TrainStats(
                train_loss=train_stats["loss"],
                train_acc=train_stats["acc"],
                val_loss=val_stats["loss"],
                val_acc=val_stats["acc"],
            )

            print(
                f"Epoch {epoch:02d}/{self.max_epochs:02d} | "
                f"train_loss={stats.train_loss:.4f} train_acc={stats.train_acc:.4f} | "
                f"val_loss={stats.val_loss:.4f} val_acc={stats.val_acc:.4f}"
            )

            if self.writer:
                self.writer.add_scalar("epoch/train_loss", stats.train_loss, epoch)
                self.writer.add_scalar("epoch/train_acc", stats.train_acc, epoch)
                self.writer.add_scalar("epoch/val_loss", stats.val_loss, epoch)
                self.writer.add_scalar("epoch/val_acc", stats.val_acc, epoch)

            # select metric to monitor
            score = getattr(stats, self.monitor)
            if self._is_best(score):
                self.best_score = score
                torch.save(model.state_dict(), self.save_path)
                print(f"Saved new best checkpoint to {self.save_path} ({self.monitor}={score:.4f})")

            if self.writer:
                self.writer.flush()

    def _run_train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: Optimizer,
        epoch: int,
    ) -> Dict[str, float]:
        model.train()

        running_loss = 0.0
        running_acc = 0.0
        n = 0

        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            x, y = batch
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            out: Dict[str, Any] = model.training_step((x, y))
            loss = out["loss"]
            acc = out["acc"]
            bs = out["batch_size"]

            loss.backward()

            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_norm)

            optimizer.step()

            running_loss += loss.item() * bs
            running_acc += acc.item() * bs
            n += bs

            self.global_step += 1

            if self.writer:
                self.writer.add_scalar("step/train_loss", loss.item(), self.global_step)
                self.writer.add_scalar("step/train_acc", acc.item(), self.global_step)

            # optional: occasional print instead of every step
            if batch_idx % 10 == 0:
                dt = time.time() - t0
                print(f"  epoch={epoch:02d} step={self.global_step} loss={loss.item():.4f} acc={acc.item():.4f} dt={dt:.1f}s")
                t0 = time.time()

        return {"loss": running_loss / max(n, 1), "acc": running_acc / max(n, 1)}

    @torch.no_grad()
    def _run_val_epoch(self, model: nn.Module, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        model.eval()

        running_loss = 0.0
        running_acc = 0.0
        n = 0

        for batch in val_loader:
            x, y = batch
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            out = model.validation_step((x, y))
            loss = out["loss"]
            acc = out["acc"]
            bs = out["batch_size"]

            running_loss += loss.item() * bs
            running_acc += acc.item() * bs
            n += bs

            if self.writer:
                self.writer.add_scalar("step/val_loss", loss.item(), self.global_step)
                self.writer.add_scalar("step/val_acc", acc.item(), self.global_step)

        return {"loss": running_loss / max(n, 1), "acc": running_acc / max(n, 1)}

    @torch.no_grad()
    def evaluate(self, model: nn.Module, loader: DataLoader) -> Dict[str, float]:
        model.eval()
        self.model.load_state_dict(torch.load(self.save_path, map_location=self.device))
        running_loss = 0.0
        running_acc = 0.0
        n = 0

        for batch in loader:
            x, y = batch
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            out = model.validation_step((x, y))
            loss = out["loss"]
            acc = out["acc"]
            bs = out["batch_size"]

            running_loss += loss.item() * bs
            running_acc += acc.item() * bs
            n += bs

        return {"loss": running_loss / max(n, 1), "acc": running_acc / max(n, 1)}

    def _is_best(self, score: float) -> bool:
        return score > self.best_score