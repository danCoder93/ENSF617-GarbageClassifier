from __future__ import annotations

from dataclasses import fields, is_dataclass
from math import isfinite
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter


class PipelineLogger:
    """Encapsulates tensorboard logging for end-to-end ML pipeline internals."""

    def __init__(self, writer: Optional[SummaryWriter] = None):
        self.writer = writer

    def log_run_start(
        self,
        cfg: Any,
        model: nn.Module,
        optimizer: Optimizer,
        train_batches: Optional[int] = None,
        val_batches: Optional[int] = None,
    ) -> None:
        if not self.writer:
            return

        if is_dataclass(cfg):
            cfg_dict = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
        else:
            cfg_dict = dict(cfg) if isinstance(cfg, dict) else {"config": str(cfg)}
        cfg_dict = {k: v for k, v in cfg_dict.items() if k != "logger"}

        self.writer.add_text("run/config", str(cfg_dict), 0)
        self.writer.add_text("run/model_class", model.__class__.__name__, 0)

        if train_batches is not None:
            self._add_scalar("run/train_batches_per_epoch", float(train_batches), 0)
        if val_batches is not None:
            self._add_scalar("run/val_batches_per_epoch", float(val_batches), 0)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._add_scalar("run/total_parameters", float(total_params), 0)
        self._add_scalar("run/trainable_parameters", float(trainable_params), 0)

        for idx, group in enumerate(optimizer.param_groups):
            lr = group.get("lr")
            if lr is not None:
                self._add_scalar(f"run/optimizer_group_{idx}_lr", float(lr), 0)

    def log_step(
        self,
        stage: str,
        step: int,
        metrics: Dict[str, float],
        internals: Dict[str, float],
        logits: Optional[torch.Tensor] = None,
        confidences: Optional[torch.Tensor] = None,
        log_histograms: bool = False,
    ) -> None:
        if not self.writer:
            return

        for name, value in metrics.items():
            self._add_scalar(f"step/{stage}_{name}", value, step)
        for name, value in internals.items():
            self._add_scalar(f"step/{stage}_{name}", value, step)

        if log_histograms and logits is not None:
            self.writer.add_histogram(f"step/{stage}_logits", logits.detach().float().cpu(), step)
        if log_histograms and confidences is not None:
            self.writer.add_histogram(f"step/{stage}_confidence", confidences.detach().float().cpu(), step)

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        best_val_acc: float,
    ) -> None:
        if not self.writer:
            return

        for name, value in train_metrics.items():
            self._add_scalar(f"epoch/train_{name}", value, epoch)
        for name, value in val_metrics.items():
            self._add_scalar(f"epoch/val_{name}", value, epoch)
        self._add_scalar("epoch/best_val_acc", best_val_acc, epoch)

    def log_checkpoint(self, epoch: int, save_path: str, score: float) -> None:
        if not self.writer:
            return

        self._add_scalar("checkpoint/best_val_acc", score, epoch)
        self.writer.add_text("checkpoint/latest_path", save_path, epoch)

    def log_evaluation(self, stage: str, metrics: Dict[str, float], step: int = 0) -> None:
        if not self.writer:
            return

        for name, value in metrics.items():
            self._add_scalar(f"evaluation/{stage}_{name}", value, step)

    def flush(self) -> None:
        if self.writer:
            self.writer.flush()

    def close(self) -> None:
        if self.writer:
            self.writer.close()

    def _add_scalar(self, tag: str, value: float, step: int) -> None:
        if not self.writer:
            return
        if value is None:
            return
        value = float(value)
        if isfinite(value):
            self.writer.add_scalar(tag, value, step)
