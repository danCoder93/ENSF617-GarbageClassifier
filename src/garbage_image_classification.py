from typing import Any, Dict

import torch
import torch.nn as nn


class GarbageImageClassification(nn.Module):
    """
    Simple wrapper that:
      - holds a backbone
      - holds a criterion (loss)
      - exposes training_step/validation_step like Lightning, but pure PyTorch
    """

    def __init__(self, backbone: nn.Module, criterion: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def _shared_step(self, batch: Any) -> Dict[str, Any]:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        return {"loss": loss, "acc": acc, "batch_size": x.size(0)}

    def training_step(self, batch: Any) -> Dict[str, Any]:
        return self._shared_step(batch)

    @torch.no_grad()
    def validation_step(self, batch: Any) -> Dict[str, Any]:
        return self._shared_step(batch)