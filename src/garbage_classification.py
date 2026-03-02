# src/models_multimodal.py
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoModel

Mode = Literal["image", "text", "multimodal"]

def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden_state: [B, T, H], mask: [B, T]
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B,T,1]
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom

class ImageEncoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        m = models.efficientnet_v2_m(weights="DEFAULT" if pretrained else None)
        self.features = m.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 1280

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)          # [B, C, H, W]
        x = self.pool(x).flatten(1)        # [B, C]
        return x

# adaptation from Marley Cheema
class TextEncoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")
        self.out_dim = self.bert.config.hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # DistilBERT has no pooler; use mean pooling (robust) or CLS token out.last_hidden_state[:,0]
        return mean_pool(out.last_hidden_state, attention_mask)

class MLPHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MultiModalClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        mode: Mode,
        freeze_image: bool = True,
        freeze_text: bool = False,
    ):
        super().__init__()
        self.mode = mode

        self.img_enc = ImageEncoder(pretrained=True)
        self.txt_enc = TextEncoder(pretrained=True)

        if freeze_image:
            for p in self.img_enc.parameters():
                p.requires_grad = False

        if freeze_text:
            for p in self.txt_enc.parameters():
                p.requires_grad = False

        if mode == "image":
            self.head = MLPHead(self.img_enc.out_dim, num_classes)
        elif mode == "text":
            self.head = MLPHead(self.txt_enc.out_dim, num_classes)
        elif mode == "multimodal":
            self.head = MLPHead(self.img_enc.out_dim + self.txt_enc.out_dim, num_classes)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, batch):
        if self.mode == "image":
            img_f = self.img_enc(batch["image"])
            return self.head(img_f)

        if self.mode == "text":
            txt_f = self.txt_enc(batch["input_ids"], batch["attention_mask"])
            return self.head(txt_f)

        # multimodal
        img_f = self.img_enc(batch["image"])
        txt_f = self.txt_enc(batch["input_ids"], batch["attention_mask"])
        fused = torch.cat([img_f, txt_f], dim=1)
        return self.head(fused)