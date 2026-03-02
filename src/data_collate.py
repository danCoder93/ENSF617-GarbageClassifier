# src/data_collate.py
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import torch

Mode = Literal["image", "text", "multimodal"]

@dataclass
class DataCollate:
    mode: Mode
    tokenizer: Optional[Any] = None
    max_length: int = 64

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # batch items come from CVPR.__getitem__:
        # {"image": img_tensor, "text": str, "label": int, "path": str}
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        paths = [b["path"] for b in batch]

        out: Dict[str, Any] = {"labels": labels, "path": paths}

        if self.mode in ("image", "multimodal"):
            images = torch.stack([b["image"] for b in batch], dim=0)
            out["image"] = images

        if self.mode in ("text", "multimodal"):
            if self.tokenizer is None:
                raise ValueError("tokenizer must be provided for text/multimodal mode")
            texts = [b["text"] for b in batch]
            out['raw_text'] = texts
            enc = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            out.update(enc)  # input_ids, attention_mask (and token_type_ids if exists)

        return out