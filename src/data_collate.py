# This file also had substantial help from ChatGPT in terms of not only debugging but implementation and FULL REFERENCE TO IT
# we wanted clear reuasble files to help with processing and as we evolved our code from early stages to the final submission
# we need great files to rerun and help with data and this was created with our insight.

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import torch

#as we have our 3 modes - text, image and multimodal
Mode = Literal["image", "text", "multimodal"]

#creating the main class for our datacolate
@dataclass
class DataCollate:
    mode: Mode
    tokenizer: Optional[Any] = None
    max_length: int = 64

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:

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