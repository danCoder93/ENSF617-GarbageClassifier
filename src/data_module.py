from dataclasses import dataclass
from typing import Dict, Literal, Optional

from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from cvpr_dataset import CVPR
from data_collate import DataCollate

Mode = Literal["image", "text", "multimodal"]

@dataclass(frozen=True)
class DataConfig:
    data_dir: str
    device: str
    inference_transform: transforms.Compose
    augmentation_transform: Optional[transforms.Compose] = None
    batch_size: int = 32
    num_workers: int = 2
    seed: int = 42
    mode: Mode = "image"
    tokenizer_name: str = "distilbert-base-uncased"
    max_length: int = 64

class DataModule:
    """
    Orchestrates datasets + dataloaders for the CVPR garbage classification dataset.
    """
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.num_class = None

        self.tokenizer = None
        if self.cfg.mode in ("text", "multimodal"):
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer_name)

        self.is_cuda = str(self.cfg.device).startswith("cuda")

        self.collate_fn = self._collate_fn()

        self.transforms = self._build_transforms()

    def _build_transforms(self) -> Dict[str, transforms.Compose]:
        if self.cfg.augmentation_transform is None:
            train_tf = self.cfg.inference_transform
        else:
            # Safe: Compose can contain another Compose
            train_tf = transforms.Compose([self.cfg.augmentation_transform, self.cfg.inference_transform])

        return {
            "train": train_tf,
            "val": self.cfg.inference_transform,
            "test": self.cfg.inference_transform,
        }
    
    def _collate_fn(self):
        return DataCollate(
        mode=self.cfg.mode,
        tokenizer=self.tokenizer,
        max_length=self.cfg.max_length)

    def setup(self) -> None:
        self.train_ds = CVPR(self.cfg.data_dir, split="train", image_transform=self.transforms["train"])
        self.val_ds = CVPR(self.cfg.data_dir, split="val", image_transform=self.transforms["val"])
        self.test_ds = CVPR(self.cfg.data_dir, split="test", image_transform=self.transforms["test"])
        self.num_class = self.train_ds.num_classes

    def train_dataloader(self) -> DataLoader:
        self._ensure_setup()
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.is_cuda,
            persistent_workers=self.cfg.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        self._ensure_setup()
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.is_cuda,
            persistent_workers=self.cfg.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        self._ensure_setup()
        return DataLoader(
            self.test_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.is_cuda,
            persistent_workers=self.cfg.num_workers > 0,
        )

    def _ensure_setup(self) -> None:
        if self.train_ds is None or self.val_ds is None or self.test_ds is None:
            raise RuntimeError("DataModule not set up. Call .setup() before requesting dataloaders.")