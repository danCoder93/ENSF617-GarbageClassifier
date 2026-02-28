from dataclasses import dataclass
from typing import Dict

from torchvision import transforms
from torch.utils.data import DataLoader

from cvpr_dataset import CVPR

@dataclass(frozen=True)
class DataConfig():
  data_dir: str
  inference_transform: transforms
  augmentation_transform: transforms.Compose
  batch_size: int = 128
  num_workers: int = 2
  pin_memory: bool = True
  persistent_workers: bool = True
  seed: int = 42

class CVPRDataModule:
    """
    Orchestrates datasets + dataloaders for the CVPR garbage classification dataset.

    Responsibilities:
      - decide transforms for train/val/test
      - instantiate CVPR datasets
      - provide DataLoaders
      - expose useful metadata (class names, num_classes)
      - expose dataset exploration methods
    """

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg

        # Will be set in setup()
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.class_names = None
        self.num_classes = None

        # Transforms decided once here (or you can decide inside setup()).
        self.transforms = self._build_transforms()

    def _build_transforms(self) -> Dict[str, transforms.Compose]:
        """
        Define stage-specific transforms.
        Use torchvision's weights transforms to match EfficientNet/MobileNet preprocessing.
        """
        # Pick the same weights family as your model to ensure correct normalization.
        return {'train': transforms.Compose([self.cfg.augmentation_transform, self.cfg.inference_transform]),
                'val' : self.cfg.inference_transform,
                'test': self.cfg.inference_transform}

    def setup(self) -> None:
        """
        Create dataset objects. Call once before requesting dataloaders.
        """
        self.train_ds = CVPR(self.cfg.data_dir, split="train", transform=self.transforms["train"])
        self.val_ds = CVPR(self.cfg.data_dir, split="val", transform=self.transforms["val"])
        self.test_ds = CVPR(self.cfg.data_dir, split="test", transform=self.transforms["test"])

        # Expose metadata from underlying ImageFolder
        self.class_names = list(self.train_ds.ds.classes)
        self.num_classes = len(self.class_names)

    def train_dataloader(self) -> DataLoader:
        self._ensure_setup()
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers and self.cfg.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        self._ensure_setup()
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers and self.cfg.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        self._ensure_setup()
        return DataLoader(
            self.test_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers and self.cfg.num_workers > 0,
        )

    def _ensure_setup(self) -> None:
        if self.train_ds is None or self.val_ds is None or self.test_ds is None:
            raise RuntimeError("DataModule not set up. Call .setup() before requesting dataloaders.")


