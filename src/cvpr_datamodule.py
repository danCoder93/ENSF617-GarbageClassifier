from dataclasses import dataclass
from typing import Dict, Optional

from torchvision import transforms
from torch.utils.data import DataLoader

from cvpr_dataset import CVPR


@dataclass(frozen=True)
class DataConfig:
    data_dir: str
    inference_transform: transforms.Compose
    augmentation_transform: Optional[transforms.Compose] = None
    batch_size: int = 256
    num_workers: int = 2
    seed: int = 42


class CVPRDataModule:
    """
    Orchestrates datasets + dataloaders for the CVPR garbage classification dataset.
    """

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.class_names = None
        self.num_classes = None

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

    def setup(self) -> None:
        self.train_ds = CVPR(self.cfg.data_dir, split="train", transform=self.transforms["train"])
        self.val_ds = CVPR(self.cfg.data_dir, split="val", transform=self.transforms["val"])
        self.test_ds = CVPR(self.cfg.data_dir, split="test", transform=self.transforms["test"])

        # Assuming your CVPR wrapper exposes underlying ImageFolder as `.ds`
        self.class_names = list(self.train_ds.ds.classes)
        self.num_classes = len(self.class_names)

    def train_dataloader(self) -> DataLoader:
        self._ensure_setup()
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=True if self.cfg.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        self._ensure_setup()
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=True if self.cfg.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        self._ensure_setup()
        return DataLoader(
            self.test_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=True if self.cfg.num_workers > 0 else False,
        )

    def _ensure_setup(self) -> None:
        if self.train_ds is None or self.val_ds is None or self.test_ds is None:
            raise RuntimeError("DataModule not set up. Call .setup() before requesting dataloaders.")