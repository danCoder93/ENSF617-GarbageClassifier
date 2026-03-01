from typing import Any, Dict, Optional
import os, re

from torchvision.datasets import ImageFolder

from torch.utils.data import Dataset


def filename_to_text(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0].lower().replace("_", " ")
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem

def split_to_subdir(split: str) -> str:
    s = split.lower()
    if s == "train":
        return "CVPR_2024_dataset_Train"
    if s == "val":
        return "CVPR_2024_dataset_Val"
    if s == "test":
        return "CVPR_2024_dataset_Test"
    raise ValueError(f"split must be one of ['train','val','test'], got '{split}'")

class CVPR(Dataset):
  '''
  Dataset for garbage classification for city of Calgary

  Author - City of Calgary
  Provider - Prof. Roberto Souza
  '''

  def __init__(self, data_dir: str, split: str = 'train', image_transform:Optional[callable] = None):
    sub_dir = split_to_subdir(split)
    path = os.path.join(data_dir, sub_dir)
    self.ds = ImageFolder(path, transform=image_transform)
    self.samples = self.ds.samples  # list[(path, class_idx)]
    self.classes = self.ds.classes
    self.num_classes = len(self.classes)
      
    
  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx: int) -> Dict[str, Any]:
    # ImageFolder already loads + transforms the image
    img, y = self.ds[idx]
    path, _ = self.samples[idx]
    text = filename_to_text(path)
    return {"image": img, "text": text, "label": y, "path": path}

  