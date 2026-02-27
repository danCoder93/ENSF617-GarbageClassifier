from typing import Optional
import os

from torchvision import datasets

from torch.utils.data import Dataset

class CVPR(Dataset):
  '''
  Dataset for garbage classification for city of Calgary

  Author - City of Calgary
  Provider - Prof. Roberto Souza
  '''

  def __init__(self, data_dir: str, split: str = 'train', transform:Optional[callable] = None):
      sub_dir = None
      match split.lower():
        case 'train':
          sub_dir = 'CVPR_2024_dataset_Train'
        case 'val':
          sub_dir = 'CVPR_2024_dataset_Val'
        case 'test':
          sub_dir = 'CVPR_2024_dataset_Test'
        case _:
          raise ValueError(f"split must be one of {{['train', 'val', 'test']}}, got '{split}'")
      
      full_path = os.path.join(data_dir, sub_dir)

      self.ds = datasets.ImageFolder(full_path, transform)
    
  def __len__(self):
    return len(self.ds)

  def __getitem__(self, idx):
    return self.ds[idx]

  