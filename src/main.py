import sys
from torchvision import models, transforms

from cvpr_datamodule import CVPRDataModule, DataConfig
from garbage_classification import FitConfig, GarbageClassification

def main():
  args = sys.argv[1:]
  default_dir: str =  r'/work/TALC/ensf617_2026w/garbage_data'
  data_dir = args[0] if len(args) > 0 else default_dir
  cvprds = CVPRDataModule(
    DataConfig(data_dir=data_dir, 
               inference_transform=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1.transforms(), 
               augmentation_transform=transforms.Compose(
                 [transforms.RandomRotation(45), transforms.RandomHorizontalFlip()]
                 )))
  
  cvprds.setup()
  
  # Define data loaders
  dataloaders = {
      "train": cvprds.train_dataloader(),
      "val": cvprds.val_dataloader(),
      "test": cvprds.test_dataloader(),
  }

  gc = GarbageClassification(FitConfig())

  gc.train(dataloaders=dataloaders)

  gc.eval(dataloader=dataloaders['test'])

if __name__ == '__main__':
  main()

