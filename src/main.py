from torchvision import models, transforms

from cvpr_datamodule import CVPRDataModule, DataConfig
from garbage_classification import FitConfig, GarbageClassification

def main():

  cvprds = CVPRDataModule(
    DataConfig(data_dir=r'/Users/danishshahid/MEng/ENSF617/Assignments/Assignment2/garbage_data', 
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

  gc.fit(dataloaders=dataloaders)

  gc.evaluate(dataloader=dataloaders['test'])

if __name__ == '__main__':
  main()

