from typing import Type
from dataclasses import dataclass

import torch # Pytorch
from torchvision import models # Pytorch vision library for import vision models

import torch.nn as nn # neural net library

import torch.optim as optim # optimizer library

@dataclass(frozen=True)
class FitConfig():
  loss_cls: Type[nn.Module] = nn.CrossEntropyLoss
  optimizer_cls: Type[optim.Optimizer] = optim.SGD
  output_classes: int = 4
  epochs: int = 5
  lr : float = 1e-3
  device: str = 'mps' if torch.backends.mps.is_available() else 'cuda:0' if torch.cuda.is_available() else 'cpu'
  verbose: bool = True

class GarbageClassification():

  def __init__(self, cfg: FitConfig):

    self.cfg = cfg

    self.device_ = torch.device(self.cfg.device)

    self.model = models.efficientnet_v2_m(weights='DEFAULT')

    # Freeze all layers except classifier
    for param in self.model.features.parameters():
      param.requires_grad = False

    # custom classifier
    num_features_classify = self.model.classifier[1].in_features
    # replace existing
    self.model.classifier[1] = nn.Linear(num_features_classify, self.cfg.output_classes)

    # move to device
    self.model = self.model.to(self.device_)

    # initialize loss function
    self.criterion = self.cfg.loss_cls()

    # initialize optimizer
    self.optimizer = self.cfg.optimizer_cls(self.model.classifier[1].parameters(), lr = self.cfg.lr)

  def fit(self, dataloaders):
    best_acc = 0.0
    
    # run full pass for epochs
    for epoch in range(self.cfg.epochs):

      print(f"Epoch {epoch + 1}/{self.cfg.epochs}")

      # different training for different phase
      for phase in ['train', 'val']:

        # if phase is train, enter in train mode
        if (phase == 'train'):
          print(f'Training...')
          self.model.train()
        else: # else get in eval mode - freezing parameters
          print(f'Validating...')
          self.model.eval()

        running_loss = 0.0
        running_corrects = 0

        print(f'Moving inputs and labels to device')
        # move inputs and labels to device
        for inputs, labels in dataloaders[phase]:
          inputs, labels = inputs.to(self.device_), labels.to(self.device_)

          # zero gradient before each run
          self.optimizer.zero_grad()

          with torch.set_grad_enabled(phase == 'train'):
            print(f'Executing forward pass...')
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            if phase == 'train':
              print(f'Executing backpropation')
              loss.backward()
              self.optimizer.step()

          running_loss += loss.item() * inputs.size(0)
          running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

        print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
              
        if phase == "val" and epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(self.model.state_dict(), "best_model.pth")

    print(f"Best val Acc: {best_acc:.4f}")

  # Test function
  def evaluate(self, dataloader):
      self.model.load_state_dict(torch.load("best_model.pth", map_location=self.device_))
      self.model.eval()
      correct = 0
      total = 0
      with torch.no_grad():
          for inputs, labels in dataloader:
              inputs, labels = inputs.to(self.device_), labels.to(self.device_)
              outputs = self.model(inputs)
              _, preds = torch.max(outputs, 1)
              correct += torch.sum(preds == labels).item()
              total += labels.size(0)
      print(f"Test Accuracy: {100 * correct / total:.2f}%")


