import torch # Pytorch
from torchvision import models # Pytorch vision library for import vision models

import torch.nn as nn # neural net library

def build_efficientnet_head(num_classes: int):
    model = models.efficientnet_v2_m(weights="DEFAULT")
    for p in model.features.parameters():
        p.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"