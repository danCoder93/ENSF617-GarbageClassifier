# This file is all bout defining our model which of course has our 3 modes which we defined and have
# based our code around - image, text and then the main mode: Multi Modal which depicts the transfusion of both

#It is based off our datacollate file too, which then interprets for all different batches that collate can return

#Full referenc to CHATGPT for help with inspiration, debugging and implementation, however this reflects the spirit of our ideas
#along with initial attempts, debugging and research and reflects our teams hollistic vision on how tackle this project

#Also full reference to ENSF 617 content and Dr. De Souza, as we leveraged his examples heavily in all apsects of our code along with the tutorials
#and none of this would be possible without it, many portions were borrowed and augmented for our data set

#this file is the crux of our assignment.

#importing needed libraries
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoModel

#defining the modes
Mode = Literal["image", "text", "multimodal"]

#this function turns token embeddings into one sentence embedding
def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6) #captures how many tokens each has
    return summed / denom #computes our average embedding across all our tokens which was taken by denom -> will be critical for our DistilledBert

# This takes our image tensor and will return a feature vector but most importantly of fixed size!!! for our subsequent layers
class ImageEncoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        m = models.efficientnet_v2_m(weights="DEFAULT" if pretrained else None)
        self.features = m.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 1280

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)          # [B, C, H, W]
        x = self.pool(x).flatten(1)        # [B, C]
        return x #here we are outputing embedding from image

# This is coming from and augmented for our text classifer
class TextEncoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased") #using our pretrained bert
        self.out_dim = self.bert.config.hidden_size #returns hidden SIZE!!

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # DistilBERT has no pooler; use mean pooling (robust) or CLS token out.last_hidden_state[:,0]
        return mean_pool(out.last_hidden_state, attention_mask)

#Our classifier head class
class MLPHead(nn.Module):

    #mapping all our embedding from above into logits!!
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) #this is getting our logits which we need to pass to CrossEntropyLoss (ENSF 617 full reference)

#This our full multimodal classifier 
class MultiModalClassifier(nn.Module):
    #wrapper that sets the modes and calls the respectibe siper modes so that we instatiate all our modes,
    #independent of which mode is chosen (so we have the option of doing multimodal if called upon and that is the mode set)
    def __init__(
        self,
        num_classes: int,
        mode: Mode,
        freeze_image: bool = True,
        freeze_text: bool = False,
    ):
        super().__init__()
        self.mode = mode

        #both image and text instantiation
        self.img_enc = ImageEncoder(pretrained=True)
        self.txt_enc = TextEncoder(pretrained=True)

        #this fully from chatgpt as it depicted him for saving compute and avoiding overfiting
        #we did find this implementation helpful when running although still unconclusive
        if freeze_image:
            for p in self.img_enc.parameters():
                p.requires_grad = False

        if freeze_text:
            for p in self.txt_enc.parameters():
                p.requires_grad = False

        #this is choosing the classifier head again based on the mode
        if mode == "image":
            self.head = MLPHead(self.img_enc.out_dim, num_classes)
        elif mode == "text":
            self.head = MLPHead(self.txt_enc.out_dim, num_classes)
        elif mode == "multimodal":
            self.head = MLPHead(self.img_enc.out_dim + self.txt_enc.out_dim, num_classes)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    #connecting back to our data pipeline for each respective node
    def forward(self, batch):
        if self.mode == "image":
            img_f = self.img_enc(batch["image"])
            return self.head(img_f)

        if self.mode == "text":
            txt_f = self.txt_enc(batch["input_ids"], batch["attention_mask"])
            return self.head(txt_f)

        # multimodal
        img_f = self.img_enc(batch["image"])
        txt_f = self.txt_enc(batch["input_ids"], batch["attention_mask"])
        fused = torch.cat([img_f, txt_f], dim=1)
        return self.head(fused)