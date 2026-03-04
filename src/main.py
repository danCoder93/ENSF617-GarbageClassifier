#Full referenc to CHATGPT for help with inspiration, debugging and implementation, however this reflects the spirit of our ideas
#along with initial attempts, debugging and research and reflects our teams hollistic vision on how tackle this project

#Also full reference to ENSF 617 content and Dr. De Souza, as we leveraged his examples heavily in all apsects of our code along with the tutorials
#and none of this would be possible without it, many portions were borrowed and augmented for our data set

#this is our command line and of course main file, which is will execute our evaluation of one of the 3 modes we have defined - text, image, classifications

import sys
import os
import time

import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from data_module import DataConfig, DataModule
from pipeline_logger import PipelineLogger
from trainer import Trainer, TrainConfig

from garbage_classification import MultiModalClassifier

#giving the option of which device to run through - found was huge for running down the road as it became so easy to define for validation down the road

def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

#create a TensorBoard and unique folder for all our runs
def make_writer(run_name: str) -> SummaryWriter:
    ts = time.strftime("%Y%m%d-%H%M%S")
    logdir = f"../runs/{run_name}/{ts}"
    return SummaryWriter(log_dir=logdir)

#this function is what connects back to the development of our data module file of which we create DataLoaders for test/val and train when called upon to run our ML model!!!
def make_loaders(data_dir: str, device:str, mode: str, tokenizer_name="distilbert-base-uncased", batch_size=32, num_workers=2, max_len=64):

    inference_tf = models.EfficientNet_V2_M_Weights.IMAGENET1K_V1.transforms()

    aug_tf = transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(),
    ])

    cfg = DataConfig(
        data_dir=data_dir,
        mode=mode,
        device=device,
        inference_transform=inference_tf,
        augmentation_transform=aug_tf,
        tokenizer_name=tokenizer_name,
        batch_size=batch_size,
        num_workers=num_workers,
        max_length=max_len
    )

    dm = DataModule(cfg)
    dm.setup()

    return dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader(), dm.num_class

#this is our single run expirement, which of course creates all the componets, the logger which calls from our pipelinelogger
#then our multimodalclassifer which again allows for modes and builds our models and then finally creating our traininhg configuration
#wthin that configuration is our ability to set EPOCHs which we overwrote to 20, has the tensorboard logging baked in the saving of the best model
#then creating the trainer with the crossEntropyLoss (coming from ENSF 617), training and evaluation
def run_one(mode: str, data_dir: str, device: str):
    logger = PipelineLogger(writer=make_writer(run_name=mode))

    train_loader, val_loader, test_loader, num_classes = make_loaders(data_dir, device, mode)

    model = MultiModalClassifier(num_classes=num_classes, mode=mode)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)

    cfg = TrainConfig(
        device=device, 
        max_epochs=20, 
        logger=logger, 
        grad_clip_norm=1.0, 
        save_path=f"../weights/best_{mode}.pth", 
        use_amp=True)
    trainer = Trainer(cfg, loss_fn=nn.CrossEntropyLoss())

    trainer.fit(model, train_loader, val_loader, optimizer)
    test_metrics = trainer.evaluate(model, test_loader, stage="test")
    print(mode, "test:", test_metrics)

    logger.close()

#and now finally our main, which executes our interface for a good user experiences and the runs!!
def main():
    run_modes = ["image", "text", "multimodal"]
    arg_mode = "multimodal"
    if len(sys.argv) >= 2:
        arg_mode = sys.argv[1]
        if arg_mode.lower() not in run_modes:
            raise ValueError(f"mode must be one of {run_modes}, got '{arg_mode}'")
    
    data_dir = r"/work/TALC/ensf617_2026w/garbage_data"
    device = pick_device()

    run_one(arg_mode, data_dir, device)

if __name__ == "__main__":
    main()
