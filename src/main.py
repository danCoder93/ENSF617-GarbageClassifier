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

def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def make_writer(run_name: str) -> SummaryWriter:
    ts = time.strftime("%Y%m%d-%H%M%S")
    logdir = f"runs/{run_name}/{ts}"
    return SummaryWriter(log_dir=logdir)

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

def run_one(mode: str, data_dir: str, device: str):
    logger = PipelineLogger(writer=make_writer(run_name=mode))

    train_loader, val_loader, test_loader, num_classes = make_loaders(data_dir, device, mode)

    model = MultiModalClassifier(num_classes=num_classes, mode=mode)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)

    cfg = TrainConfig(
        device=device, 
        max_epochs=30, 
        logger=logger, 
        grad_clip_norm=1.0, 
        save_path=f"best_{mode}.pth", 
        use_amp=True)
    trainer = Trainer(cfg, loss_fn=nn.CrossEntropyLoss())

    trainer.fit(model, train_loader, val_loader, optimizer)
    test_metrics = trainer.evaluate(model, test_loader, stage="test")
    print(mode, "test:", test_metrics)

    logger.close()

def main():
    data_dir = r"/work/TALC/ensf617_2026w/garbage_data"
    device = pick_device()

    for mode in ["image", "text", "multimodal"]:
        run_one(mode, data_dir, device)

if __name__ == "__main__":
    main()
