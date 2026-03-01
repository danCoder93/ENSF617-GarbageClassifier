import sys
from torchvision import models, transforms

import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from cvpr_datamodule import CVPRDataModule, DataConfig
from garbage_classification import pick_device, build_efficientnet_head
from garbage_image_classification import GarbageImageClassification
from garbage_image_trainer import TrainConfig, GarbageImageTrainer


def main():
    args = sys.argv[1:]
    default_dir: str = r"/work/TALC/ensf617_2026w/garbage_data"
    data_dir = args[0] if len(args) > 0 else default_dir

    inference_tf = models.EfficientNet_V2_M_Weights.IMAGENET1K_V1.transforms()
    aug_tf = transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(),
    ])

    data_cfg = DataConfig(
        data_dir=data_dir,
        inference_transform=inference_tf,
        augmentation_transform=aug_tf,
        batch_size=128,
        num_workers=2,
    )

    dm = CVPRDataModule(data_cfg)
    dm.setup()

    num_classes = dm.num_classes
    backbone = build_efficientnet_head(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    model = GarbageImageClassification(backbone, criterion)

    optimizer = optim.SGD(backbone.parameters(), lr=1e-3)

    writer = SummaryWriter("runs/garbage_classification")
    trainer = GarbageImageTrainer(
        train_config=TrainConfig(
            device=pick_device(),
            max_epochs=5,
            writer=writer,
            grad_clip_norm=1.0,
            save_path="best_model.pth",
            monitor="val_acc",
        )
    )

    trainer.train(
        model,
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(),
        optimizer=optimizer)

    trainer.evaluate(
        model,
        dm.test_dataloader)
    writer.close()


if __name__ == "__main__":
    main()