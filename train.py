"""训练入口。

示例：
    python train.py
    python train.py --epochs 30 --batch-size 8 --scheduler cosine
    python train.py --optimizer adam --lr 0.0001 --wandb
    python train.py --backbone dinov3 --freeze-backbone
"""

import argparse

import torch
from torch.utils.data import DataLoader

from config import Config, set_seed
from dataloader import build_datasets
from model import CScanRegressor
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a C-Scan Impact Energy regressor"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--optimizer",
        choices=["adam", "adamw", "sgd", "rmsprop"],
        default=None,
    )
    parser.add_argument(
        "--scheduler",
        choices=["plateau", "cosine", "constant"],
        default=None,
    )
    parser.add_argument(
        "--backbone",
        choices=["resnet18", "resnet34", "resnet50", "dinov3"],
        default=None,
    )
    parser.add_argument("--dinov3-path", default=None)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto、cpu 或 cuda")
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def apply_cli_overrides(config: Config, args) -> Config:
    if args.epochs is not None:
        config.trainer.epochs = args.epochs
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.lr is not None:
        config.optimizer.learning_rate = args.lr
    if args.optimizer is not None:
        config.optimizer.name = args.optimizer
    if args.scheduler is not None:
        config.scheduler.name = args.scheduler
    if args.backbone is not None:
        config.model.backbone = args.backbone
    if args.dinov3_path is not None:
        config.model.dinov3_path = args.dinov3_path
    if args.freeze_backbone:
        config.model.freeze_backbone = True
    if args.hidden_dim is not None:
        config.model.hidden_dim = args.hidden_dim
    if args.device is not None:
        config.trainer.device = args.device
    if args.wandb:
        config.trainer.wandb_enabled = True
    return config


def main():
    args = parse_args()
    config = apply_cli_overrides(Config(), args)
    set_seed(config.trainer.seed)

    train_dataset, validation_dataset = build_datasets(
        image_dir=config.data.image_dir,
        condition_dir=config.data.condition_dir,
        image_size=config.data.image_size,
        validation_ratio=config.data.validation_ratio,
        seed=config.trainer.seed,
        thickness_per_ply_mm=config.data.thickness_per_ply_mm,
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=pin_memory,
    )

    model = CScanRegressor(config)
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        validation_loader=validation_loader,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
