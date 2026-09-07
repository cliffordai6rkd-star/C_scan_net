"""项目配置。

所有训练参数集中放在这里，model 和 trainer 都接收同一个 Config 实例。
这样可以在 train.py 中统一修改参数，也可以在实验时复制 Config 创建不同配置。
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional
import json
import random

import torch


@dataclass
class DataConfig:
    image_dir: str = "dataset/all_c_scans_new"
    condition_dir: str = "dataset/impact_conditions"
    image_size: int = 224
    batch_size: int = 128
    num_workers: int = 0
    validation_ratio: float = 0.2
    thickness_per_ply_mm: float = 0.1875


@dataclass
class ModelConfig:
    # 当前可选 resnet18、resnet34、resnet50、dinov3。
    backbone: str = "resnet18"
    pretrained: bool = True
    dinov3_path: str = (
        "/home/rei/mnt/code/lcx/model/"
        "dinov3-vitb16-pretrain-lvd1689m"
    )
    freeze_backbone: bool = True
    hidden_dim: int = 128
    dropout: float = 0.007
    output_dim: int = 1


@dataclass
class OptimizerConfig:
# adam
# adamw
# sgd
# rmsprop
    name: str = "adamw"
    learning_rate: float = 5e-4
    weight_decay: float = 3e-4
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999


@dataclass
class SchedulerConfig:
    # plateau: 验证集指标不提升时，学习率乘以 factor（默认减半）。
    # cosine: 余弦退火。
    # constant: 学习率保持不变。
    name: str = "constant"
    factor: float = 0.5
    patience: int = 5
    min_lr: float = 1e-6
    t_max: Optional[int] = None


@dataclass
class TrainerConfig:
    epochs: int = 100
    device: str = "cuda:0"
    seed: int = 42
    loss: str = "mse"
    progress_bar: bool = True
    gradient_clip_norm: Optional[float] = 1.0
    use_amp: bool = False
    checkpoint_dir: str = "checkpoints"
    monitor: str = "val_loss"
    monitor_mode: str = "min"
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-6
    save_best_only: bool = False

    # wandb 默认关闭。打开前先安装 wandb 并完成登录。
    wandb_enabled: bool = True
    wandb_project: str = "c-scan-impact-energy"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def __post_init__(self) -> None:
        # seed 放在 trainer 中，保留一个顶层属性访问入口更方便。
        self.seed = self.trainer.seed

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, ensure_ascii=False, indent=2)


def resolve_device(device_name: str) -> torch.device:
    """将 auto/cpu/cuda 解析成实际的 torch.device。"""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "配置要求使用 CUDA，但当前 PyTorch 没有可用的 CUDA 设备。"
        )

    return torch.device(device_name)


def set_seed(seed: int) -> None:
    """固定常用随机源，保证同一配置下尽量复现实验。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
