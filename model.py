"""C-Scan 图像回归模型。"""

from pathlib import Path
from typing import Dict, Type

import torch
from torch import nn
from torchvision import models


class CScanRegressor(nn.Module):
    """使用 ResNet 提取 C-Scan 特征并回归 Impact Energy。

    参数通过 config.model 透传：

    - backbone: resnet18/resnet34/resnet50/dinov3
    - pretrained: 是否加载 ImageNet 权重
    - dinov3_path: 本地 Hugging Face DINOv3 模型目录
    - freeze_backbone: 是否冻结特征提取器，只训练回归头
    - hidden_dim: 回归头隐藏层宽度；设为 0 时使用线性回归头
    - dropout: 回归头 dropout 比例
    - output_dim: 输出维度，当前单任务应为 1
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        model_config = config.model
        backbone_name = model_config.backbone.lower()
        self.backbone_type = backbone_name

        if backbone_name in {"dinov3", "dinov3-vitb16", "dinov3_vitb16"}:
            self.backbone = self._load_dinov3(model_config.dinov3_path)
            feature_dim = int(self.backbone.config.hidden_size)
        else:
            self.backbone, feature_dim = self._load_resnet(
                backbone_name,
                pretrained=model_config.pretrained,
            )

        hidden_dim = int(model_config.hidden_dim)
        output_dim = int(model_config.output_dim)
        dropout = float(model_config.dropout)

        if hidden_dim < 0:
            raise ValueError("config.model.hidden_dim 不能小于 0")
        if output_dim <= 0:
            raise ValueError("config.model.output_dim 必须大于 0")
        if output_dim != 1:
            raise ValueError(
                "当前 dataloader 只有 Impact Energy 一个标签，"
                "因此 config.model.output_dim 必须为 1。"
                "将来补充 CAI Strength 标签后再扩展为多任务输出。"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("config.model.dropout 必须位于 [0, 1)")

        # hidden_dim=0 时不增加隐藏层，便于做简单线性回归基线。
        if hidden_dim > 0:
            self.regression_head = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.regression_head = nn.Linear(feature_dim, output_dim)

        if model_config.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    @staticmethod
    def _load_resnet(
        backbone_name: str,
        pretrained: bool,
    ):
        constructors: Dict[str, Type[nn.Module]] = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
        }
        weight_defaults = {
            "resnet18": models.ResNet18_Weights.DEFAULT,
            "resnet34": models.ResNet34_Weights.DEFAULT,
            "resnet50": models.ResNet50_Weights.DEFAULT,
        }

        if backbone_name not in constructors:
            supported = ", ".join(
                sorted((*constructors.keys(), "dinov3"))
            )
            raise ValueError(
                f"不支持的 backbone: {backbone_name}，可选：{supported}"
            )

        # pretrained=False 是默认值，避免在无网络环境中自动下载权重。
        backbone = constructors[backbone_name](
            weights=(
                weight_defaults[backbone_name]
                if pretrained
                else None
            )
        )
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, feature_dim

    @staticmethod
    def _load_dinov3(model_path: str) -> nn.Module:
        """从本地目录加载 Hugging Face 格式的 DINOv3。"""
        model_path = str(Path(model_path).expanduser())
        if not Path(model_path).is_dir():
            raise FileNotFoundError(
                f"DINOv3 本地目录不存在: {model_path}"
            )

        try:
            from transformers import AutoModel
        except ImportError as error:
            raise ImportError(
                "使用 DINOv3 需要安装 transformers。"
                "请运行: pip install transformers>=4.56.0"
            ) from error

        try:
            return AutoModel.from_pretrained(
                model_path,
                local_files_only=True,
            )
        except OSError as error:
            raise OSError(
                f"无法从本地目录加载 DINOv3: {model_path}。"
                "请确认目录包含 config.json 和 model.safetensors。"
            ) from error

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """返回形状为 [batch_size, output_dim] 的预测值。"""
        if self.backbone_type in {
            "dinov3",
            "dinov3-vitb16",
            "dinov3_vitb16",
        }:
            # DINOv3 默认训练尺寸为 224。开启位置编码插值后，
            # 将来把 config.data.image_size 改成其他尺寸也能工作。
            outputs = self.backbone(
                pixel_values=images,
                interpolate_pos_encoding=True,
            )
            features = getattr(outputs, "pooler_output", None)
            if features is None:
                # 某些 transformers 版本不返回 pooler_output，退回 class token。
                features = outputs.last_hidden_state[:, 0]
        else:
            features = self.backbone(images)
        return self.regression_head(features)
