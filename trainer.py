"""训练、验证、指标、checkpoint 和可选 wandb 日志。"""

from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
import math

import torch
from torch import nn
from torch.optim import Adam, AdamW, RMSprop, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, ReduceLROnPlateau
from tqdm.auto import tqdm

from config import Config, resolve_device


def build_optimizer(model: nn.Module, config: Config) -> torch.optim.Optimizer:
    """根据配置创建优化器。"""
    optimizer_config = config.optimizer
    name = optimizer_config.name.lower()
    # 冻结 backbone 时只把可训练参数交给优化器，避免无意义地维护冻结参数状态。
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("模型没有可训练参数，请关闭 freeze_backbone 或检查模型配置。")

    common = {
        "lr": optimizer_config.learning_rate,
        "weight_decay": optimizer_config.weight_decay,
    }

    if name == "adam":
        return Adam(
            parameters,
            **common,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
        )
    if name == "adamw":
        return AdamW(
            parameters,
            **common,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
        )
    if name == "sgd":
        return SGD(
            parameters,
            **common,
            momentum=optimizer_config.momentum,
        )
    if name == "rmsprop":
        return RMSprop(
            parameters,
            **common,
            momentum=optimizer_config.momentum,
        )

    raise ValueError(
        f"不支持的优化器: {name}，可选：adam、adamw、sgd、rmsprop"
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Config,
):
    """根据配置创建学习率调度器。

    plateau 在验证集 loss 连续若干轮不下降时将学习率乘以 0.5；
    cosine 按 epoch 做余弦退火；constant 始终保持初始学习率。
    """
    scheduler_config = config.scheduler
    name = scheduler_config.name.lower()

    if name in {"plateau", "reduce_on_plateau", "reduce"}:
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_config.factor,
            patience=scheduler_config.patience,
            min_lr=scheduler_config.min_lr,
        )

    if name == "cosine":
        t_max = scheduler_config.t_max or config.trainer.epochs
        return CosineAnnealingLR(
            optimizer,
            T_max=max(1, t_max),
            eta_min=scheduler_config.min_lr,
        )

    if name in {"constant", "none", "fixed"}:
        return LambdaLR(optimizer, lr_lambda=lambda _epoch: 1.0)

    raise ValueError(
        f"不支持的 scheduler: {name}，可选：plateau、cosine、constant"
    )


def build_loss(name: str) -> nn.Module:
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "mae" or name == "l1":
        return nn.L1Loss()
    if name == "smooth_l1":
        return nn.SmoothL1Loss()
    raise ValueError("loss 可选：mse、mae/l1、smooth_l1")


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """用原生 PyTorch 计算回归指标，避免依赖 sklearn。"""
    predictions = predictions.detach().float().reshape(-1)
    targets = targets.detach().float().reshape(-1)
    errors = predictions - targets

    mse = torch.mean(errors.square()).item()
    mae = torch.mean(errors.abs()).item()
    target_mean = torch.mean(targets)
    total_sum_of_squares = torch.sum((targets - target_mean).square())
    residual_sum_of_squares = torch.sum(errors.square())

    if total_sum_of_squares.item() == 0.0:
        r2 = 0.0
    else:
        r2 = (1.0 - residual_sum_of_squares / total_sum_of_squares).item()

    return {
        "mse": mse,
        "rmse": math.sqrt(max(mse, 0.0)),
        "mae": mae,
        "r2": r2,
    }


class EarlyStopping:
    """监控一个验证指标，连续 patience 轮没有改善时停止训练。"""

    def __init__(
        self,
        patience: int,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        self.patience = max(0, int(patience))
        self.min_delta = float(min_delta)
        self.mode = mode.lower()
        if self.mode not in {"min", "max"}:
            raise ValueError("early stopping 的 mode 只能是 min 或 max")
        self.best: Optional[float] = None
        self.bad_epochs = 0

    def update(self, value: float) -> Tuple[bool, bool]:
        """返回 (是否停止, 是否刷新最佳值)。"""
        if self.best is None:
            improved = True
        elif self.mode == "min":
            improved = value < self.best - self.min_delta
        else:
            improved = value > self.best + self.min_delta

        if improved:
            self.best = value
            self.bad_epochs = 0
            return False, True

        self.bad_epochs += 1
        should_stop = self.patience > 0 and self.bad_epochs >= self.patience
        return should_stop, False


class Trainer:
    """可配置的训练入口。"""

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_loader,
        validation_loader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
    ):
        self.config = config
        self.device = resolve_device(config.trainer.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.optimizer = optimizer or build_optimizer(model, config)
        self.scheduler = scheduler or build_scheduler(
            self.optimizer,
            config,
        )
        self.criterion = build_loss(config.trainer.loss)
        self.use_amp = bool(
            config.trainer.use_amp and self.device.type == "cuda"
        )
        self.scaler = torch.amp.GradScaler(
            device="cuda",
            enabled=self.use_amp,
        )
        self.checkpoint_dir = Path(config.trainer.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.early_stopping = EarlyStopping(
            patience=config.trainer.early_stopping_patience,
            min_delta=config.trainer.early_stopping_min_delta,
            mode=config.trainer.monitor_mode,
        )
        self.wandb = None

        if config.trainer.wandb_enabled:
            try:
                import wandb
            except ImportError as error:
                raise RuntimeError(
                    "wandb_enabled=True，但没有安装 wandb。"
                    "请运行 pip install wandb，或将配置改为 False。"
                ) from error

            self.wandb = wandb.init(
                project=config.trainer.wandb_project,
                entity=config.trainer.wandb_entity,
                name=config.trainer.wandb_run_name,
                config=config.to_dict(),
            )

    def _prepare_batch(self, batch: dict):
        images = batch["image"].to(self.device, non_blocking=True)
        targets = batch["target"].to(self.device, non_blocking=True)
        return images, targets

    @staticmethod
    def _match_target_shape(
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # dataloader 的单任务 target 是 [B]，模型输出是 [B, 1]。
        if predictions.ndim == 2 and predictions.shape[-1] == 1:
            return targets.reshape(-1, 1)
        return targets.reshape_as(predictions)

    def train_one_batch(self, batch: dict) -> Dict[str, float]:
        """训练一个 batch，并返回 loss 和 batch 指标。"""
        self.model.train()
        images, targets = self._prepare_batch(batch)
        self.optimizer.zero_grad(set_to_none=True)

        autocast_context = (
            torch.amp.autocast(device_type="cuda")
            if self.use_amp
            else nullcontext()
        )

        with autocast_context:
            predictions = self.model(images)
            targets = self._match_target_shape(predictions, targets)
            loss = self.criterion(predictions, targets)

        self.scaler.scale(loss).backward()

        if self.config.trainer.gradient_clip_norm is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.trainer.gradient_clip_norm,
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()

        metrics = compute_metrics(predictions, targets)
        metrics["loss"] = float(loss.detach().item())
        return metrics

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        totals = {"loss": 0.0, "mse": 0.0, "rmse": 0.0, "mae": 0.0, "r2": 0.0}
        batch_count = 0

        progress = tqdm(
            self.train_loader,
            desc=f"Train {epoch + 1}/{self.config.trainer.epochs}",
            leave=False,
            disable=not self.config.trainer.progress_bar,
        )

        for batch in progress:
            batch_metrics = self.train_one_batch(batch)
            batch_count += 1
            for key in totals:
                totals[key] += batch_metrics[key]

            current_lr = self.optimizer.param_groups[0]["lr"]
            progress.set_postfix(
                loss=f"{batch_metrics['loss']:.4f}",
                lr=f"{current_lr:.2e}",
            )

        if batch_count == 0:
            raise RuntimeError("train_loader 为空")

        return {
            key: value / batch_count
            for key, value in totals.items()
        }

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        losses = []
        all_predictions = []
        all_targets = []

        progress = tqdm(
            self.validation_loader,
            desc=f"Valid {epoch + 1}/{self.config.trainer.epochs}",
            leave=False,
            disable=not self.config.trainer.progress_bar,
        )

        for batch in progress:
            images, targets = self._prepare_batch(batch)
            predictions = self.model(images)
            targets = self._match_target_shape(predictions, targets)
            loss = self.criterion(predictions, targets)

            losses.append(float(loss.item()))
            all_predictions.append(predictions.detach().cpu())
            all_targets.append(targets.detach().cpu())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        if not losses:
            raise RuntimeError("validation_loader 为空")

        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        metrics = compute_metrics(
             predictions,
            targets,
        )       
        
        metrics["loss"] = metrics["mse"]
        
        return metrics

    def _step_scheduler(self, validation_loss: float) -> None:
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(validation_loss)
        else:
            self.scheduler.step()

    def _get_monitor_value(self, validation_metrics: Dict[str, float]) -> float:
        """从验证指标中取出早停监控值。"""
        monitor = self.config.trainer.monitor.lower()
        if monitor.startswith("val_"):
            metric_name = monitor[4:]
        else:
            metric_name = monitor

        if metric_name not in validation_metrics:
            available = ", ".join(sorted(validation_metrics))
            raise ValueError(
                f"无法监控指标 {monitor}，可选验证指标：{available}"
            )

        return float(validation_metrics[metric_name])

    def _save_checkpoint(
        self,
        epoch: int,
        history: list,
        filename: str,
    ) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_monitor_value": self.early_stopping.best,
            "monitor": self.config.trainer.monitor,
            "history": history,
            "config": self.config.to_dict(),
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)

    def fit(self) -> list:
        history = []
        best_saved = False

        for epoch in range(self.config.trainer.epochs):
            train_metrics = self.train_one_epoch(epoch)
            validation_metrics = self.validate(epoch)
            self._step_scheduler(validation_metrics["loss"])
            monitor_value = self._get_monitor_value(validation_metrics)

            current_lr = self.optimizer.param_groups[0]["lr"]
            row = {
                "epoch": epoch + 1,
                "lr": current_lr,
                "monitor": monitor_value,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in validation_metrics.items()},
            }
            history.append(row)

            print(
                f"Epoch {epoch + 1:03d} | "
                f"train_loss={train_metrics['loss']:.6f} | "
                f"val_loss={validation_metrics['loss']:.6f} | "
                f"val_rmse={validation_metrics['rmse']:.6f} | "
                f"val_r2={validation_metrics['r2']:.4f} | "
                f"lr={current_lr:.2e}"
            )

            if self.wandb is not None:
                self.wandb.log(row, step=epoch + 1)

            should_stop, is_best = self.early_stopping.update(
                monitor_value
            )

            if is_best:
                self._save_checkpoint(
                    epoch=epoch + 1,
                    history=history,
                    filename="best.pt",
                )
                best_saved = True

            if not self.config.trainer.save_best_only:
                self._save_checkpoint(
                    epoch=epoch + 1,
                    history=history,
                    filename="last.pt",
                )

            if should_stop:
                print(
                    "Early stopping: "
                    f"连续 {self.early_stopping.bad_epochs} 轮没有改善。"
                )
                break

        if self.wandb is not None:
            self.wandb.finish()

        if not best_saved:
            print("警告：训练期间没有保存 best.pt。")

        history_path = self.checkpoint_dir / "history.json"
        with open(history_path, "w", encoding="utf-8") as file:
            json.dump(history, file, ensure_ascii=False, indent=2)

        return history
