"""超参数网格搜索。

搜索维度：

* 优化器类型
* learning rate
* weight decay
* 回归头 dropout

网格搜索为了保证不同 trial 可比较，强制只使用 constant scheduler。
数据集只建立一次，所有 trial 复用同一个 train/validation 划分。

示例：

    python grid_search.py \\
        --backbone dinov3 \\
        --freeze-backbone \\
        --optimizers adamw,adam \\
        --lrs 1e-4,5e-4 \\
        --weight-decays 0,1e-4 \\
        --dropouts 0,0.2 \\
        --epochs 10 \\
        --batch-size 16 \\
        --device cuda:0
"""

from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import argparse
import json

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import Config, set_seed
from dataloader import build_datasets
from model import CScanRegressor
from trainer import Trainer


SUPPORTED_OPTIMIZERS = ("adam", "adamw", "sgd", "rmsprop")


@dataclass
class GridSearchSpace:
    """网格搜索空间及搜索级别配置。"""

    optimizers: List[str] = field(default_factory=lambda: ["adamw"])
    learning_rates: List[float] = field(
        default_factory=lambda: [1e-4, 5e-4, 1e-3]
    )
    weight_decays: List[float] = field(
        default_factory=lambda: [0.0, 1e-4, 3e-4]
    )
    dropouts: List[float] = field(
        default_factory=lambda: [0.0, 0.2]
    )
    output_dir: str = "grid_search_runs"
    wandb_enabled: bool = False
    max_trials: Optional[int] = None


def _validate_search_space(space: GridSearchSpace) -> None:
    """检查搜索空间，尽早提示非法值。"""
    if not space.optimizers:
        raise ValueError("optimizers 不能为空")

    invalid_optimizers = [
        name for name in space.optimizers
        if name.lower() not in SUPPORTED_OPTIMIZERS
    ]
    if invalid_optimizers:
        raise ValueError(
            f"不支持的优化器: {invalid_optimizers}；"
            f"可选：{', '.join(SUPPORTED_OPTIMIZERS)}"
        )

    if not space.learning_rates or any(
        value <= 0 for value in space.learning_rates
    ):
        raise ValueError("learning_rates 必须是正数列表")

    if not space.weight_decays or any(
        value < 0 for value in space.weight_decays
    ):
        raise ValueError("weight_decays 必须是非负数列表")

    if not space.dropouts or any(
        value < 0 or value >= 1 for value in space.dropouts
    ):
        raise ValueError("dropouts 必须位于 [0, 1)")

    if space.max_trials is not None and space.max_trials <= 0:
        raise ValueError("max_trials 必须大于 0")


class GridSearcher:
    """执行超参数网格搜索。"""

    def __init__(
        self,
        base_config: Config,
        search_space: GridSearchSpace,
    ):
        self.base_config = deepcopy(base_config)
        self.search_space = search_space

        # 明确限制：网格搜索不允许动态改变学习率。
        if self.base_config.scheduler.name.lower() != "constant":
            raise ValueError(
                "GridSearcher 只支持 scheduler.name='constant'。"
                f"当前配置为 {self.base_config.scheduler.name!r}。"
            )

        _validate_search_space(search_space)
        self.output_dir = Path(search_space.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 只建立一次数据集，保证所有 trial 使用相同的数据划分。
        self.train_dataset, self.validation_dataset = build_datasets(
            image_dir=self.base_config.data.image_dir,
            condition_dir=self.base_config.data.condition_dir,
            image_size=self.base_config.data.image_size,
            validation_ratio=self.base_config.data.validation_ratio,
            seed=self.base_config.trainer.seed,
            thickness_per_ply_mm=(
                self.base_config.data.thickness_per_ply_mm
            ),
        )

    def iter_trials(self) -> Iterable[Dict[str, object]]:
        """按固定顺序产生每个 trial 的超参数组合。"""
        combinations = product(
            self.search_space.optimizers,
            self.search_space.learning_rates,
            self.search_space.weight_decays,
            self.search_space.dropouts,
        )

        for index, (optimizer, learning_rate, weight_decay, dropout) in enumerate(combinations, start=1):
            if (
                self.search_space.max_trials is not None
                and index > self.search_space.max_trials
            ):
                break

            yield {
                "optimizer": optimizer.lower(),
                "learning_rate": float(learning_rate),
                "weight_decay": float(weight_decay),
                "dropout": float(dropout),
            }

    def _make_loaders(self, config: Config):
        """为每个 trial 创建独立 DataLoader，但复用相同 Dataset。"""
        pin_memory = self._device_is_cuda(config.trainer.device)
        generator = torch.Generator()
        generator.manual_seed(config.trainer.seed)

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=pin_memory,
            generator=generator,
        )
        validation_loader = DataLoader(
            self.validation_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=pin_memory,
        )
        return train_loader, validation_loader

    @staticmethod
    def _device_is_cuda(device_name: str) -> bool:
        if device_name == "auto":
            return torch.cuda.is_available()
        return device_name.startswith("cuda")

    @staticmethod
    def _best_history_row(history: list, config: Config) -> dict:
        if not history:
            raise RuntimeError("trial 没有产生训练 history")

        key = "monitor"
        mode = config.trainer.monitor_mode.lower()

        if mode == "max":
            return max(history, key=lambda row: row[key])
        return min(history, key=lambda row: row[key])

    def run(self) -> List[dict]:
        results = []
        trials = list(self.iter_trials())

        if not trials:
            raise RuntimeError("搜索空间没有生成任何 trial")

        print(f"Grid search trials: {len(trials)}")
        progress = tqdm(trials, desc="Grid search", unit="trial")

        for trial_index, hyperparameters in enumerate(progress, start=1):
            config = deepcopy(self.base_config)
            config.scheduler.name = "constant"
            config.optimizer.name = hyperparameters["optimizer"]
            config.optimizer.learning_rate = hyperparameters["learning_rate"]
            config.optimizer.weight_decay = hyperparameters["weight_decay"]
            config.model.dropout = hyperparameters["dropout"]
            config.trainer.checkpoint_dir = str(
                self.output_dir / f"trial_{trial_index:03d}"
            )
            # 网格搜索默认关闭 wandb；需要时每个 trial 使用独立 run name。
            config.trainer.wandb_enabled = self.search_space.wandb_enabled
            if config.trainer.wandb_enabled:
                config.trainer.wandb_run_name = (
                    f"grid-{trial_index:03d}-"
                    f"{hyperparameters['optimizer']}-"
                    f"lr{hyperparameters['learning_rate']}-"
                    f"wd{hyperparameters['weight_decay']}-"
                    f"drop{hyperparameters['dropout']}"
                )

            set_seed(config.trainer.seed)
            train_loader, validation_loader = self._make_loaders(config)

            result = {
                "trial": trial_index,
                **hyperparameters,
                "scheduler": "constant",
                "checkpoint_dir": config.trainer.checkpoint_dir,
            }

            try:
                model = CScanRegressor(config)
                trainer = Trainer(
                    model=model,
                    config=config,
                    train_loader=train_loader,
                    validation_loader=validation_loader,
                )
                history = trainer.fit()
                best_row = self._best_history_row(history, config)
                result.update(
                    {
                        "status": "success",
                        "best_epoch": best_row["epoch"],
                        "best_monitor": best_row["monitor"],
                        "best_val_loss": best_row["val_loss"],
                        "best_val_rmse": best_row["val_rmse"],
                        "best_val_mae": best_row["val_mae"],
                        "best_val_r2": best_row["val_r2"],
                    }
                )
            except Exception as error:
                # 单个 trial 失败不影响其余组合，并把原因写进汇总文件。
                result.update(
                    {
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )

            results.append(result)
            progress.set_postfix(status=result["status"])
            self._save_results(results)

        successful = [
            result for result in results
            if result["status"] == "success"
        ]
        if successful:
            mode = self.base_config.trainer.monitor_mode.lower()
            if mode == "max":
                best = max(successful, key=lambda row: row["best_monitor"])
            else:
                best = min(successful, key=lambda row: row["best_monitor"])
            with open(self.output_dir / "best.json", "w", encoding="utf-8") as file:
                json.dump(best, file, ensure_ascii=False, indent=2)
            print(
                "Best trial: "
                f"#{best['trial']} ({best['best_monitor']:.6f})"
            )
        else:
            print("没有成功完成的 trial，请检查 summary.json 中的 error。")

        return results

    def _save_results(self, results: List[dict]) -> None:
        with open(self.output_dir / "summary.json", "w", encoding="utf-8") as file:
            json.dump(results, file, ensure_ascii=False, indent=2)


def _parse_list(text: str, value_type):
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(value_type(item))
    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grid search optimizer/lr/weight-decay/dropout"
    )
    parser.add_argument("--optimizers", default="adamw")
    parser.add_argument("--lrs", default="1e-4,5e-4,1e-3")
    parser.add_argument("--weight-decays", default="0,1e-4,3e-4")
    parser.add_argument("--dropouts", default="0,0.2")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--backbone", choices=["resnet18", "resnet34", "resnet50", "dinov3"], default=None)
    parser.add_argument("--dinov3-path", default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="grid_search_runs")
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()
    # 搜索器硬性要求 constant；CLI 不提供 scheduler 参数，避免误配。
    config.scheduler.name = "constant"

    if args.epochs is not None:
        config.trainer.epochs = args.epochs
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.backbone is not None:
        config.model.backbone = args.backbone
    if args.dinov3_path is not None:
        config.model.dinov3_path = args.dinov3_path
    if args.hidden_dim is not None:
        config.model.hidden_dim = args.hidden_dim
    if args.device is not None:
        config.trainer.device = args.device
    if args.freeze_backbone:
        config.model.freeze_backbone = True

    space = GridSearchSpace(
        optimizers=_parse_list(args.optimizers, str),
        learning_rates=_parse_list(args.lrs, float),
        weight_decays=_parse_list(args.weight_decays, float),
        dropouts=_parse_list(args.dropouts, float),
        output_dir=args.output_dir,
        wandb_enabled=args.wandb,
        max_trials=args.max_trials,
    )
    GridSearcher(config, space).run()


if __name__ == "__main__":
    main()
