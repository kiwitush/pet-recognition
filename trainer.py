"""
封装训练循环、验证、早停、checkpoint等功能

核心设计:
- 差异化学习率: 新的全连接层 (及注意力模块) 使用 lr_head, backbone 使用 lr
- 混合精度训练 
- 使用 swanlab 日志记录
"""

import os
import random
import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from config import Config


# 工具函数
def init_logger(cfg: Config):
    """根据配置初始化日志记录器 (swanlab)"""
    import swanlab
    run = swanlab.init(project=cfg.project_name, experiment_name=cfg.experiment_name)
    run.config.update({k: v for k, v in vars(cfg).items()})
    return run

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """
    早停: 验证集指标连续 patience 个 epoch 未提升则触发
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, val_metric: float) -> bool:
        if self.best_score is None:
            self.best_score = val_metric
        elif val_metric < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = val_metric
            self.counter = 0
        return self.should_stop


# 定义 Trainer 类
class Trainer:
    def __init__(self, model: nn.Module, config: Config,
                 train_loader: DataLoader, val_loader: DataLoader,
                 logger=None):
        self.model = model.to(config.device)
        self.cfg = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger 

        self.device = config.device
        self.scaler = GradScaler(enabled=config.mixed_precision)
        self.early_stopping = EarlyStopping(patience=config.early_stopping_patience)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.best_val_acc = 0.0
        self.current_epoch = 0

    # 优化器 & 调度器
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """构建差异化学习率的优化器: fc / attention / backbone 三组"""
        fc_params = []
        att_params = []
        backbone_params = []

        head_keys = ("fc.", "heads.head.", "head.")

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(k in name for k in head_keys):
                fc_params.append(param)
            elif any(k in name for k in ("att2", "att3", "att4")):
                att_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = [
            {"params": fc_params,        "lr": self.cfg.lr_head},
            {"params": att_params,        "lr": self.cfg.lr_attention},
            {"params": backbone_params,   "lr": self.cfg.lr},
        ]

        if self.cfg.optimizer == "adamw":
            return torch.optim.AdamW(param_groups, weight_decay=self.cfg.weight_decay)
        else:
            return torch.optim.SGD(param_groups, momentum=0.9,
                                   weight_decay=self.cfg.weight_decay)

    def _build_scheduler(self):
        if self.cfg.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.cfg.epochs
            )
        elif self.cfg.lr_scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.cfg.lr_step_size, gamma=self.cfg.lr_gamma
            )
        return None

    # 单 epoch 训练和验证
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            with autocast(self.device, enabled=self.cfg.mixed_precision):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            if batch_idx % self.cfg.log_interval == 0:
                print(f"  batch {batch_idx:3d}/{len(self.train_loader)}  loss={loss.item():.4f}")

        return {
            "train_loss": total_loss / len(self.train_loader),
            "train_acc":  correct / total,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in self.val_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        return {
            "val_loss": total_loss / len(self.val_loader),
            "val_acc":  correct / total,
        }


    # 主训练循环
    def fit(self) -> Dict[str, float]:
        print(f"\n{'='*50}")
        print(f"实验: {self.cfg.experiment_name}  |  设备: {self.device}")
        print(f"backbone lr: {self.cfg.lr}  |  head lr: {self.cfg.lr_head}  |  weight decay: {self.cfg.weight_decay}")
        print(f"epochs: {self.cfg.epochs}  |  optimizer: {self.cfg.optimizer}  |  mixed precision: {self.cfg.mixed_precision}")
        print(f"{'='*50}\n")

        t0 = time.time()

        for epoch in range(1, self.cfg.epochs + 1):
            self.current_epoch = epoch
            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            if self.scheduler is not None:
                self.scheduler.step()

            # 日志
            fc_lr = self.optimizer.param_groups[0]["lr"]
            att_lr = self.optimizer.param_groups[1]["lr"] if len(self.optimizer.param_groups) > 2 else 0
            backbone_lr = self.optimizer.param_groups[-1]["lr"]
            print(f"epoch {epoch:2d}/{self.cfg.epochs}  "
                  f"train_loss={train_metrics['train_loss']:.4f}  "
                  f"train_acc={train_metrics['train_acc']:.4f}  "
                  f"val_loss={val_metrics['val_loss']:.4f}  "
                  f"val_acc={val_metrics['val_acc']:.4f}  "
                  f"lr_fc={fc_lr:.2e}  lr_att={att_lr:.2e}  lr_bb={backbone_lr:.2e}")

            if self.logger is not None:
                self.logger.log({
                    "epoch": epoch,
                    **train_metrics,
                    **val_metrics,
                    "lr_fc": fc_lr,
                    "lr_attention": att_lr,
                    "lr_backbone": backbone_lr,
                })

            # 保存最佳模型
            if val_metrics["val_acc"] > self.best_val_acc:
                self.best_val_acc = val_metrics["val_acc"]
                self.save_checkpoint("best.pth")
                print(f"  >> 最佳模型已保存 (val_acc={self.best_val_acc:.4f})")

            # 早停
            if self.early_stopping(val_metrics["val_acc"]):
                print(f"早停触发 (epoch {epoch})")
                break

        elapsed = time.time() - t0
        print(f"\n训练完成. 耗时 {elapsed/60:.1f} min, 最佳 val_acc={self.best_val_acc:.4f}")
        return {"best_val_acc": self.best_val_acc}


    # Checkpoint
    def save_checkpoint(self, filename: str) -> None:
        path = os.path.join(self.cfg.checkpoint_dir, self.cfg.experiment_name)
        os.makedirs(path, exist_ok=True)
        torch.save({
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "config": self.cfg,
        }, os.path.join(path, filename))

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_val_acc = ckpt["best_val_acc"]
        return ckpt["epoch"]
