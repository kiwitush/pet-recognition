"""
超参数集中管理, 各模块通过 Config 对象获取参数
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class Config:
    experiment_name: str = "baseline"
    project_name: str = "pet-recognition"
    seed: int = 42

    # 数据集
    data_root: str = "./data"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    val_split: float = 0.15
    num_classes: int = 37

    # 模型
    model_name: Literal["resnet18", "resnet34", "resnet50", "vit_tiny", "vit_small"] = "resnet34"
    pretrained: bool = True
    attention_type: Optional[Literal["se", "cbam"]] = None

    # 训练
    epochs: int = 50
    lr: float = 1e-4
    lr_head: float = 1e-3
    lr_attention: float = 5e-4
    weight_decay: float = 1e-4
    optimizer: Literal["adamw", "sgd"] = "adamw"
    lr_scheduler: Optional[Literal["cosine", "step"]] = "cosine"
    lr_step_size: int = 10
    lr_gamma: float = 0.1
    label_smoothing: float = 0.1
    mixed_precision: bool = True

    # 早停
    early_stopping_patience: int = 10

    # 日志与保存
    logger: Literal["wandb", "swanlab", "none"] = "swanlab"
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 20

    # 设备
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)


# 默认配置
def baseline_config() -> Config:
    """预训练微调 ResNet-34 Baseline"""
    return Config(
        experiment_name="baseline_resnet34",
        model_name="resnet34",
        pretrained=True,
        epochs=50,
    )


def contrast_config() -> Config:
    """预训练消融: 随机初始化 vs 预训练"""
    return Config(
        experiment_name="contrast",
        model_name="resnet18",
        pretrained=False,
        epochs=50,
    )


def attention_config(att_type: Literal["se", "cbam"] = "cbam") -> Config:
    """进阶模型: ResNet-18 + 注意力机制 (se / cbam)"""
    return Config(
        experiment_name=f"attention_resnet18_{att_type}",
        model_name="resnet18",
        pretrained=True,
        attention_type=att_type,
        epochs=50,
    )


def vit_config(vit_type: Literal["vit_tiny", "vit_small"] = "vit_tiny") -> Config:
    """ViT 实验: ViT-Tiny/16 或 ViT-Small/16"""
    return Config(
        experiment_name=vit_type,
        model_name=vit_type,
        pretrained=True,
        epochs=80,
        lr=5e-5,
        lr_head=5e-4,
        weight_decay=1e-3,
        label_smoothing=0.15,
    )


if __name__ == "__main__":
    cfg = baseline_config()
    print(f"实验: {cfg.experiment_name}")
    print(f"模型: {cfg.model_name}  |  预训练: {cfg.pretrained}  |  注意力: {cfg.attention_type}")
    print(f"设备: {cfg.device}  |  混合精度: {cfg.mixed_precision}")
    print(f"学习率: backbone={cfg.lr}, head={cfg.lr_head}  |  epochs: {cfg.epochs}")
    print(f"日志: {cfg.logger}")
