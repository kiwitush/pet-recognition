"""
Vision Transformer: 支持 ViT-Tiny/16 和 ViT-Small/16, ImageNet 预训练 (timm)
"""

import timm
import torch.nn as nn
from typing import Literal

MODEL_NAMES = {
    "vit_tiny":  "vit_tiny_patch16_224",
    "vit_small": "vit_small_patch16_224",
}


def build_vit_model(model_name: Literal["vit_tiny", "vit_small"] = "vit_tiny",
                    num_classes: int = 37, pretrained: bool = True) -> nn.Module:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"不支持: {model_name}, 可选: {list(MODEL_NAMES.keys())}")

    timm_name = MODEL_NAMES[model_name]
    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
    return model


if __name__ == "__main__":
    for name in ["vit_tiny", "vit_small"]:
        m = build_vit_model(name, num_classes=37, pretrained=False)
        total = sum(p.numel() for p in m.parameters())
        print(f"{name}: {total/1e6:.2f}M 参数")
