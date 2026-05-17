"""
Baseline 模型: 
微调在 ImageNet 上预训练的 ResNet, 修改输出层适配 37 分类
"""

import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights


MODEL_FACTORY = {
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (resnet34, ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V1),
}


def build_baseline_model(model_name: str, num_classes: int = 37,
                         pretrained: bool = True) -> nn.Module:
    """构建 Baseline ResNet, 替换输出层为 num_classes 分类."""
    if model_name not in MODEL_FACTORY:
        raise ValueError(f"不支持的模型: {model_name}. 可选: {list(MODEL_FACTORY.keys())}")

    fn, weights = MODEL_FACTORY[model_name]

    if pretrained:
        model = fn(weights=weights)
    else:
        model = fn(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


if __name__ == "__main__":
    for name in ["resnet18", "resnet34"]:
        m = build_baseline_model(name, num_classes=37, pretrained=True)
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"{name}: 总参数 {total/1e6:.2f}M, 可训练 {trainable/1e6:.2f}M")
