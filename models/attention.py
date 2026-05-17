"""
进阶模型: ResNet + 注意力机制 (SE / CBAM), 在 layer2/layer3/layer4 之后插入注意力模块
"""

from typing import Literal

import torch.nn as nn
from torchvision.models import resnet18, resnet34
from torchvision.models import ResNet18_Weights, ResNet34_Weights

from .att_block import SELayer, CBAM


def _get_attention_module(att_type: str, channels: int) -> nn.Module:
    if att_type == "se":
        return SELayer(channels)
    elif att_type == "cbam":
        return CBAM(channels)
    else:
        raise ValueError(f"目前仅支持 SE 和 CBAM 注意力类型, 不支持: {att_type}")


def _inject_attention(model: nn.Module) -> None:
    """替换 model.forward, 在 layer2/layer3/layer4 之后插入注意力模块."""
    def new_forward(x):
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        # 不在 layer1 后加注意力

        x = model.layer2(x)
        x = model.att2(x)

        x = model.layer3(x)
        x = model.att3(x)

        x = model.layer4(x)
        x = model.att4(x)

        x = model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = model.fc(x)
        return x

    model.forward = new_forward


def build_advanced_model(model_name: str, num_classes: int = 37,
                         pretrained: bool = True,
                         attention_type: Literal["se", "cbam"] = "cbam") -> nn.Module:
    """
    构建带注意力的进阶 ResNet, layer1 之后不加注意力 (特征图大, 开销高收益小)
    """
    if model_name == "resnet18":
        fn = resnet18
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        channels = [64, 128, 256, 512]
    elif model_name == "resnet34":
        fn = resnet34
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        channels = [64, 128, 256, 512]
    else:
        raise ValueError(f"目前仅支持 resnet18/resnet34, 不支持: {model_name}")

    model = fn(weights=weights)

    # 在 layer2/layer3/layer4 之后插入注意力模块
    model.att2 = _get_attention_module(attention_type, channels[1])
    model.att3 = _get_attention_module(attention_type, channels[2])
    model.att4 = _get_attention_module(attention_type, channels[3])

    # 替换输出层
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # 替换 forward, 在指定位置插入注意力计算
    _inject_attention(model)

    return model


if __name__ == "__main__":
    for att in ["se", "cbam"]:
        m = build_advanced_model("resnet18", num_classes=37, attention_type=att)
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"resnet18 + {att.upper()}: 总参数 {total/1e6:.2f}M, 可训练 {trainable/1e6:.2f}M")
