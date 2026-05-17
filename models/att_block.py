"""
注意力模块: 
SE-block 和 CBAM 
"""

import torch
import torch.nn as nn


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation：
    通道全局平均池化 → FC-ReLU-FC-Sigmoid(每个通道学出一个权重) → 乘回原特征图
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=True)
        self.sigmoid = nn.Sigmoid()
        # 初始化最后一层 bias 为正, 使 sigmoid 输出接近 1 (注意力初始"透明")
        nn.init.constant_(self.fc2.bias, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.fc2(torch.relu(self.fc1(y)))
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


class ChannelAttention(nn.Module):
    """CBAM 的通道注意力子模块: 同时使用 AvgPool 和 MaxPool 压缩间维度"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 初始化最后一层 bias 为正, 使 sigmoid 输出接近 1 (注意力初始"透明")
        nn.init.constant_(self.fc2.bias, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = self.fc2(torch.relu(self.fc1(self.avg_pool(x).view(b, c))))
        max = self.fc2(torch.relu(self.fc1(self.max_pool(x).view(b, c))))
        return torch.sigmoid(avg + max).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """CBAM 的空间注意力子模块: 沿通道维度做 Avg/Max, 卷积产生空间权重图"""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=True)
        # 初始化 bias 为正, 使 sigmoid 输出接近 1 (注意力初始"透明")
        nn.init.constant_(self.conv.bias, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        max = x.max(dim=1, keepdim=True).values
        return torch.sigmoid(self.conv(torch.cat([avg, max], dim=1)))


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module: 通道注意力 + 空间注意力
    输入特征图先经过通道注意力模块, 再经过空间注意力模块
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x
