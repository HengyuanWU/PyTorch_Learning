"""
DenseNet 模型定义模块

实现标准 DenseNet-121 和 CIFAR-10 定制版 DenseNet 模型：
- 标准 DenseNet-121: 原始论文实现 (L=121, k=32)
- DenseNet-CIFAR: 针对 CIFAR-10 优化的轻量版 (L=100, k=12, θ=0.5)

包含的类：
- DenseLayer: 基础密集连接层（含 Bottleneck 设计）
- DenseBlock: 密集块（多个 DenseLayer 组成）
- Transition: 过渡层（特征图压缩）
- DenseNet: 标准 DenseNet-121 实现
- DenseNetCustom: CIFAR-10 定制版实现

DenseNet 核心特点：
1. 密集连接：每层接收前面所有层的特征图作为输入
2. 增长率 (k): 控制每层新增特征图数量
3. 瓶颈层 (BC): 通过 1×1 卷积减少计算量
4. 过渡层: 通过 θ (0<θ≤1) 控制特征图压缩比例
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class DenseLayer(nn.Module):
    """
    DenseNet 基础层（含 Bottleneck 设计）
    结构：BN → ReLU → 1×1Conv(4k) → BN → ReLU → 3×3Conv(k)
    
    :param in_channels: 输入通道数
    :param growth_rate: 增长率 k（每层新增特征图数量）
    :param bottleneck: 是否使用瓶颈层设计
    :param dropout: Dropout 概率（默认 0 表示不使用）
    """
    def __init__(self, in_channels: int, growth_rate: int, 
                 bottleneck: bool = True, dropout: float = 0.0):
        super().__init__()
        inter_channels = growth_rate * 4 if bottleneck else in_channels
        layers = [
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        ]
        
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
            
        layers.extend([
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DenseBlock(nn.Module):
    """
    密集块：包含多个 DenseLayer，每层输出与前面所有层输出在通道维度拼接
    
    :param num_layers: 该块包含的 DenseLayer 数量
    :param in_channels: 输入通道数
    :param growth_rate: 增长率 k
    :param bottleneck: 是否使用瓶颈层
    :param dropout: Dropout 概率
    """
    def __init__(self, num_layers: int, in_channels: int, 
                 growth_rate: int, bottleneck: bool = True, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = DenseLayer(
                in_channels + i * growth_rate,
                growth_rate,
                bottleneck,
                dropout
            )
            self.layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)
        return torch.cat(features, dim=1)

class Transition(nn.Module):
    """
    过渡层：压缩特征图尺寸和通道数
    
    :param in_channels: 输入通道数
    :param compression: 压缩系数 θ (输出通道 = θ * in_channels)
    :param dropout: Dropout 概率
    """
    def __init__(self, in_channels: int, compression: float = 0.5, dropout: float = 0.0):
        super().__init__()
        out_channels = int(in_channels * compression)
        layers = [
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.AvgPool2d(2, stride=2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DenseNet(nn.Module):
    """
    标准 DenseNet-121 实现 (L=121, k=32)
    结构：
    - 初始卷积：7×7, stride=2 → 3×3 maxpool
    - DenseBlock1 (6 layers) → Transition
    - DenseBlock2 (12 layers) → Transition
    - DenseBlock3 (24 layers) → Transition
    - DenseBlock4 (16 layers)
    - 全局平均池化 → 全连接层
    
    :param num_classes: 分类类别数
    :param growth_rate: 增长率 k (默认 32)
    :param compression: 过渡层压缩系数 θ (默认 0.5)
    :param dropout: Dropout 概率 (默认 0)
    """
    def __init__(self, num_classes: int = 10, growth_rate: int = 32, 
                 compression: float = 0.5, dropout: float = 0.0):
        super().__init__()
        # 初始卷积 (ImageNet 尺寸适配)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # DenseBlock 配置 (6-12-24-16)
        block_config = [6, 12, 24, 16]
        channels = 64
        
        # 构建 DenseBlock 和 Transition
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, channels, growth_rate, True, dropout)
            self.features.add_module(f'denseblock{i+1}', block)
            channels += num_layers * growth_rate
            
            if i != len(block_config) - 1:  # 最后一个block后不加Transition
                trans = Transition(channels, compression, dropout)
                self.features.add_module(f'transition{i+1}', trans)
                channels = int(channels * compression)
        
        # 分类头
        self.classifier = nn.Linear(channels, num_classes)
        
        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class DenseNetCustom(nn.Module):
    """
    CIFAR-10 定制版 DenseNet (L=100, k=12, θ=0.5)
    结构：
    - 初始卷积：3×3 → 3×3
    - DenseBlock1 (16 layers) → Transition
    - DenseBlock2 (16 layers) → Transition
    - DenseBlock3 (16 layers)
    - 全局平均池化 → 全连接层
    
    :param num_classes: 分类类别数 (CIFAR-10 默认 10)
    :param growth_rate: 增长率 k (默认 12)
    :param compression: 过渡层压缩系数 θ (默认 0.5)
    :param dropout: Dropout 概率 (默认 0)
    """
    def __init__(self, num_classes: int = 10, growth_rate: int = 12,
                 compression: float = 0.5, dropout: float = 0.0):
        super().__init__()
        # CIFAR-10 适配的初始卷积 (保持32×32分辨率)
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # 3个DenseBlock (每个16层)
        block_config = [16, 16, 16]
        channels = 16
        
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, channels, growth_rate, True, dropout)
            self.features.add_module(f'denseblock{i+1}', block)
            channels += num_layers * growth_rate
            
            if i != len(block_config) - 1:  # 最后一个block后不加Transition
                trans = Transition(channels, compression, dropout)
                self.features.add_module(f'transition{i+1}', trans)
                channels = int(channels * compression)
        
        # 分类头
        self.classifier = nn.Linear(channels, num_classes)
        
        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
