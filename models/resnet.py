# models/resnet.py
"""
ResNet-18 模型定义模块

本模块实现了经典的 ResNet-18 卷积神经网络模型，适用于 CIFAR-10 等图像分类任务。
输入图像需为 3×32×32 大小（例如CIFAR-10图像），输出为指定类别数的分类结果。

包含的类：
- BasicBlock: 基本残差块（ResNet的基本构建单元）
- ResNet18: 实现 ResNet-18 模型的 PyTorch 模型类

网络结构层次：
Input (3×32×32)
   ↓
Conv1 + BN + ReLU → [64×32×32]
   ↓
Layer1: BasicBlock×2 → [64×32×32]
   ↓
Layer2: BasicBlock×2 → [128×16×16]
   ↓
Layer3: BasicBlock×2 → [256×8×8]
   ↓
Layer4: BasicBlock×2 → [512×4×4]
   ↓
AdaptiveAvgPool2d(1×1) → [512×1×1]
   ↓
Flatten → [512]
   ↓
FC → [num_classes] （输出类别分数）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    ResNet 基本残差块

    结构：
      Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> Add(Shortcut)
      当输入输出通道数不一致时，使用1x1卷积调整维度

    层次细节：
      - Conv1: Conv2d(in_channels→out_channels, kernel=3×3, stride=stride) → 输出特征图 H×W
      - Conv2: Conv2d(out_channels→out_channels, kernel=3×3, stride=1)
      - Shortcut: 当 stride≠1 或 in_channels≠out_channels 时，使用1x1卷积调整维度
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        """
        初始化基本残差块

        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param stride: 卷积步长（用于第一个卷积层）
        """
        super().__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                             stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                             stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 残差连接中的1x1卷积（当需要调整维度时）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """
        前向传播

        :param x: 输入张量，形状 [B, C_in, H, W]
        :return: 输出张量，形状 [B, C_out, H', W']
        """
        # 残差分支
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # 残差连接 + ReLU
        out += self.shortcut(x)
        return F.relu(out)

class ResNet18(nn.Module):
    """
    ResNet-18 模型结构（He et al., 2015）

    输入要求：3×32×32（适用于CIFAR-10数据集）

    网络结构流程：
    Input (3×32×32)
       ↓
    Conv1: Conv(3→64, 3×3, stride=1, padding=1) → [64×32×32]
       ↓
    Layer1: 2×BasicBlock(64→64) → [64×32×32]
       ↓
    Layer2: BasicBlock(64→128) ×2 → [128×16×16]
       ↓
    Layer3: BasicBlock(128→256) ×2 → [256×8×8]
       ↓
    Layer4: BasicBlock(256→512) ×2 → [512×4×4]
       ↓
    AvgPool: AdaptiveAvgPool2d(1×1) → [512×1×1]
       ↓
    FC: Linear(512 → num_classes) → [num_classes]
    """
    def __init__(self, num_classes=10):
        """
        初始化 ResNet-18 模型

        :param num_classes: 分类类别数，默认 10
        """
        super().__init__()
        # 初始特征提取层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 网络主干
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        
        # 分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """
        构建残差块组

        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param blocks: 块数量
        :param stride: 步长（用于第一个块的下采样）
        :return: 包含多个 BasicBlock 的序列
        """
        layers = []
        # 第一个块可能需要调整步长和通道数
        layers.append(BasicBlock(in_channels, out_channels, stride))
        # 后续块保持相同通道数和步长=1
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播

        :param x: 输入张量，形状 [B, 3, 32, 32]
        :return: logits，形状 [B, num_classes]
        """
        # 初始卷积
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 64, 32, 32]
        
        # 网络主干
        x = self.layer1(x)  # [B, 64, 32, 32]
        x = self.layer2(x)  # [B, 128, 16, 16]
        x = self.layer3(x)  # [B, 256, 8, 8]
        x = self.layer4(x)  # [B, 512, 4, 4]
        
        # 分类
        x = self.avgpool(x)  # [B, 512, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 512]
        return self.fc(x)