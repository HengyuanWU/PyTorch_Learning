"""
LeNet-5 模型定义模块

本模块实现了经典的 LeNet-5 卷积神经网络模型，适用于 MNIST 等图像分类任务。
输入图像需为 32x32 大小（例如对 MNIST 图像进行填充），输出为指定类别数的分类结果。

包含的类：
- LeNet5: 实现 LeNet-5 模型的 PyTorch 模型类

Input (1×32×32)
   ↓
Conv1 + ReLU (6×5×5) → [6×28×28]
   ↓
AvgPool (2×2) → [6×14×14]
   ↓
Conv2 + ReLU (16×5×5) → [16×10×10]
   ↓
AvgPool (2×2) → [16×5×5]
   ↓
Flatten → [400]
   ↓
FC1 + ReLU → [120]
   ↓
FC2 + ReLU → [84]
   ↓
FC3 → [10] （输出类别分数）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    """
    LeNet-5 模型结构（LeCun et al., 1998）：
      - 输入：1×32×32（MNIST 图像通过 Pad(2) 或 Resize 到 32×32）
      - C1 -> ReLU -> S2 (AvgPool 2×2)
      - C3 -> ReLU -> S4 (AvgPool 2×2)
      - C5 -> ReLU
      - F6 -> ReLU
      - 输出层
    层次细节：
      - C1: Conv2d(1→6, kernel=5×5)，输出特征图 6×28×28
      - S2: AvgPool2d(kernel=2, stride=2)，输出 6×14×14
      - C3: Conv2d(6→16, kernel=5×5) ，输出 16×10×10
      - S4: AvgPool2d(kernel=2, stride=2)，输出 16×5×5
      - C5: Conv2d(16→120, kernel=5×5)，输出 120×1×1（等价于全连接到 120 维）
      - F6: Linear(120 → 84)
      - 输出层: Linear(84 → num_classes)
    """
    def __init__(self, num_classes: int = 10):
        """
        初始化 LeNet-5 模型

        :param num_classes: 分类类别数，默认 10
        """
        super().__init__()
        # C1: 输入通道 1，输出通道 6，卷积核大小 5×5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        # S2: 平均池化层，kernel=2，stride=2
        self.pool  = nn.AvgPool2d(kernel_size=2, stride=2)
        # C3: 输入 6，输出 16，卷积核 5×5
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # C5: 输入 16，输出 120，卷积核 5×5
        # 等同于把 16×5×5 的特征图映射到 120 维向量
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        # F6: 全连接层 120→84
        self.fc1   = nn.Linear(in_features=120, out_features=84)
        # 输出层: 全连接 84→num_classes
        self.fc2   = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        :param x: 输入张量，形状 [B, 1, 32, 32]
        :return: logits，形状 [B, num_classes]
        """
        # C1 → ReLU → S2
        x = F.relu(self.conv1(x))  # [B, 6, 28, 28]
        x = self.pool(x)           # [B, 6, 14, 14]

        # C3 → ReLU → S4
        x = F.relu(self.conv2(x))  # [B, 16, 10, 10]
        x = self.pool(x)           # [B, 16, 5, 5]

        # C5 → ReLU
        x = F.relu(self.conv3(x))  # [B, 120, 1, 1]

        # 展平到 [B, 120]
        x = x.view(x.size(0), -1)

        # F6 → ReLU
        x = F.relu(self.fc1(x))    # [B, 84]

        # 输出层（不加激活，配合 CrossEntropyLoss）
        x = self.fc2(x)            # [B, num_classes]
        return x
