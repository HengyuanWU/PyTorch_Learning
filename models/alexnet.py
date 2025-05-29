"""
AlexNet 模型定义模块

本模块实现了经典的 AlexNet 卷积神经网络模型（Krizhevsky et al., 2012），
适用于 ImageNet、CIFAR-10 等图像分类任务。

输入图像需为 3×224×224（如果使用 CIFAR-10，可先 Resize 或 CenterCrop 到 224×224）。
输出为指定类别数的分类结果。

包含的类：
- AlexNet: 实现 AlexNet 模型的 PyTorch 模型类

网络结构概览：
Input (3×224×224)
 ↓
Conv1 + ReLU (64×11×11, stride=4, padding=2) → [64×55×55]
 ↓
MaxPool (3×3, stride=2) → [64×27×27]
 ↓
Conv2 + ReLU (192×5×5, padding=2) → [192×27×27]
 ↓
MaxPool (3×3, stride=2) → [192×13×13]
 ↓
Conv3 + ReLU (384×3×3, padding=1) → [384×13×13]
 ↓
Conv4 + ReLU (256×3×3, padding=1) → [256×13×13]
 ↓
Conv5 + ReLU (256×3×3, padding=1) → [256×13×13]
 ↓
MaxPool (3×3, stride=2) → [256×6×6]
 ↓
Flatten → [256*6*6=9216]
 ↓
FC1 + ReLU + Dropout → [4096]
 ↓
FC2 + ReLU + Dropout → [4096]
 ↓
FC3 → [num_classes]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    """
    AlexNet (Krizhevsky et al., 2012) PyTorch 实现。

    :param num_classes: 分类类别数，默认 1000（ImageNet），CIFAR-10 时可设为10
    :param dropout_rate: Dropout 概率，默认 0.5
    """

    def __init__(self, num_classes: int = 1000, dropout_rate: float = 0.5):
        super().__init__()
        # 特征提取部分
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # [3→64, 11×11]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # [64×55×55→64×27×27]

            nn.Conv2d(64, 192, kernel_size=5, padding=2),  # [192×27×27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # [192×27×27→192×13×13]

            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # [384×13×13]
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # [256×13×13]
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # [256×13×13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # [256×13×13→256×6×6]
        )
        # 自适应池化保证 6×6 输出（可选）
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        :param x: 输入张量，形状 [B, 3, 224, 224]
        :return: logits，形状 [B, num_classes]
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # [B, 256*6*6]
        x = self.classifier(x)
        return x
