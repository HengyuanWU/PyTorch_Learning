# models/vgg.py

"""
VGG 模型定义模块

本模块实现了经典的 VGG-16 和 VGG-19 卷积神经网络模型，适用于
ImageNet、CIFAR-10 等图像分类任务。输入图像需为 3×224×224 大小，
输出为指定类别数的分类结果。

包含的类：
- VGG16: VGG-16 模型类
- VGG19: VGG-19 模型类
"""

import torch
import torch.nn as nn


class VGG16(nn.Module):
    """
    VGG-16 模型结构（Simonyan & Zisserman, 2014）：
      - 输入：3×224×224
      - 卷积块：2 → 2 → 3 → 3 → 3 个 Conv3×3 + ReLU，每块后接 MaxPool2×2
      - 分类器：4096 → 4096 → num_classes
    """
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        # 特征提取部分（五个卷积块）
        self.features = nn.Sequential(
            # Block1: 64, 64
            nn.Conv2d(3,  64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),                    # → [B,64,112,112]

            # Block2: 128, 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128,128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),                    # → [B,128,56,56]

            # Block3: 256, 256, 256
            nn.Conv2d(128,256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256,256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256,256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),                    # → [B,256,28,28]

            # Block4: 512, 512, 512
            nn.Conv2d(256,512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512,512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512,512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),                    # → [B,512,14,14]

            # Block5: 512, 512, 512
            nn.Conv2d(512,512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512,512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512,512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),                    # → [B,512,7,7]
        )

        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,        4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        :param x: 输入张量，形状 [B, 3, 224, 224]
        :return: logits，形状 [B, num_classes]
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平为 [B, 512*7*7]
        x = self.classifier(x)
        return x


class VGG19(nn.Module):
    """
    VGG-19 模型结构：
      - 输入：3×224×224
      - 卷积块：2 → 2 → 4 → 4 → 4 个 Conv3×3 + ReLU，每块后接 MaxPool2×2
      - 分类器：4096 → 4096 → num_classes
    """
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        # 特征提取部分（五个卷积块）
        self.features = nn.Sequential(
            # Block1: 64, 64
            nn.Conv2d(3,  64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),                    # → [B,64,112,112]

            # Block2: 128, 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128,128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),                    # → [B,128,56,56]

            # Block3: 256, 256, 256, 256
            nn.Conv2d(128,256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256,256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256,256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256,256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),                    # → [B,256,28,28]

            # Block4: 512, 512, 512, 512
            nn.Conv2d(256,512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512,512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512,512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512,512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),                    # → [B,512,14,14]

            # Block5: 512, 512, 512, 512
            nn.Conv2d(512,512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512,512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512,512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512,512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),                    # → [B,512,7,7]
        )

        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,        4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
