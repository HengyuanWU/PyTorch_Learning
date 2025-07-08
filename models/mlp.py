import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoLayerMLP(nn.Module):
    """
    两层多层感知机（MLP）模型结构
      - 输入：3×32×32（需先展平为3072维向量）
      - FC1 + ReLU → [256]
      - FC2 → [10] （输出类别分数）
    层次细节：
      - FC1: Linear(3072→256)
      - FC2: Linear(256→num_classes)
    """

    def __init__(self, num_classes: int = 10, input_channels: int = 3, image_size: int = 32):
        """
        初始化两层MLP模型

        :param num_classes: 分类类别数，默认 10
        :param input_channels: 输入图像通道数，默认 3 (CIFAR-10)
        :param image_size: 输入图像大小，默认 32×32
        """
        super().__init__()
        # 计算输入特征数量
        self.input_features = input_channels * image_size * image_size
        
        # 输入层（3072）→ 隐藏层（256）
        self.fc1 = nn.Linear(in_features=self.input_features, out_features=256)
        # 输出层（256）→ 分类结果
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        :param x: 输入张量，形状 [B, C, H, W] 其中C=3, H=W=32
        :return: logits，形状 [B, num_classes]
        """
        batch_size = x.size(0)
        # 展平输入 [B, C, H, W] → [B, C*H*W]
        x = x.view(batch_size, -1)

        # FC1 → ReLU
        x = F.relu(self.fc1(x))  # [B, 256]

        # 输出层
        return self.fc2(x)  # [B, num_classes]
