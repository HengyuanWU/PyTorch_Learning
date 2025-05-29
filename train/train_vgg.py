"""
VGG 模型定义与训练模块

本模块实现 VGG 系列模型（VGG16/VGG19 等），适配 CIFAR-10 图像分类任务。
输入图像需调整为 224×224 大小，输出为 10 类别分类结果。

典型 VGG16 结构：
Input (3×224×224)
   ↓
Conv3x3(64) → ReLU → Conv3x3(64) → ReLU → MaxPool2d
   ↓
Conv3x3(128) → ReLU → Conv3x3(128) → ReLU → MaxPool2d
   ↓
Conv3x3(256) → ReLU ×3 → MaxPool2d
   ↓
Conv3x3(512) → ReLU ×3 → MaxPool2d
   ↓
Conv3x3(512) → ReLU ×3 → MaxPool2d
   ↓
Flatten → FC(4096) → ReLU → Dropout → FC(4096) → ReLU → Dropout → FC(10)
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from models.vgg import VGG16, VGG19  # VGG 模型定义（根据实现可能包含不同配置）
from utils import PROJECT_ROOT
from utils.trainer import fit  # 通用训练流程封装


def get_cifar10_loader(batch_size: int = 64, train: bool = True) -> DataLoader:
    """构建适配 VGG 的 CIFAR-10 数据加载管道

    输入处理流程：
      CIFAR-10 (32×32) → Resize(224×224) → ToTensor → Normalize

    :param batch_size: 单批次样本数（训练集建议64，验证集128）
    :param train: 数据模式选择（True=训练集，False=验证集）
    :return: 配置完成的数据加载器
    """
    # 图像预处理流水线（适配 ImageNet 预训练尺寸）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # VGG标准输入尺寸（原始论文使用224×224）
        transforms.ToTensor(),  # [0,255] → [0.0,1.0] 并转换维度顺序 HWC→CHW
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10三通道均值（基于50k训练集计算）
            std=[0.2023, 0.1994, 0.2010]  # CIFAR-10三通道标准差
        )
    ])

    # 数据集路径配置（保持与项目其他模块一致）
    root_dir = Path(PROJECT_ROOT) / 'data'  # 统一数据存储路径

    return DataLoader(
        CIFAR10(
            root=root_dir,  # 数据集存储根目录（自动创建子目录）
            train=train,  # 模式切换：True=训练集(50k样本)
            download=True,  # 自动下载缺失数据集（仅首次需要）
            transform=transform  # 应用预处理流水线
        ),
        batch_size=batch_size,  # 典型配置：训练64/验证128（平衡显存与效率）
        shuffle=train,  # 仅训练集启用数据打乱（防止过拟合）
        num_workers=4,  # 数据加载子进程数（建议为CPU物理核心数75%）
        pin_memory=True  # 启用锁页内存（提升GPU数据传输效率约20%）
    )


def run_vgg_training(
        model_type: str = 'vgg16',  # 模型变体选择（vgg16含13个卷积层，vgg19含16个卷积层）
        epochs: int = 50,  # 训练轮次（VGG需要更长时间收敛）
        batch_size: int = 64,  # 训练批次大小（根据显存调整）
        lr: float = 0.005,  # 初始学习率（推荐值0.01-0.001）
        momentum: float = 0.9,  # 动量系数（加速收敛）
        weight_decay: float = 5e-4,  # L2正则化强度（应对全连接层过拟合）
        patience: int = 5,  # 早停耐心值（连续无改进轮数）
        min_delta: float = 1e-3,  # 损失最小改进阈值（绝对值）
        log_interval: int = 100,  # 日志打印间隔（按批次计数）
        output_dir: str = None,  # 模型输出目录（默认 outputs/vgg16/）
) -> list[tuple[float, float]]:
    """执行 VGG 模型完整训练流程

    典型训练配置：
      - 优化器：带动量的SGD（配合学习率衰减）
      - 损失函数：交叉熵损失（内置softmax）
      - 正则化策略：L2权重衰减 + Dropout

    :return: 各epoch验证指标列表（loss, accuracy）
    """
    # -------------------- 硬件资源配置 -------------------- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[VGG] Using device: {device}")  # 输出设备信息便于问题排查

    # -------------------- 输出目录管理 -------------------- #
    if output_dir is None:
        output_dir = Path(__file__).parents[1] / 'outputs' / model_type  # 按模型类型分目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # 自动创建不存在的目录

    # -------------------- 数据流水线 -------------------- #
    train_loader = get_cifar10_loader(batch_size, True)  # 训练数据加载器（启用shuffle）
    val_loader = get_cifar10_loader(batch_size * 2, False)  # 验证数据加载器（增大batch_size）

    # -------------------- 模型配置 -------------------- #
    num_classes = 10
    if model_type == 'vgg16':
        model = VGG16(num_classes=num_classes).to(device)
    elif model_type == 'vgg19':
        model = VGG19(num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")  # 适配CIFAR-10的10分类任务
    """
    VGG典型配置参数：
      - vgg16: conv_layers=[2,2,3,3,3] （13个卷积层）
      - vgg19: conv_layers=[2,2,4,4,4] （16个卷积层）
      每阶段后接2×2 MaxPooling
    """

    # -------------------- 优化策略 -------------------- #
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失（内置LogSoftmax + NLLLoss）

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,  # 初始学习率（比AlexNet更小，因参数量更大）
        momentum=momentum,  # 动量系数（缓解深层网络梯度消失）
        weight_decay=weight_decay  # 强L2正则（应对全连接层4096×4096参数）
    )

    # -------------------- 训练执行 -------------------- #
    history = fit(
        model=model,  # 要训练的VGG实例
        train_loader=train_loader,  # 训练数据迭代器
        val_loader=val_loader,  # 验证数据迭代器
        optimizer=optimizer,  # 优化器实例
        criterion=criterion,  # 损失函数实例
        device=device,  # 训练设备上下文
        epochs=epochs,  # 最大迭代次数（可能被早停终止）
        patience=patience,  # 早停观察窗口（连续无改进轮数）
        min_delta=min_delta,  # 改进阈值（需验证损失下降超过该值）
        log_interval=log_interval,  # 每100批次打印训练进度
        output_dir=str(output_dir),  # 模型检查点保存路径
        prefix=model_type  # 模型文件名前缀（如vgg16_epoch10.pth）
    )
    return history


if __name__ == "__main__":
    """命令行直接运行时执行默认训练配置"""
    run_vgg_training()  # 建议配置：vgg16 + 50epochs + lr=0.005
