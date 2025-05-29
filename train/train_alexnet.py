# train/train_alexnet.py

"""
AlexNet 模型训练系统（CIFAR-10 数据集）

功能特性：
- 支持通过脚本直接运行训练
- 提供可配置的训练参数接口
- 实现早停机制和模型检查点保存

依赖模块：
- models/alexnet.py    : AlexNet 网络结构实现
- utils/trainer.py     : 通用训练流程（含训练周期、验证、早停逻辑）
- torchvision 0.13+    : 提供 CIFAR-10 数据集加载和预处理能力
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from models.alexnet import AlexNet
from utils import PROJECT_ROOT, get_dataloader
from utils.trainer import fit


def get_cifar10_loader(batch_size: int = 64, train: bool = True) -> DataLoader:
    """
    构造用于 AlexNet 的 CIFAR-10 数据加载器。

    使用通用数据加载工具 `get_dataloader`，包含图像预处理、标准化等操作，
    并自动下载数据集。

    :param batch_size: 单批次加载样本数量，默认 64
    :param train: True 加载训练集；False 加载测试集
    :return: 配置完成的 CIFAR-10 DataLoader
    """
    # 构建数据集存储路径（项目根目录下的data文件夹）
    root_dir = Path(PROJECT_ROOT) / 'data'

    # 定义数据预处理流程（顺序执行）
    # 1. 尺寸调整 -> 2. 张量转换 -> 3. 标准化处理
    transform = transforms.Compose([
        transforms.Resize((227, 227)),  # 缩放至 227×227（满足 AlexNet 输入尺寸）
        transforms.ToTensor(),  # 转为 [0,1] 范围的 Tensor
        transforms.Normalize(  # CIFAR-10 三通道标准化
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])

    # 创建数据加载器并返回
    return get_dataloader(
        dataset_name='cifar10',    # 使用 CIFAR-10 数据集
        batch_size=batch_size,     # 用户指定的批次大小
        train=train,               # 训练/验证模式选择
        root=str(root_dir),        # 数据集存储根路径
        transform=transform,       # 应用定义的数据预处理流程
        download=True,             # 自动下载缺失数据集
        shuffle=train              # 训练集打乱顺序，验证集保持顺序
    )


def run_alexnet_training(
        *,
        epochs: int = 20,
        train_batch_size: int = 64,
        test_batch_size: int = 100,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        patience: int = 5,
        min_delta: float = 1e-3,
        log_interval: int = 200,
        output_dir: str = None,
) -> list[tuple[float, float]]:
    """执行 AlexNet 模型的完整训练流程。

    使用带动量的 SGD 优化器，支持 L2 正则化和早停机制，返回每个 epoch 的验证指标。

    :param epochs: 最大训练轮次，默认 20
    :type epochs: int
    :param train_batch_size: 训练集批量大小，默认 64
    :type train_batch_size: int
    :param test_batch_size: 验证集批量大小，默认 100
    :type test_batch_size: int
    :param lr: 初始学习率，默认 0.01
    :type lr: float
    :param momentum: SGD 动量系数，默认 0.9（缓解局部极小值问题）
    :type momentum: float
    :param weight_decay: L2 正则化强度，默认 1e-4（控制模型复杂度）
    :type weight_decay: float
    :param patience: 早停耐心值，默认 5（连续 5 轮验证损失无改进则终止）
    :type patience: int
    :param min_delta: 最小改进阈值，默认 0.001（损失变化小于此值视为无改进）
    :type min_delta: float
    :param log_interval: 日志间隔，默认 200 批次（控制训练日志输出频率）
    :type log_interval: int
    :param output_dir: 输出目录路径，默认项目根目录下的 outputs/alexnet
    :type output_dir: str
    :return: 各 epoch 验证指标列表，元素为 (loss, accuracy) 元组
    :rtype: list[tuple[float, float]]
    """
    # -------------------- 硬件资源配置 -------------------- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 优先使用 GPU 加速
    print(f"[AlexNet] Using device: {device}")

    # -------------------- 输出目录管理 -------------------- #
    if output_dir is None:
        output_dir = Path(__file__).parents[1] / 'outputs' / 'alexnet'  # 默认输出路径
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # 递归创建目录（忽略已存在情况）

    # -------------------- 数据流水线 -------------------- #
    train_loader = get_cifar10_loader(batch_size=train_batch_size, train=True)  # 训练数据加载器
    val_loader = get_cifar10_loader(batch_size=test_batch_size, train=False)  # 验证数据加载器

    # -------------------- 模型配置 -------------------- #
    model = AlexNet(num_classes=10).to(device)  # 初始化模型（适配 CIFAR-10 的 10 分类任务）

    # -------------------- 优化策略 -------------------- #
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数（内置 softmax 计算）
    optimizer = optim.SGD(
        model.parameters(),  # 需要优化的模型参数
        lr=lr,  # 基础学习率（配合学习率调度器效果更佳）
        momentum=momentum,  # 动量系数（累积梯度方向加速收敛）
        weight_decay=weight_decay  # L2 正则化项系数（权重衰减）
    )

    # -------------------- 训练执行 -------------------- #
    history = fit(
        model=model,  # 要训练的 AlexNet 实例
        train_loader=train_loader,  # 训练数据加载器（批量 64）
        val_loader=val_loader,  # 验证数据加载器（批量 100）
        optimizer=optimizer,  # 配置好的优化器实例
        criterion=criterion,  # 损失函数实例
        device=device,  # 训练设备上下文
        epochs=epochs,  # 最大迭代次数（可能提前终止）
        patience=patience,  # 早停观察窗口大小（连续 patience 轮无改进）
        min_delta=min_delta,  # 改进量阈值（需 val_loss 下降超过该值）
        log_interval=log_interval,  # 每 200 个批次打印进度信息
        output_dir=str(output_dir),  # 模型检查点保存路径
        prefix='alexnet'  # 模型文件前缀（生成如 alexnet_epoch5.pth）
    )
    return history


if __name__ == "__main__":
    # 命令行直接运行时执行默认配置训练
    run_alexnet_training()  # 使用所有默认参数启动训练流程
