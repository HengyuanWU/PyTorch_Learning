# train/train_lenet.py

"""
训练 LeNet-5 模型（MNIST），并提供可在 Notebook 中调用的 run_lenet_training 函数

依赖：
- models/lenet.py       : LeNet-5 模型定义
- utils/trainer.py      : 通用训练流程封装（包含训练周期、验证、早停等功能）
- utils/dataloader.py   : 数据集加载工具（支持自动下载与标准化预处理）
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from models.lenet import LeNet5
from utils.dataloader import get_dataloader, PROJECT_ROOT
from utils.trainer import fit


def get_mnist_loader(batch_size: int = 64, train: bool = True) -> DataLoader:
    """构造用于 LeNet-5 的 MNIST 数据加载器。

    使用 [get_dataloader](file://utils/dataloader.py) 工具函数实现标准化数据加载流程，
    包含图像尺寸适配、归一化等预处理操作。

    :param batch_size: 数据加载的批量大小，默认 64
    :type batch_size: int
    :param train: True 返回训练集；False 返回测试集
    :type train: bool
    :return: 配置完成的 MNIST 数据加载器
    :rtype: DataLoader
    """
    root_dir = Path(PROJECT_ROOT) / 'data'
    transform = transforms.Compose([
        transforms.Pad(2),  # 填充 2 像素使 28×28 → 32×32（适配 LeNet 输入尺寸）
        transforms.ToTensor(),  # 转换为 [0,1] 范围的 Tensor
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 标准化参数（均值, 标准差）
    ])
    return get_dataloader(
        dataset_name='mnist',  # 指定加载 MNIST 数据集（内置支持的手写数字数据集）
        batch_size=batch_size,  # 设置数据加载的批量大小（影响梯度更新频率）
        train=train,  # 模式选择：True 加载训练集(60000样本)/False 加载测试集(10000样本)
        root=root_dir,  # 数据集存储根目录（自动下载时保存到此路径）
        transform=transform,  # 应用预处理组合（填充尺寸→张量转换→标准化）
        download=True,  # 当数据集不存在时自动下载 MNIST
        shuffle=train  # 训练集启用数据打乱（防止过拟合），验证/测试集保持顺序
    )

def run_lenet_training(
        *,
        epochs: int = 10,
        train_batch_size: int = 64,
        test_batch_size: int = 1000,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-5,
        patience: int = 3,
        min_delta: float = 0.0,
        log_interval: int = 100,
        output_dir: str = None,
) -> list[tuple[float, float]]:
    """在 MNIST 数据集上执行 LeNet-5 的完整训练流程。

    使用 SGD 优化器进行训练，支持早停机制和模型检查点保存，返回每个 epoch 的验证指标。

    :param epochs: 最大训练轮次，默认 10
    :type epochs: int
    :param train_batch_size: 训练数据批量大小，默认 64
    :type train_batch_size: int
    :param test_batch_size: 验证数据批量大小，默认 1000
    :type test_batch_size: int
    :param lr: 初始学习率，默认 0.01
    :type lr: float
    :param momentum: SGD 动量系数，默认 0.9
    :type momentum: float
    :param weight_decay: L2 正则化系数，默认 1e-5
    :type weight_decay: float
    :param patience: 早停耐心值（连续无改进轮次），默认 3
    :type patience: int
    :param min_delta: 损失改进最小识别阈值，默认 0.0
    :type min_delta: float
    :param log_interval: 训练日志打印间隔（按批次计数），默认 100
    :type log_interval: int
    :param output_dir: 模型输出目录，默认路径为项目根目录下的 outputs/lenet
    :type output_dir: str
    :return: 包含每个 epoch 验证指标 (loss, accuracy) 的列表
    :rtype: list[tuple[float, float]]
    """
    # 设备与路径准备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 优先使用 GPU
    print(f"[LeNet] Using device: {device}")

    # 输出目录处理逻辑（自动创建不存在的目录）
    if output_dir is None:
        output_dir = Path(__file__).parents[1] / 'outputs' / 'lenet'  # 默认存储路径
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 数据加载
    train_loader = get_mnist_loader(batch_size=train_batch_size, train=True)
    val_loader = get_mnist_loader(batch_size=test_batch_size, train=False)

    # 模型配置（适配 MNIST 的 10 分类任务）
    model = LeNet5(num_classes=10).to(device)

    # 优化器配置（带 L2 正则化的带动量 SGD）
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,  # 基础学习率
        momentum=momentum,  # 动量系数（加速收敛）
        weight_decay=weight_decay  # 权重衰减系数（防止过拟合）
    )

    # 调用通用训练流程（包含早停和模型保存）
    history = fit(
        model=model,  # 要训练的 LeNet-5 实例
        train_loader=train_loader,  # 训练数据加载器（批量 64）
        val_loader=val_loader,  # 验证数据加载器（批量 1000）
        optimizer=optimizer,  # 配置好的优化器
        criterion=nn.CrossEntropyLoss(),  # 交叉熵损失函数
        device=device,  # 训练设备
        epochs=epochs,  # 最大迭代次数
        patience=patience,  # 早停机制参数
        min_delta=min_delta,  # 损失改进识别阈值
        log_interval=log_interval,  # 每 100 批次打印日志
        output_dir=str(output_dir),  # 模型保存路径
        prefix='lenet'  # 模型文件名前缀（如 lenet_epoch3.pth）
    )
    return history


if __name__ == '__main__':
    # 脚本直接运行时执行默认配置训练
    run_lenet_training()  # 使用所有默认参数启动训练
