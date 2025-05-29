# train/train_resnet.py
"""
ResNet-18 模型训练模块（CIFAR-10）

提供可在 Notebook 中调用的 run_resnet_training 函数，实现 ResNet-18 在 CIFAR-10 数据集上的完整训练流程。
包含数据加载、模型配置、优化器设置和训练过程封装。

依赖：
- models/resnet.py       : ResNet-18 模型定义
- utils/trainer.py       : 通用训练流程封装（包含早停、学习率调度等功能）
- utils/dataloader.py    : 数据集加载工具（支持自动下载与标准化预处理）
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from models.resnet import ResNet18
from utils.dataloader import get_dataloader, PROJECT_ROOT
from utils.trainer import fit

def get_cifar_loader(batch_size: int = 128, train: bool = True) -> DataLoader:
    """
    构造用于 ResNet-18 的 CIFAR-10 数据加载器。

    使用 [get_dataloader](file://utils/dataloader.py) 工具函数实现标准化数据加载流程，
    包含图像归一化等预处理操作。

    :param batch_size: 数据加载的批量大小，默认 128
    :type batch_size: int
    :param train: True 返回训练集；False 返回测试集
    :type train: bool
    :return: 配置完成的 CIFAR-10 数据加载器
    :rtype: DataLoader
    """
    root_dir = Path(PROJECT_ROOT) / 'data'
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为 [0,1] 范围的 Tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 标准化参数
    ])
    return get_dataloader(
        dataset_name='cifar-10',  # 指定加载 CIFAR-10 数据集
        batch_size=batch_size,  # 设置数据加载的批量大小
        train=train,  # 模式选择：True 加载训练集(50000样本)/False 加载测试集(10000样本)
        root=root_dir,  # 数据集存储根目录
        transform=transform,  # 应用预处理组合（张量转换→标准化）
        download=True,  # 当数据集不存在时自动下载
        shuffle=train  # 训练集启用数据打乱，验证/测试集保持顺序
    )

def run_resnet_training(
        *,
        epochs: int = 30,
        train_batch_size: int = 128,
        test_batch_size: int = 1000,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        patience: int = 5,
        min_delta: float = 0.001,
        log_interval: int = 100,
        output_dir: str = None,
) -> list[tuple[float, float]]:
    """
    在 CIFAR-10 数据集上执行 ResNet-18 的完整训练流程。

    使用 SGD + 动态学习率策略进行训练，支持早停机制和模型检查点保存，返回每个 epoch 的验证指标。

    :param epochs: 最大训练轮次，默认 30
    :type epochs: int
    :param train_batch_size: 训练数据批量大小，默认 128
    :type train_batch_size: int
    :param test_batch_size: 验证数据批量大小，默认 1000
    :type test_batch_size: int
    :param lr: 初始学习率，默认 0.1
    :type lr: float
    :param momentum: SGD 动量系数，默认 0.9
    :type momentum: float
    :param weight_decay: L2 正则化系数，默认 5e-4
    :type weight_decay: float
    :param patience: 早停耐心值（连续无改进轮次），默认 5
    :type patience: int
    :param min_delta: 损失改进最小识别阈值，默认 0.001
    :type min_delta: float
    :param log_interval: 训练日志打印间隔（按批次计数），默认 100
    :type log_interval: int
    :param output_dir: 模型输出目录，默认路径为项目根目录下的 outputs/resnet
    :type output_dir: str
    :return: 包含每个 epoch 验证指标 (loss, accuracy) 的列表
    :rtype: list[tuple[float, float]]
    """
    # 设备与路径准备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 优先使用 GPU
    print(f"[ResNet] Using device: {device}")

    # 输出目录处理逻辑（自动创建不存在的目录）
    if output_dir is None:
        output_dir = Path(__file__).parents[1] / 'outputs' / 'resnet'  # 默认存储路径
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 数据加载
    train_loader = get_cifar_loader(batch_size=train_batch_size, train=True)
    val_loader = get_cifar_loader(batch_size=test_batch_size, train=False)

    # 模型配置（适配 CIFAR-10 的 10 分类任务）
    model = ResNet18(num_classes=10).to(device)

    # 优化器配置（带 L2 正则化的带动量 SGD）
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,  # 基础学习率
        momentum=momentum,  # 动量系数（加速收敛）
        weight_decay=weight_decay  # 权重衰减系数（防止过拟合）
    )

    # 学习率调度器（当验证损失停滞时降低学习率）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # 当监测值（loss）停止下降时调整学习率
        factor=0.1,  # 学习率衰减因子（新学习率为原学习率 × factor）
        patience=2,  # 在调整学习率前等待的无改进 epoch 数
        verbose=True  # 打印学习率调整信息
    )

    # 调用通用训练流程（包含早停和模型保存）
    history = fit(
        model=model,  # 要训练的 ResNet-18 实例
        train_loader=train_loader,  # 训练数据加载器（批量 128）
        val_loader=val_loader,  # 验证数据加载器（批量 1000）
        optimizer=optimizer,  # 配置好的优化器
        criterion=nn.CrossEntropyLoss(),  # 交叉熵损失函数
        device=device,  # 训练设备
        epochs=epochs,  # 最大迭代次数
        patience=patience,  # 早停机制参数
        min_delta=min_delta,  # 损失改进识别阈值
        log_interval=log_interval,  # 每 100 批次打印日志
        output_dir=str(output_dir),  # 模型保存路径
        prefix='resnet',  # 模型文件名前缀（如 resnet_epoch3.pth）
        scheduler=scheduler  # 学习率调度器
    )
    return history


if __name__ == '__main__':
    # 脚本直接运行时执行默认配置训练
    run_resnet_training()  # 使用所有默认参数启动训练