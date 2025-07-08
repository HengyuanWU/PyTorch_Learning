# train/train_mlp.py
"""
两层多层感知机（MLP）模型训练模块（CIFAR-10）

提供可在 Notebook 中调用的 run_mlp_training 函数，实现 TwoLayerMLP 在 CIFAR-10 数据集上的完整训练流程。
包含数据加载、模型配置、优化器设置和训练过程封装。

依赖：
- models/mlp.py        : TwoLayerMLP 模型定义
- utils/trainer.py     : 通用训练流程封装（包含训练周期、验证、早停逻辑）
- utils/dataloader.py  : 数据集加载工具（支持自动下载与标准化预处理）
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional, List, Tuple, Union
import numpy as np

from models.mlp import TwoLayerMLP
from utils.dataloader import get_dataloader, PROJECT_ROOT
from utils.trainer import fit

def get_cifar10_loader(batch_size: int = 64, train: bool = True) -> DataLoader:
    """
    构造用于 MLP 的 CIFAR-10 数据加载器。

    使用通用数据加载工具 [get_dataloader](file:///C:/Users/WHY/Projects/Personal/PyTorch_Learning/utils/dataloader.py#L24-L65)，包含图像预处理、标准化等操作，
    并自动下载数据集。输入张量形状保持 [B, 3, 32, 32] 以适配 MLP 输入要求。

    :param batch_size: 单批次加载样本数量，默认 64
    :param train: True 加载训练集；False 加载测试集
    :return: 配置完成的 CIFAR-10 DataLoader
    """
    # 构建数据集存储路径（项目根目录下的data文件夹）
    root_dir = Path(PROJECT_ROOT) / 'data'

    # 定义数据预处理流程（顺序执行）
    # 1. 张量转换 -> 2. 标准化处理
    transform = transforms.Compose([
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

def run_mlp_training(
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
        output_dir: Optional[Union[str, Path]] = None,
) -> List[Tuple[float, float]]:
    """执行 MLP 模型的完整训练流程。

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
    :param output_dir: 输出目录路径，默认项目根目录下的 outputs/mlp
    :type output_dir: Optional[Union[str, Path]]
    :return: 各 epoch 验证指标列表，元素为 (loss, accuracy) 元组
    :rtype: List[Tuple[float, float]]
    """
    # -------------------- 硬件资源配置 -------------------- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 优先使用 GPU 加速
    print(f"[MLP] Using device: {device}")

    # -------------------- 输出目录管理 -------------------- #
    if output_dir is None:
        output_dir = Path(__file__).parents[1] / 'outputs' / 'mlp'  # 默认输出路径
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)  # 递归创建目录（忽略已存在情况）

    # -------------------- 数据流水线 -------------------- #
    train_loader = get_cifar10_loader(batch_size=train_batch_size, train=True)  # 训练数据加载器
    val_loader = get_cifar10_loader(batch_size=test_batch_size, train=False)  # 验证数据加载器

    # -------------------- 模型配置 -------------------- #
    # 确保使用最新的模型定义（适用于CIFAR-10的3通道图像）
    model = TwoLayerMLP(num_classes=10).to(device)  # 初始化模型（适配 CIFAR-10 的 10 分类任务）

    # -------------------- 优化策略 -------------------- #
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数（内置 softmax 计算）
    
    # 使用更小的学习率以提高稳定性
    if lr > 0.1:
        print(f"警告: 学习率 {lr} 可能过高，考虑降低到 0.01 以下")
    
    # 使用梯度裁剪以避免梯度爆炸
    max_grad_norm = 1.0
    
    optimizer = optim.SGD(
        model.parameters(),  # 需要优化的模型参数
        lr=min(lr, 0.01),    # 限制学习率上限
        momentum=momentum,  # 动量系数（累积梯度方向加速收敛）
        weight_decay=weight_decay  # L2 正则化项系数（权重衰减）
    )

    # 防止NaN值的修正函数
    def fix_nan_values(model):
        """检查并修复模型中的NaN参数值"""
        for name, param in model.named_parameters():
            if torch.isnan(param.data).any():
                print(f"检测到NaN值在参数 {name}，尝试修复...")
                # 使用小的随机值替换NaN值
                mask = torch.isnan(param.data)
                param.data[mask] = torch.randn_like(param.data[mask]) * 0.001
                print(f"已修复参数 {name}")

    # 梯度裁剪函数
    def clip_gradients(model, max_norm):
        """裁剪模型梯度，避免梯度爆炸"""
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    # 修改训练流程以包含梯度裁剪
    def custom_train_epoch(model, loader, optimizer, criterion, device, log_interval):
        model.train()
        running_loss = 0.
        for idx, (x, y) in enumerate(loader, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            
            # 检查损失是否为NaN
            if torch.isnan(loss):
                print(f"警告: 检测到NaN损失值! 跳过此批次更新。")
                # 尝试修复模型参数
                fix_nan_values(model)
                continue
                
            loss.backward()
            
            # 应用梯度裁剪
            clip_gradients(model, max_grad_norm)
            
            optimizer.step()
            
            # 在更新后检查参数是否有NaN
            has_nan = False
            for name, param in model.named_parameters():
                if torch.isnan(param.data).any():
                    has_nan = True
                    break
            
            if has_nan:
                print("警告: 更新后检测到NaN参数值，尝试修复...")
                fix_nan_values(model)
            
            running_loss += loss.item()
            if idx % log_interval == 0:
                print(f"[{idx*len(x)}/{len(loader.dataset)}] loss: {running_loss/log_interval:.4f}")
                running_loss = 0.

    # -------------------- 训练执行 -------------------- #
    # 使用自定义训练函数替换默认的训练函数
    history = fit(
        model=model,  # 要训练的 MLP 实例
        train_loader=train_loader,  # 训练数据加载器
        val_loader=val_loader,  # 验证数据加载器
        optimizer=optimizer,  # 配置好的优化器实例
        criterion=criterion,  # 损失函数实例
        device=device,  # 训练设备上下文
        epochs=epochs,  # 最大迭代次数（可能提前终止）
        patience=patience,  # 早停观察窗口大小
        min_delta=min_delta,  # 改进量阈值
        log_interval=log_interval,  # 日志打印频率
        output_dir=str(output_path),  # 模型保存路径
        prefix='mlp',  # 模型文件前缀
        custom_train_epoch_fn=custom_train_epoch  # 使用自定义的训练函数
    )
    return history

if __name__ == "__main__":
    # 命令行直接运行时执行默认配置训练
    run_mlp_training()  # 使用所有默认参数启动训练流程
