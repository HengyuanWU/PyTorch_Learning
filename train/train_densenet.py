# train/train_densenet.py

"""
训练 DenseNet 模型（CIFAR-10），支持两种架构：
1. 标准 DenseNet-121
2. CIFAR-10 定制版 DenseNet (L=100, k=12, θ=0.5)

依赖：
- models/densenet.py      : DenseNet 模型定义
- utils/trainer.py        : 通用训练流程封装
- utils/dataloader.py     : 数据集加载工具
"""

import os
from pathlib import Path
from typing import Literal, Optional, List, Tuple, cast, Protocol

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from models.densenet import DenseNet, DenseNetCustom
from utils.dataloader import get_dataloader, PROJECT_ROOT
from utils.trainer import fit

# 类型定义
ModelType = Literal['densenet121', 'densenetcustom']

# 定义一个协议类型，确保对象有__len__方法
class Sized(Protocol):
    def __len__(self) -> int: ...

def get_cifar10_loader(batch_size: int = 64, train: bool = True) -> DataLoader:
    """构造用于 DenseNet 的 CIFAR-10 数据加载器。
    
    使用 [get_dataloader](file://utils/dataloader.py) 工具函数实现标准化数据加载流程，
    包含数据增强、归一化等预处理操作。

    :param batch_size: 数据加载的批量大小，默认 64
    :type batch_size: int
    :param train: True 返回训练集；False 返回测试集
    :type train: bool
    :return: 配置完成的 CIFAR-10 数据加载器
    :rtype: DataLoader
    """
    root_dir = Path(PROJECT_ROOT) / 'data'
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
        transforms.RandomCrop(32, padding=4) if train else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    return get_dataloader(
        dataset_name='cifar10',
        batch_size=batch_size,
        train=train,
        root=str(root_dir),
        transform=transform,
        download=True,
        shuffle=train
    )

def run_densenet_training(
        model_type: ModelType = 'densenetcustom',
        *,
        epochs: int = 300,
        train_batch_size: int = 64,
        test_batch_size: int = 256,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        patience: int = 10,
        min_delta: float = 0.001,
        log_interval: int = 100,
        output_dir: Optional[str] = None,
        optimiser: str = 'sgd',  # 添加optimiser参数，默认为sgd
        silent: bool = False,     # 添加silent参数，默认为False
) -> List[Tuple[float, float]]:
    """在 CIFAR-10 数据集上执行 DenseNet 的完整训练流程。

    使用 SGD 优化器进行训练，支持学习率调度、早停机制和模型检查点保存。

    :param model_type: 模型类型 ('densenet121' 或 'densenetcustom')
    :type model_type: str
    :param epochs: 最大训练轮次，默认 300
    :type epochs: int
    :param train_batch_size: 训练数据批量大小，默认 64
    :type train_batch_size: int
    :param test_batch_size: 验证数据批量大小，默认 256
    :type test_batch_size: int
    :param lr: 初始学习率，默认 0.1
    :type lr: float
    :param momentum: SGD 动量系数，默认 0.9
    :type momentum: float
    :param weight_decay: L2 正则化系数，默认 1e-4
    :type weight_decay: float
    :param patience: 早停耐心值（连续无改进轮次），默认 10
    :type patience: int
    :param min_delta: 损失改进最小识别阈值，默认 0.001
    :type min_delta: float
    :param log_interval: 训练日志打印间隔（按批次计数），默认 100
    :type log_interval: int
    :param output_dir: 模型输出目录，默认路径为项目根目录下的 outputs/densenet
    :type output_dir: str
    :param optimiser: 优化器类型 ('sgd' 或 'adam')，默认 'sgd'
    :type optimiser: str
    :param silent: 是否禁止输出训练过程中的日志信息，默认False
    :type silent: bool
    :return: 包含每个 epoch 验证指标 (loss, accuracy) 的列表
    :rtype: List[Tuple[float, float]]
    """
    # 设备与路径准备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not silent:
        print(f"[DenseNet] Using device: {device}")

    # 输出目录处理
    if output_dir is None:
        output_dir = str(Path(__file__).parents[1] / 'outputs' / 'densenet')
    else:
        output_dir = str(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 数据加载
    train_loader = get_cifar10_loader(batch_size=train_batch_size, train=True)
    val_loader = get_cifar10_loader(batch_size=test_batch_size, train=False)
    
    # 强制转换为有__len__方法的类型
    train_dataset = cast(Sized, train_loader.dataset)
    val_dataset = cast(Sized, val_loader.dataset)
    
    # 模型初始化
    if model_type == 'densenet121':
        model = DenseNet(num_classes=10).to(device)
    else:
        model = DenseNetCustom(num_classes=10).to(device)
    if not silent:
        print(f"[DenseNet] Initialized {model_type} model with {sum(p.numel() for p in model.parameters()):,} parameters")

    # 优化器配置
    if optimiser.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:  # 默认使用SGD
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True
        )
    
    # 学习率调度（在1/2和3/4训练进度时降低学习率）
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[epochs//2, epochs*3//4],
        gamma=0.1
    )

    # 自定义训练循环（因为需要手动调用scheduler）
    os.makedirs(output_dir, exist_ok=True)
    best_loss = float('inf')
    no_improve = 0
    best_path = os.path.join(output_dir, f"{model_type}_best.pth")
    history: List[Tuple[float, float]] = []

    for epoch in range(1, epochs+1):
        if not silent:
            print(f"\n=== Epoch {epoch} ===")
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        for idx, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if idx % log_interval == 0 and not silent:
                print(f"[{idx*len(x)}/{len(train_dataset)}] loss: {running_loss/log_interval:.4f}")
                running_loss = 0.0
        
        # 验证阶段
        model.eval()
        total_loss, correct = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                total_loss += nn.CrossEntropyLoss()(logits, y).item() * x.size(0)
                correct += (logits.argmax(1) == y).sum().item()
        avg_loss = total_loss / len(val_dataset)
        acc = 100.0 * correct / len(val_dataset)
        if not silent:
            print(f"Val loss: {avg_loss:.4f}, Acc: {acc:.2f}%")
        history.append((avg_loss, acc))
        
        # 学习率调度
        scheduler.step()
        
        # 早停和模型保存
        if best_loss - avg_loss > min_delta:
            best_loss, no_improve = avg_loss, 0
            torch.save(model.state_dict(), best_path)
            if not silent:
                print(f"  → loss improved, saved to {best_path}")
        else:
            no_improve += 1
            if not silent:
                print(f"  → no improvement ({no_improve}/{patience})")
            
        # 保存检查点
        ckpt = os.path.join(output_dir, f"{model_type}_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt)
        
        if no_improve >= patience:
            if not silent:
                print(f"Early stopping at epoch {epoch}")
            model.load_state_dict(torch.load(best_path))
            break
            
    return history

if __name__ == '__main__':
    # 脚本直接运行时执行默认配置训练（使用定制版DenseNet）
    run_densenet_training(model_type='densenetcustom') 