# train/train_transformer.py

"""
训练 Transformer 模型（AG_NEWS 文本分类），并提供可在 Notebook 中调用的 run_transformer_training 函数

依赖：
- models/transformer.py  : Transformer 模型定义
- utils/trainer.py       : 通用训练流程封装（包含训练周期、验证、早停等功能）
- utils/text_dataloader.py : 文本数据集加载工具（AG_NEWS 处理）
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from models.transformer import Transformer
from utils.text_dataloader import get_ag_news_dataloader
from utils.dataloader import PROJECT_ROOT
from utils.trainer import fit


def get_ag_news_loaders(batch_size: int = 64, max_len: int = 256):
    """构造用于 Transformer 模型的 AG_NEWS 数据加载器。

    使用 utils/text_dataloader.py 中定义的 get_ag_news_dataloader 函数实现数据加载流程，
    包含分词、词表构建、填充等预处理操作。

    :param batch_size: 数据加载的批量大小，默认 64
    :param max_len: 文本序列最大长度，默认 256
    :return: 训练加载器, 验证加载器, 词表大小, 类别数
    """
    # 设置数据存储路径
    data_dir = Path(PROJECT_ROOT) / 'data'
    
    train_loader, vocab, vocab_size, num_classes = get_ag_news_dataloader(
        batch_size=batch_size, 
        max_len=max_len,
        train=True,
        root=str(data_dir)
    )
    
    val_loader, _, _, _ = get_ag_news_dataloader(
        batch_size=batch_size, 
        max_len=max_len,
        train=False,
        root=str(data_dir)
    )
    
    return train_loader, val_loader, vocab_size, num_classes


def run_transformer_training(
        *,
        epochs: int = 5,
        train_batch_size: int = 32,
        val_batch_size: int = 64,
        max_len: int = 256,
        embed_dim: int = 128,
        num_heads: int = 4,
        hidden_dim: int = 512,
        num_layers: int = 3,
        lr: float = 5e-4,
        weight_decay: float = 1e-5,
        patience: int = 2,
        min_delta: float = 0.0,
        log_interval: int = 100,
        output_dir: str = "",
) -> list[tuple[float, float]]:
    """在 AG_NEWS 数据集上执行 Transformer 的完整训练流程。

    使用 AdamW 优化器进行训练，支持早停机制和模型检查点保存，返回每个 epoch 的验证指标。

    :param epochs: 最大训练轮次，默认 5
    :param train_batch_size: 训练数据批量大小，默认 32
    :param val_batch_size: 验证数据批量大小，默认 64
    :param max_len: 文本序列最大长度，默认 256
    :param embed_dim: 词嵌入维度，默认 128
    :param num_heads: 注意力头数量，默认 4
    :param hidden_dim: 前馈网络隐藏层维度，默认 512
    :param num_layers: Transformer编码器层数，默认 3
    :param lr: 初始学习率，默认 5e-4
    :param weight_decay: L2 正则化系数，默认 1e-5
    :param patience: 早停耐心值（连续无改进轮次），默认 2
    :param min_delta: 损失改进最小识别阈值，默认 0.0
    :param log_interval: 训练日志打印间隔（按批次计数），默认 100
    :param output_dir: 模型输出目录，默认路径为项目根目录下的 outputs/transformer
    :return: 包含每个 epoch 验证指标 (loss, accuracy) 的列表
    """
    # 设备与路径准备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Transformer] Using device: {device}")

    # 输出目录处理逻辑（自动创建不存在的目录）
    if not output_dir:
        output_dir = str(Path(__file__).parents[1] / 'outputs' / 'transformer')
    os.makedirs(output_dir, exist_ok=True)

    # 数据加载
    train_loader, val_loader, vocab_size, num_classes = get_ag_news_loaders(
        batch_size=train_batch_size,
        max_len=max_len
    )
    print(f"词表大小: {vocab_size}, 类别数: {num_classes}")

    # 模型配置
    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes
    ).to(device)

    # 优化器配置
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # 自定义训练函数处理掩码
    def train_epoch_with_mask(model, loader, optimizer, criterion, device, log_interval):
        model.train()
        running_loss = 0.
        for idx, (tokens, labels, masks) in enumerate(loader, 1):
            tokens, labels, masks = tokens.to(device), labels.to(device), masks.to(device)
            
            optimizer.zero_grad()
            # 将掩码传递给模型
            outputs = model(tokens, src_key_padding_mask=masks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if idx % log_interval == 0:
                print(f"[{idx*len(tokens)}/{len(loader.dataset)}] loss: {running_loss/log_interval:.4f}")
                running_loss = 0.
    
    # 自定义验证函数处理掩码
    def evaluate_with_mask(model, loader, criterion, device):
        model.eval()
        total_loss, correct = 0., 0
        with torch.no_grad():
            for tokens, labels, masks in loader:
                tokens, labels, masks = tokens.to(device), labels.to(device), masks.to(device)
                # 将掩码传递给模型
                logits = model(tokens, src_key_padding_mask=masks)
                total_loss += criterion(logits, labels).item() * tokens.size(0)
                correct += (logits.argmax(1) == labels).sum().item()
        avg_loss = total_loss / len(loader.dataset)
        acc = 100. * correct / len(loader.dataset)
        print(f"Val loss: {avg_loss:.4f}, Acc: {acc:.2f}%")
        return avg_loss, acc

    # 调用通用训练流程（传递自定义训练函数）
    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        epochs=epochs,
        patience=patience,
        min_delta=min_delta,
        log_interval=log_interval,
        output_dir=output_dir,
        prefix='transformer',
        custom_train_epoch_fn=train_epoch_with_mask
    )
    return history


if __name__ == '__main__':
    # 脚本直接运行时执行默认配置训练
    run_transformer_training() 