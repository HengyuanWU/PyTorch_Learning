# utils/trainer.py

import os
from pathlib import Path
import torch

def train_epoch(model, loader, optimizer, criterion, device, log_interval):
    model.train()
    running_loss = 0.
    for idx, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if idx % log_interval == 0:
            print(f"[{idx*len(x)}/{len(loader.dataset)}] loss: {running_loss/log_interval:.4f}")
            running_loss = 0.

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0., 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
    avg_loss = total_loss / len(loader.dataset)
    acc = 100. * correct / len(loader.dataset)
    print(f"Val loss: {avg_loss:.4f}, Acc: {acc:.2f}%")
    return avg_loss, acc

def fit(model,
        train_loader, val_loader,
        optimizer, criterion,
        device,
        epochs=10,
        patience=3,
        min_delta=0.,
        log_interval=100,
        output_dir="outputs",
        prefix="model",
        custom_train_epoch_fn=None):
    """
    通用训练流程，支持早停和检查点保存
    
    :param model: 要训练的模型
    :param train_loader: 训练数据加载器
    :param val_loader: 验证数据加载器
    :param optimizer: 优化器
    :param criterion: 损失函数
    :param device: 训练设备
    :param epochs: 训练轮数
    :param patience: 早停耐心值
    :param min_delta: 最小改进阈值
    :param log_interval: 日志打印间隔
    :param output_dir: 输出目录
    :param prefix: 保存文件前缀
    :param custom_train_epoch_fn: 可选的自定义训练函数，用于替代默认的train_epoch函数
    :return: 历史指标列表
    """
    # 确保输出目录存在
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    best_loss = float('inf')
    no_improve = 0
    best_path = os.path.join(output_dir, f"{prefix}_best.pth")
    history = []

    # 使用自定义训练函数或默认函数
    train_fn = custom_train_epoch_fn if custom_train_epoch_fn else train_epoch

    for epoch in range(1, epochs+1):
        print(f"\n=== Epoch {epoch} ===")
        train_fn(model, train_loader, optimizer, criterion, device, log_interval)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history.append((val_loss, val_acc))

        # Early Stopping
        if best_loss - val_loss > min_delta:
            best_loss, no_improve = val_loss, 0
            torch.save(model.state_dict(), best_path)
            print(f"  → loss improved, saved to {best_path}")
        else:
            no_improve += 1
            print(f"  → no improvement ({no_improve}/{patience})")

        # always save checkpoint
        ckpt = os.path.join(output_dir, f"{prefix}_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt)

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            try:
                if os.path.exists(best_path):
                    model.load_state_dict(torch.load(best_path, weights_only=True))
                else:
                    print(f"Warning: Best model file {best_path} not found. Using current model state.")
            except Exception as e:
                print(f"Error loading best model: {e}. Using current model state.")
            break

    return history
