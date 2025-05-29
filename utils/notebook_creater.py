import json

# 注意这里使用原始字符串 r'''...'''，可以保留 JSON 里的所有转义符和双引号
notebook_str = r'''
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "\"\"\"\nResNet 模型定义与训练模块\n\n本模块实现 ResNet 系列模型（ResNet18/ResNet34 等），适配 CIFAR-10 图像分类任务。\n输入图像需调整为 224×224 大小，输出为 10 类别分类结果。\n\n典型 ResNet18 结构：\nInput (3×224×224)\n   ↓\nConv7x7(64) → BatchNorm → ReLU → MaxPool2d\n   ↓\n[Conv3x3(64) → BatchNorm → ReLU] ×2 → Downsample → [Conv3x3(128) → BatchNorm → ReLU] ×2 → …\n   ↓\nAveragePool → FC(10)\n\"\"\""
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "execution_count": null,
      "outputs": [],
      "source": [
        "from pathlib import Path\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "from torch.utils.data import DataLoader\n",
        "from torchvision import transforms\n",
        "from torchvision.datasets import CIFAR10\n",
        "\n",
        "from models.resnet import ResNet18, ResNet34  # ResNet 模型定义\n",
        "from utils import PROJECT_ROOT\n",
        "from utils.trainer import fit  # 通用训练流程封装\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "execution_count": null,
      "outputs": [],
      "source": [
        "def get_cifar10_loader(batch_size: int = 64, train: bool = True) -> DataLoader:\n",
        "    \"\"\"构建适配 ResNet 的 CIFAR-10 数据加载管道\n",
        "\n",
        "    输入处理流程：\n",
        "      CIFAR-10 (32×32) → Resize(224×224) → ToTensor → Normalize\n",
        "\n",
        "    :param batch_size: 单批次样本数（训练集建议64，验证集128）\n",
        "    :param train: 数据模式选择（True=训练集，False=验证集）\n",
        "    :return: 配置完成的数据加载器\n",
        "    \"\"\"\n",
        "    transform = transforms.Compose([\n",
        "        transforms.Resize((224, 224)),  # ResNet 标准输入尺寸\n",
        "        transforms.ToTensor(),\n",
        "        transforms.Normalize(\n",
        "            mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 三通道均值\n",
        "            std=[0.2023, 0.1994, 0.2010]    # CIFAR-10 三通道标准差\n",
        "        )\n",
        "    ])\n",
        "    root_dir = Path(PROJECT_ROOT) / 'data'  # 统一数据存储路径\n",
        "\n",
        "    return DataLoader(\n",
        "        CIFAR10(\n",
        "            root=root_dir,\n",
        "            train=train,\n",
        "            download=True,\n",
        "            transform=transform\n",
        "        ),\n",
        "        batch_size=batch_size,\n",
        "        shuffle=train,\n",
        "        num_workers=4,\n",
        "        pin_memory=True\n",
        "    )\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "execution_count": null,
      "outputs": [],
      "source": [
        "def run_resnet_training(\n",
        "        model_type: str = 'resnet18',  # 模型变体选择（resnet18含18层，resnet34含34层）\n",
        "        epochs: int = 100,              # 训练轮次\n",
        "        batch_size: int = 64,           # 训练批次大小\n",
        "        lr: float = 0.01,               # 初始学习率\n",
        "        momentum: float = 0.9,          # 动量系数\n",
        "        weight_decay: float = 1e-4,     # L2 正则化强度\n",
        "        patience: int = 5,              # 早停耐心值\n",
        "        min_delta: float = 1e-3,         # 损失最小改进阈值\n",
        "        log_interval: int = 100,        # 日志打印间隔\n",
        "        output_dir: str = None,         # 模型输出目录\n",
        ") -> list[tuple[float, float]]:\n",
        "    \"\"\"执行 ResNet 模型完整训练流程\n",
        "\n",
        "    :return: 各 epoch 验证指标列表（loss, accuracy）\n",
        "    \"\"\"\n",
        "    # 硬件资源配置\n",
        "    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "    print(f'[ResNet] Using device: {device}')\n",
        "\n",
        "    # 输出目录管理\n",
        "    if output_dir is None:\n",
        "        output_dir = Path(__file__).parents[1] / 'outputs' / model_type\n",
        "    output_dir = Path(output_dir)\n",
        "    output_dir.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "    # 数据流水线\n",
        "    train_loader = get_cifar10_loader(batch_size, True)\n",
        "    val_loader = get_cifar10_loader(batch_size * 2, False)\n",
        "\n",
        "    # 模型配置\n",
        "    num_classes = 10\n",
        "    if model_type == 'resnet18':\n",
        "        model = ResNet18(num_classes=num_classes).to(device)\n",
        "    elif model_type == 'resnet34':\n",
        "        model = ResNet34(num_classes=num_classes).to(device)\n",
        "    else:\n",
        "        raise ValueError(f'Unsupported model_type: {model_type}')\n",
        "\n",
        "    # 优化策略\n",
        "    criterion = nn.CrossEntropyLoss()\n",
        "    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)\n",
        "\n",
        "    # 训练执行\n",
        "    history = fit(\n",
        "        model=model,\n",
        "        train_loader=train_loader,\n",
        "        val_loader=val_loader,\n",
        "        optimizer=optimizer,\n",
        "        criterion=criterion,\n",
        "        device=device,\n",
        "        epochs=epochs,\n",
        "        patience=patience,\n",
        "        min_delta=min_delta,\n",
        "        log_interval=log_interval,\n",
        "        output_dir=str(output_dir),\n",
        "        prefix=model_type\n",
        "    )\n",
        "    return history\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "execution_count": null,
      "outputs": [],
      "source": [
        "if __name__ == '__main__':\n",
        "    # 命令行直接运行时执行默认训练配置\n",
        "    history = run_resnet_training()\n",
        "\n",
        "    # 可视化训练曲线\n",
        "    import matplotlib.pyplot as plt\n",
        "\n",
        "    epochs = list(range(1, len(history) + 1))\n",
        "    losses, accs = zip(*history)\n",
        "\n",
        "    plt.figure()\n",
        "    plt.plot(epochs, losses, label='Validation Loss')\n",
        "    plt.xlabel('Epoch')\n",
        "    plt.ylabel('Loss')\n",
        "    plt.title('ResNet Validation Loss')\n",
        "    plt.legend()\n",
        "    plt.show()\n",
        "\n",
        "    plt.figure()\n",
        "    plt.plot(epochs, accs, label='Validation Accuracy')\n",
        "    plt.xlabel('Epoch')\n",
        "    plt.ylabel('Accuracy')\n",
        "    plt.title('ResNet Validation Accuracy')\n",
        "    plt.legend()\n",
        "    plt.show()\n"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "version": "3.x"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
'''

# 解析并写入文件
notebook_json = json.loads(notebook_str)
with open("../notebooks/resnet.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, ensure_ascii=False, indent=2)

print("✅ 生成完成：resnet.ipynb")
