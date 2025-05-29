# utils/dataloader.py

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def find_project_root(marker_dirs=('utils', 'models', 'train')):
    """
    从当前文件开始，向上走，直到同时发现 marker_dirs 中列出的文件夹，
    就认定那里是项目根目录。
    """
    p = Path(__file__).resolve()
    for parent in p.parents:
        if all((parent / d).is_dir() for d in marker_dirs):
            return parent
    raise RuntimeError(f"Cannot find project root above {__file__}")


# 然后在文件任意位置调用：
PROJECT_ROOT = find_project_root()


def get_dataloader(
        dataset_name: str,
        batch_size: int = 64,
        train: bool = True,
        root: str = './data',
        transform: transforms.Compose = None,
        download: bool = True,
        shuffle: bool = None
) -> DataLoader:
    """
    获取指定数据集的 DataLoader。

    参数:
    - dataset_name: str, 数据集名称，支持 'mnist', 'cifar10' 等。
    - batch_size: int, 批次大小，默认为 64。
    - train: bool, 是否使用训练集，默认为 True。
    - root: str, 数据集存储路径。
    - transform: 可选的图像变换操作。
    - download: bool, 是否自动下载数据集。
    - shuffle: 是否打乱数据，默认为 None（根据 train 自动决定）。

    返回:
    - DataLoader: 构建好的 DataLoader 实例。
    """
    if transform is None:
        # 默认标准化参数适用于 MNIST 或 CIFAR-10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    if shuffle is None:
        shuffle = train

    if dataset_name.lower() == 'mnist':
        dataset = datasets.MNIST(root=root, train=train, download=download, transform=transform)
    elif dataset_name.lower() == 'cifar10':
        dataset = datasets.CIFAR10(root=root, train=train, download=download, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
