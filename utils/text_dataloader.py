# utils/text_dataloader.py

"""
文本数据集加载模块

提供文本类数据集（如AG_NEWS）的加载和预处理功能。
包含分词、词表构建、填充、掩码生成等操作。
"""

import torch
import os
import sys
import csv
import requests
import io
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Any, Tuple, Callable, Iterator, Union

# 检查 torchtext 可用性
try:
    import torchtext
    print(f"已成功导入 torchtext，版本: {torchtext.__version__}")
    TORCHTEXT_AVAILABLE = True
except ImportError as e:
    TORCHTEXT_AVAILABLE = False
    print(f"警告: torchtext 导入失败: {e}")
    print("请安装 torchtext: pip install torchtext==0.14.0")
except Exception as e:
    TORCHTEXT_AVAILABLE = False
    print(f"警告: torchtext 初始化失败: {e}")
    print(f"Python 路径: {sys.path}")
    print("请确保安装了兼容的 torchtext 版本: pip install torchtext==0.14.0")


# AG_NEWS数据集URLs
AG_NEWS_URLS = {
    "train": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
    "test": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
}


class AGNewsDataset(Dataset):
    """AG_NEWS 数据集包装类"""
    def __init__(self, data_items):
        """
        初始化 AG_NEWS 数据集
        
        :param data_items: 预处理后的数据项列表，每项包含 (token_ids, label, mask)
        """
        self.data = data_items
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def simple_tokenizer(text):
    """简单的空格分词器"""
    return text.lower().split()


def build_vocab_from_data(texts, min_freq=1):
    """
    从文本数据构建词表
    
    :param texts: 文本数据列表
    :param min_freq: 最小词频
    :return: 词表字典，从词到索引的映射
    """
    counter = {}
    for text in texts:
        tokens = simple_tokenizer(text)
        for token in tokens:
            counter[token] = counter.get(token, 0) + 1
    
    # 按词频排序
    sorted_tokens = sorted([(freq, token) for token, freq in counter.items() if freq >= min_freq], 
                          reverse=True)
    
    # 创建词表
    vocab = {"<unk>": 0, "<pad>": 1}
    for _, token in sorted_tokens:
        vocab[token] = len(vocab)
    
    return vocab


class Vocab:
    """词表类，提供从词到ID的映射"""
    def __init__(self, vocab_dict):
        self.stoi = vocab_dict  # string to index
        self.itos = {idx: word for word, idx in vocab_dict.items()}  # index to string
        self.unk_idx = vocab_dict["<unk>"]
    
    def __getitem__(self, token):
        return self.stoi.get(token, self.unk_idx)
    
    def __len__(self):
        return len(self.stoi)


def download_and_parse_csv(url, root_dir):
    """
    下载并解析CSV数据集文件
    
    :param url: 数据集URL
    :param root_dir: 保存路径
    :return: 解析后的数据 [(label, text)]
    """
    # 确保目录存在
    os.makedirs(root_dir, exist_ok=True)
    
    # 构建本地文件路径
    filename = url.split('/')[-1]
    local_path = os.path.join(root_dir, filename)
    
    # 检查文件是否已存在
    if not os.path.exists(local_path):
        print(f"下载数据集: {url}")
        response = requests.get(url)
        response.raise_for_status()
        
        # 保存文件
        with open(local_path, 'wb') as f:
            f.write(response.content)
        print(f"已保存至: {local_path}")
    else:
        print(f"使用本地缓存: {local_path}")
    
    # 读取并解析CSV
    data = []
    with open(local_path, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if len(row) >= 2:
                label = int(row[0])  # 第一列是标签 (1-4)
                text = ' '.join(row[1:])  # 剩余列是文本
                data.append((label, text))
    
    return data


def get_ag_news_dataloader(batch_size=64, max_len=256, train=True, root='./data'):
    """
    获取 AG_NEWS 数据集的 DataLoader。

    参数:
    - batch_size: int, 批次大小，默认为 64。
    - max_len: int, 文本序列最大长度，默认为 256。
    - train: bool, 是否使用训练集，默认为 True。
    - root: str, 数据集存储路径，默认为 './data'。

    返回:
    - dataloader: AG_NEWS 数据集的 DataLoader 实例
    - vocab: 词汇表对象
    - vocab_size: 词汇表大小
    - num_classes: 类别数量 (4 for AG_NEWS)
    """
    # 确保数据目录存在
    data_dir = os.path.join(root, 'ag_news')
    os.makedirs(data_dir, exist_ok=True)
    
    # 下载并解析训练集(用于构建词表)
    train_data = download_and_parse_csv(AG_NEWS_URLS["train"], data_dir)
    
    # 如果是测试集，也下载解析
    if not train:
        data = download_and_parse_csv(AG_NEWS_URLS["test"], data_dir)
    else:
        data = train_data
    
    # 构建词表(仅使用训练集)
    print("构建词表...")
    train_texts = [text for _, text in train_data]
    vocab_dict = build_vocab_from_data(train_texts)
    vocab_obj = Vocab(vocab_dict)
    vocab_size = len(vocab_obj)
    print(f"词表大小: {vocab_size}")
    
    # 处理数据集
    processed_data = []
    print(f"处理{'训练' if train else '测试'}集...")
    
    for label, text in data:
        # 分词
        tokens = simple_tokenizer(text)
        
        # 转为索引
        token_ids = [vocab_obj[token] for token in tokens]
        
        # 截断或填充
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        else:
            token_ids.extend([vocab_obj["<pad>"]] * (max_len - len(token_ids)))
        
        # 创建填充掩码
        mask = torch.zeros(max_len, dtype=torch.bool)
        if len(tokens) < max_len:
            mask[len(tokens):] = True
            
        # AG_NEWS 标签从 1 开始，减 1 使其从 0 开始
        processed_data.append(
            (torch.tensor(token_ids, dtype=torch.long),
             torch.tensor(label - 1, dtype=torch.long),
             mask)
        )
    
    # 创建 Dataset
    dataset = AGNewsDataset(processed_data)
    
    # 创建 DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=train, 
        collate_fn=lambda batch: (
            torch.stack([item[0] for item in batch]),
            torch.tensor([item[1] for item in batch]),
            torch.stack([item[2] for item in batch])
        )
    )
    
    return dataloader, vocab_obj, vocab_size, 4  # AG_NEWS 有 4 个类别 