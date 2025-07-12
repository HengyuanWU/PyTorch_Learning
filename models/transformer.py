"""
Transformer 模型定义模块

本模块实现了用于文本分类的 Transformer 模型，基于 PyTorch 的 nn.TransformerEncoder 构建。
模型包含词嵌入层、位置编码和 Transformer 编码器，适用于各类文本分类任务（如 AG_NEWS 新闻分类）。

包含的类：
- Transformer: 实现文本分类 Transformer 模型的 PyTorch 模型类

输入（文本序列）
   ↓
词嵌入 + 位置编码  → [batch_size, seq_len, embed_dim]
   ↓
TransformerEncoder  → [batch_size, seq_len, embed_dim]
   ↓
平均池化（序列维度） → [batch_size, embed_dim]
   ↓
全连接层 → [batch_size, num_classes]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    位置编码模块，为序列中的每个位置生成唯一的编码
    
    实现基于 Transformer 论文中的正弦/余弦位置编码方法：
    PE(pos, 2i)   = sin(pos/10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
    """
    def __init__(self, embed_dim, max_len=5000):
        """
        初始化位置编码模块
        
        :param embed_dim: 词嵌入维度
        :param max_len: 支持的最大序列长度，默认5000
        """
        super().__init__()
        # 创建一个足够大的位置编码矩阵 (max_len, embed_dim)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        # 应用正弦和余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加批次维度并注册为非参数缓冲区
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        添加位置编码到输入嵌入中
        
        :param x: 输入嵌入 [batch_size, seq_len, embed_dim]
        :return: 位置编码加入后的嵌入 [batch_size, seq_len, embed_dim]
        """
        return x + self.pe[:, :x.size(1)]


class Transformer(nn.Module):
    """
    用于文本分类的 Transformer 模型：
      - 输入：文本序列（token ID）
      - 词嵌入 + 位置编码
      - TransformerEncoder 层（多个编码器层堆叠）
      - 序列平均池化
      - 全连接分类层
    """
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, num_classes, dropout=0.1):
        """
        初始化 Transformer 模型
        
        :param vocab_size: 词表大小
        :param embed_dim: 词嵌入维度
        :param num_heads: 注意力头数量
        :param hidden_dim: 前馈网络隐藏层维度
        :param num_layers: Transformer 编码器层数
        :param num_classes: 分类类别数
        :param dropout: Dropout 比例，默认 0.1
        """
        super().__init__()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 位置编码层
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        # Transformer 编码器层定义
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True  # 输入格式为 [batch, seq, features]
        )
        
        # 堆叠多个编码器层
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出分类层
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # 初始化参数
        self._init_parameters()
        
    def _init_parameters(self):
        """初始化模型参数"""
        # 初始化嵌入层
        nn.init.xavier_uniform_(self.embedding.weight)
        
        # 初始化分类器权重
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, x, src_key_padding_mask=None):
        """
        前向传播
        
        :param x: 输入序列张量，形状 [batch_size, seq_len]
        :param src_key_padding_mask: 源序列的填充掩码，形状 [batch_size, seq_len]
                                    其中 True 表示被掩盖的位置（填充位置）
        :return: logits，形状 [batch_size, num_classes]
        """
        # 词嵌入 [batch_size, seq_len] -> [batch_size, seq_len, embed_dim]
        embedded = self.embedding(x)
        
        # 添加位置编码
        embedded = self.pos_encoder(embedded)
        
        # 通过 Transformer 编码器
        # src_key_padding_mask 中 True 表示需要掩盖的位置（如填充位置）
        encoded = self.transformer_encoder(embedded, src_key_padding_mask=src_key_padding_mask)
        
        # 平均池化所有token的表示（忽略掩码位置）
        if src_key_padding_mask is not None:
            # 创建反向掩码：False 表示有效位置，True 表示填充位置
            mask = ~src_key_padding_mask
            # 将 mask 扩展为与 encoded 相同的维度
            mask = mask.unsqueeze(-1).expand_as(encoded).float()
            # 仅对有效位置计算平均值
            pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            # 如果没有掩码，简单地在序列维度上平均
            pooled = encoded.mean(dim=1)
        
        # 分类输出
        logits = self.classifier(pooled)
        return logits 