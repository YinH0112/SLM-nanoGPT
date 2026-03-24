import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 512 # 最大输入序列长度
    batch_size: int = 12 # 批次大小
    n_layer: int = 12 # 层数量
    n_head: int = 12 # 头数
    n_embd: int = 768 # 嵌入维度
    hidden_dim: int = n_embd
    dropout: float = 0.1
    head_size: int = n_embd // n_head
    vocab_size: int = 50257

class SingleHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.query = nn.Linear(config.hidden_dim, config.head_size)
        self.key = nn.Linear(config.hidden_dim, config.head_size)
        self.value = nn.Linear(config.hidden_dim, config.head_size)
       
        self.register_buffer(
            "attention_mask",
            torch.tril(
                torch.ones(config.block_size, config.block_size)
            )
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, hidden_dim = x.shape
        # 批次大小，序列长度，隐藏维度
        q, k, v = self.query(x), self.key(x), self.value(x)
        
        weight = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)
        weight = weight.masked_fill(
            self.attention_mask[:seq_len, :seq_len] == 0,
            float("-inf")
        )
        weight = F.softmax(weight, dim=-1)

        weight = self.dropout(weight)

        output = weight @ v
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                SingleHeadAttention(config)
                for _ in range(config.n_head)
            ]
        )
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        output = torch.cat(
            [h(x) for h in self.heads],
            dim=-1
        )

        output = self.proj(output)
        output = self.dropout(output)
        return output

class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.att = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x
    
class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(
            config.vocab_size,
            config.hidden_dim
        )
        self.position_embedding_table = nn.Embedding(
            config.block_size,
            config.n_embd
        )
        
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )

        self.ln_final = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        # 现在的SLM, 会用tie weight减少参数
        # 即token_embedding_table和lm_head的权重相同
        self.token_embedding_table.weight = self.lm_head.weight

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            # 初始化权重为正态分布，标准差为0.02
            module.weight.data.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        # ids 是输入的索引，targets 是目标的索引 
        batch, seq_len = idx.size()
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(seq_len, device=idx.device))
        # token_emb 为什么可以和 pos_emb 相加
        #token_emb 和 pos_emb 可以相加，是因为广播机制使形状对齐
        #且这种加法将语义和位置信息融合到了同一个向量空间中，是 Transformer 高效且有效的设计
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        # lm_head的输出是一个 (batch_size, seq_len, vocab_size) 的张量
        if targets is None:
            loss = None
        else:
            batch, seq_len, vocab_size = logits.shape
            logits = logits.view(batch * seq_len, vocab_size)
            targets = targets.view(batch * seq_len)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
       