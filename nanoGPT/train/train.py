import sys
import os
# 将项目根目录加入模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from model.model import GPTConfig, GPTModel
from data.lm_dataset import GPTDataset
from torch.utils.data import DataLoader, random_split

def main():
    # 模型初始化
    model = GPTModel(config=GPTConfig())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f} M")

    # 优化器和学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)  # 如果按 batch 更新，T_max 应为总步数
    config = GPTConfig()
    block_size = config.block_size

    # 创建数据集时传入 block_size
    # 数据集使用mobvoi_seq_monkey_general_open_corpus.jsonl， 这里只是空的。
    full_dataset = GPTDataset("data/train.jsonl", block_size)
    train_dataset, val_dataset = random_split(full_dataset, [0.9, 0.1])
    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False)

    # 训练和评估函数
    def train_one_epoch(model, optimizer, scheduler, loader, device):
        model.train()
        total_loss = 0.0
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()   # 每个 batch 后更新学习率

            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

        return total_loss / len(loader)

    def evaluate(model, loader, device):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                total_loss += loss.item()
        return total_loss / len(loader)

    # 训练循环
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, scheduler, train_loader, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}: train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}")

if __name__ == "__main__":
    main()