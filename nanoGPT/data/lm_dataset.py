import json

from torch.utils.data import Dataset # PyTorch 核心库，实现数据集和数据加载；
import torch


class GPTDataset(Dataset):
    def __init__(self, path, block_size: int):
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = block_size

        self.encoded_data = []

        self.eos_token = self.enc.encode(
            "<|endoftext|>",
            allowed_special={"<|endoftext|>"}
        )[0]
        raw_data = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i >=1000:
                    break
                try:
                    text = json.loads(line.strip())["text"]
                    raw_data.append(text)
                except Exception as e:
                    continue
               
        full_encoded = []
        for text in raw_data:
            encoded_text = self.enc.encode(text)
            full_encoded.extend(encoded_text + [self.eos_token])
        
    def __len__(self):
        return len(self.encoded_data)
    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
                          
