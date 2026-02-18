import torch
import torch.nn as nn
import random
from typing import List

random.seed(42)
torch.manual_seed(42)

prompts = ["春江", "往事"]
max_new_token = 100
# 改为对齐python版本，实验对比耗时：notes/experiment_simplebigrammodel_python_vs_torch.md
max_iters = 8000
batch_size = 32
block_size = 8

# 改为cuda运行报错，耗时3m多
device = 'cpu' #'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

with open('ci.txt', 'r', encoding='utf-8') as f:
    text = f.read()

class Tokenizer:
    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s]

    def decode(self, l: List[int]) -> str:
        return ''.join([self.itos[i] for i in l])

class BigramLanguageModel():
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        # 参见 notes/pytorch_vs_python_list.md 了解 PyTorch 张量 vs Python 列表的优化细节
        self.transition = torch.zeros((vocab_size, vocab_size), device=device)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx shape: (B, T)
        B, T = idx.shape
        result = torch.zeros((B, T, self.vocab_size), device=device)
        for b in range(B):
            for t in range(T):
                result[b][t] = self.transition[idx[b][t]]
        return result  # shape: (B, T, vocab_size)

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            # 获取最后一个token的预测
            logits = self(idx)[:, -1, :]  # (B, vocab_size)
            # 参见 notes/torch_clamp_multinomial.md 了解 torch.clamp 和 torch.multinomial 的用法
            # 将计数转换为概率
            probs = logits / torch.clamp(logits.sum(dim=-1, keepdim=True), min=1.0)
            # 采样下一个token
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # 拼接到序列中
            idx = torch.cat([idx, next_token], dim=1)  # (B, T+1)
        return idx

def get_batch(tokens: torch.Tensor, batch_size: int, block_size: int):
    ix = torch.randint(len(tokens) - block_size, (batch_size,), device=device)
    x = torch.stack([tokens[i:i+block_size] for i in ix])
    y = torch.stack([tokens[i+1:i+block_size+1] for i in ix])
    return x, y

tokenizer = Tokenizer(text)
vocab_size = tokenizer.vocab_size

# 将文本编码为tensor
tokens = torch.tensor(tokenizer.encode(text)).to(device)

model = BigramLanguageModel(vocab_size)

# 优化后的训练过程
for iter in range(max_iters):
    x_batch, y_batch = get_batch(tokens, batch_size, block_size)
    for i in range(batch_size):
        for j in range(block_size):
            x = x_batch[i, j]
            y = y_batch[i, j]
            model.transition[x, y] += 1

# 将prompts转换为tensor并处理
prompt_tokens = torch.stack([torch.tensor(tokenizer.encode(p)) for p in prompts])

# 生成
result = model.generate(prompt_tokens, max_new_token)

# 解码并打印结果
for tokens in result:
    print(tokenizer.decode(tokens.tolist()))
    print('-'*10)
