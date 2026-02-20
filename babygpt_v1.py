import torch
# 参见 notes/babygpt_v1_vs_simplebigrammodel.md 了解与 SimpleBigramModel 的对比分析
import torch.nn as nn
from torch.nn import functional as F
from typing import List
import time

torch.manual_seed(42)

prompts = ["春江", "往事"] # 推理的输入prompts
max_new_token = 100 # 推理生成的最大tokens数量

max_iters = 5000 # 训练的最大迭代次数
eval_iters = 100 # 评估的迭代次数
eval_interval = 200 # 评估的间隔
batch_size = 32 # 每个批次的大小
block_size = 8 # 每个序列的最大长度
learning_rate = 1e-2 # 学习率
n_embed = 32 # 嵌入层的维度
tain_data_ratio = 0.9 # 训练数据占数据集的比例，剩下的是验证数据

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

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

class BabyGPT(nn.Module):

    def __init__(self, vocab_size: int, n_embd: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # 嵌入层，把token映射到n_embd维空间
        self.lm_head = nn.Linear(n_embd, vocab_size) # 线性层，把n_embd维空间映射到vocab_size维空间，

    def forward(self, idx, targets=None):
        tok_emb = self.token_embedding_table(idx) # 获得token的嵌入表示 (B,T,n_embd)
        logits = self.lm_head(tok_emb) # 通过线性层，把embedding结果重新映射回vocab_size维空间 (B,T,vocab_size)

        if targets is None: # 推理场景，不需要计算损失值
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # 把(B,T,C)的形状转换为(B*T,C)，因为交叉熵损失函数第一个参数只接受二维输入。这个操作并没有丢失信息
            targets = targets.view(B*T) # 把(B,T)的形状转换为(B*T)，因为交叉熵损失函数第二个参数只接受一维输入。这个操作并没有丢失信息
            loss = F.cross_entropy(logits, targets) # 计算交叉熵损失
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx) # logits的形状是(B,T,vocab_size)，每一个token都计算了下一个token的概率
            logits = logits[:, -1, :] # 实际上我们只需要最后一个token算出来的值
            probs = F.softmax(logits, dim=-1) # 使用softmax函数算概率分布，这里dim=-1表示对最后一个维度进行softmax
            idx_next = torch.multinomial(probs, num_samples=1) # 根据概率分布随机采样，这里num_samples=1表示采样一个token
            idx = torch.cat((idx, idx_next), dim=1) # 把采样的token拼接到序列后面
        return idx

tokenizer = Tokenizer(text)
vocab_size = tokenizer.vocab_size

raw_data = torch.tensor(tokenizer.encode(text), dtype=torch.long).to(device)
n = int(tain_data_ratio*len(raw_data))
data = {'train': raw_data[:n], 'val': raw_data[n:]}

def get_batch(data, batch_size, block_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# 参见 notes/estimate_loss_and_cross_entropy.md 了解 estimate_loss 函数和交叉熵损失函数
@torch.no_grad()
def estimate_loss(model, data, batch_size, block_size, eval_iters):
    '''
    计算模型在训练集和验证集上的损失
    '''
    out = {}
    model.eval() # 切换到评估模式
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data[split], batch_size, block_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # 切换回训练模式
    return out

model = BabyGPT(vocab_size, n_embed).to(device)

# 训练
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

start_time = time.time()
tokens_processed = 0

for iter in range(max_iters):
    x, y = get_batch(data['train'], batch_size, block_size)
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    tokens_processed += batch_size * block_size

    if iter % eval_interval == 0:
        elapsed = time.time() - start_time
        tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
        losses = estimate_loss(model, data, batch_size, block_size, eval_iters)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, speed: {tokens_per_sec:.2f} tokens/sec")

# 推理
prompt_tokens = torch.stack([torch.tensor(tokenizer.encode(p)).to(device) for p in prompts])

# 生成
result = model.generate(prompt_tokens, max_new_token)

# 解码并打印结果
for tokens in result:
    print(tokenizer.decode(tokens.tolist()))
    print('-'*10)
