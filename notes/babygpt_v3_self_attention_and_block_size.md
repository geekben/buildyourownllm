### babygpt_v3: 自注意力机制与 block_size （(B,T,C) 中的T）

#### v3 相比 v2 的核心变化

v3 在 v2 的基础上新增了 `Head` 类，实现了**自注意力机制（Self-Attention）**：

```python
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v
        return out
```

**核心逻辑**：每个 token 的输出不再只是它自己的 embedding，而是与序列中**所有之前 token** 的加权组合。

---

#### `block_size` 就是上下文窗口

`block_size` 在这个项目中承担了**三个角色**，都指向同一件事——**模型能"看到"的最大 token 数量**：

##### 1. 训练时：决定每个训练样本的长度

```python
# get_batch 中
x = data[i : i+block_size]       # 输入：连续 block_size 个 token
y = data[i+1 : i+block_size+1]   # 标签：往后错一位
```

训练时模型只在长度为 `block_size=8` 的片段上学习，所以它**从未见过超过 8 个 token 的上下文关系**。

##### 2. Attention 掩码：物理上限制了"能看多远"

```python
self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
```

这个下三角掩码的大小就是 `block_size × block_size`，Attention 的计算范围被**硬性限制**在 `block_size` 以内。

##### 3. 推理时：截断输入，只取最后 T 个 token

```python
def forward(self, idx, targets=None):
    B, T = idx.shape
    T = min(T, self.block_size)
    idx = idx[:, -T:]  # 只取最后 block_size 个 token
```

**如果输入 query 长度 > `block_size`，前面的 token 会被直接丢弃，只有最后 `block_size` 个 token 进入神经网络。**

---

#### 推理过程中的滑动窗口效果

在 `generate` 中，每生成一个新 token，`idx` 就会变长：

```python
idx = torch.cat((idx, idx_next), dim=1)  # 序列越来越长
```

但下一次 `forward` 时又会被截断：

```python
idx = idx[:, -T:]  # 只看最后 block_size 个
```

推理过程就像一个**滑动窗口**：

```
生成第 1 个 token：看 [春, 江]                    （长度 2 < 8，全部使用）
生成第 2 个 token：看 [春, 江, 花]                 （长度 3 < 8，全部使用）
...
生成第 7 个 token：看 [春, 江, 花, 月, 夜, 人, 来, 去]  （长度 8 = block_size，刚好）
生成第 8 个 token：看 [江, 花, 月, 夜, 人, 来, 去, 了]  （长度 9 > 8，"春"被丢弃！）
生成第 9 个 token：看 [花, 月, 夜, 人, 来, 去, 了, 也]  （"春""江"都丢了）
```

**模型的"记忆"只有 `block_size` 这么长，更早的内容完全看不到。**

---

#### `block_size` 的影响总结

| 方面 | 影响 |
|------|------|
| **上下文能力** | 越大，模型能记住更长的依赖关系 |
| **显存占用** | 注意力矩阵是 `T×T`，显存随 `block_size²` 增长 |
| **计算量** | 自注意力计算复杂度 O(T²) |
| **位置编码** | `position_embedding_table` 大小为 `block_size × n_embed` |

这就是为什么 v11 把 `block_size` 从 8 提升到 256：

| 版本 | `block_size` | 含义 |
|---|---|---|
| v1 ~ v10 | **8** | 只能看 8 个字的上下文，基本只能学到局部搭配 |
| v11 ~ v12 | **256** | 能看 256 个字，可以学到一整首词的结构和韵律 |

---

#### 为什么现代 LLM 的上下文窗口是重要指标

现实中 LLM 的上下文窗口（4K、8K、128K）是重要的技术指标——**更长的窗口 = 更好的理解能力，但也意味着更高的计算成本**。

解决长上下文的常见技术：
- **KV Cache**：缓存已计算的 key/value，避免重复计算（本项目有示例）
- **Flash Attention**：优化注意力计算的显存访问模式
- **Sliding Window Attention**：只关注局部窗口内的 token
- **ALiBi / RoPE**：相对位置编码，允许外推到更长序列
