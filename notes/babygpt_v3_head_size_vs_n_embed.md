### head_size 与 n_embed 的关系

在阅读 `babygpt_v3_self_attention.py` 时，可能会困惑：

> BabyGPT 初始化时传入 `Head(n_embed)`，但 Head 类内部又是 `Linear(n_embed, head_size)`，这不重复了吗？

关键在于理解**单头 vs 多头注意力**的设计差异。

---

#### v3（单头）：`head_size == n_embed`

```python
# babygpt_v3_self_attention.py
n_embed = 32

# BabyGPT.__init__ 中
self.sa_head = Head(n_embed)  # head_size = 32
```

此时 `head_size = n_embed = 32`，所以 `Linear(n_embed, head_size)` 实际是 `Linear(32, 32)`。

---

#### v4（多头）：`n_embed = num_heads × head_size`

```python
# babygpt_v4_multihead_attention.py
n_embed = 32
n_head = 4  # 新增

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)

# BabyGPT.__init__ 中
self.sa_heads = MultiHeadAttention(n_head, n_embed//n_head)  # head_size = 32//4 = 8
```

每个 head 的输出维度是 8，4 个 head 拼接后：

```python
return torch.cat([h(x) for h in self.heads], dim=-1)  # 4 × 8 = 32 = n_embed
```

---

#### 核心关系

| 版本 | 公式 | 说明 |
|------|------|------|
| v3 单头 | `head_size = n_embed` | 一个大 head |
| v4+ 多头 | `n_embed = num_heads × head_size` | 多个小 head 拼接 |

---

#### 为什么 Linear 输入是 `n_embed` 而不是 `head_size`？

因为 **输入 `x` 的维度始终是 `n_embed`**：

```python
def forward(self, x):
    B, T, C = x.shape  # C = n_embed = 32（不变）
    k = self.key(x)    # Linear(n_embed, head_size): 输入 32 维，输出 head_size 维
```

每个 Head 都从完整的 `n_embed` 维输入中"提取"自己关注的那部分信息，输出 `head_size` 维。

- **单头时**：直接输出 `n_embed` 维，无需后续处理
- **多头时**：每个头输出 `head_size = n_embed / num_heads` 维，多个头拼接后还原为 `n_embed` 维

---

#### 维度变化示意图

**单头（v3）**：
```
x: (B, T, n_embed=32)
    ↓ Linear(32, 32) × 3 (key, query, value)
k/q/v: (B, T, 32)
    ↓ attention
out: (B, T, 32)  → 直接使用
```

**多头（v4）**：
```
x: (B, T, n_embed=32)
    ↓ 4 个 Head，每个 Linear(32, 8) × 3
k/q/v: (B, T, 8) × 4 heads
    ↓ attention × 4
out: (B, T, 8) × 4 heads
    ↓ concat
out: (B, T, 32)  → 拼接还原
```

---

#### 多头的意义

多头注意力让模型能从**不同子空间**学习不同的关注模式。例如在 NLP 中：
- 一个 head 可能关注**语法结构**
- 另一个 head 可能关注**语义关联**
- 还有一个 head 可能关注**指代关系**

最终拼接后，模型综合了多种视角的信息。
