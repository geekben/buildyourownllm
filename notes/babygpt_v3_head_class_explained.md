### Head 类详解：自注意力机制的实现

#### 完整代码

```python
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape  # (batch_size, block_size, n_embed)
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v
        return out
```

---

#### 核心概念：每个 token 的双重角色

自注意力的关键：**每个 token 同时扮演两个角色**——既是提问者，也是被提问者。

```
Token A:
  - Query: "我想关注什么类型的 token？"（发出关注请求）
  - Key:   "我是什么类型的 token"（被别人识别）
  - Value: "我的实际内容"（被别人读取）
```

**Query @ Key.T 的含义**：>"我的关注偏好" 和 "你的身份标签" 匹配度有多高？

---

#### 比喻：社交网络

| 概念 | 比喻 | 说明 |
|------|------|------|
| **Query** | "交友偏好" | 每个人发出的——"我想认识什么样的人" |
| **Key** | "个人标签" | 每个人的——"我是什么样的人" |
| **Value** | "知识/资源" | 每个人的——"我能提供什么" |
| **Attention** | "匹配度" | 你的偏好和我的标签越匹配，我们越应该交流 |

**流程**：
1. A 的 Query 说："我想认识懂机器学习的人"
2. B 的 Key 说："我是机器学习专家"
3. 匹配度高 → A 会关注 B 的 Value（吸收 B 的知识）

**核心思想**：
- Query 和 Key 做点积 → 得到"匹配度分数"
- 分数越高，越关注对方的 Value
- 最终输出是所有 token 信息的加权组合

---

#### 逐行解析

##### 1. 三个线性层：Key、Query、Value

```python
self.key = nn.Linear(n_embed, head_size, bias=False)
self.query = nn.Linear(n_embed, head_size, bias=False)
self.value = nn.Linear(n_embed, head_size, bias=False)
```

每个 token 的 embedding（32 维）通过三个不同的线性变换，得到三种表示：

```
输入 x: (B, T, 32)
    ↓
k: (B, T, head_size)  → "我的身份标签"（被别人识别）
q: (B, T, head_size)  → "我的关注偏好"（发出关注请求）
v: (B, T, head_size)  → "我的实际内容"（被别人读取）
```

**为什么 `bias=False`？**  
在注意力机制中，bias 是可选的。去掉 bias 可以减少参数量，实践中效果相当。

---

##### 2. `register_buffer`：注册缓冲区

```python
self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
```

**`register_buffer` 的作用**：
- 将张量注册为模块的"缓冲区"，但**不是模型参数**
- 会随模型一起移动到 GPU/CPU（`model.to(device)` 时自动处理）
- 会随模型一起保存/加载（`torch.save` / `torch.load`）
- 但**不会被优化器更新**（不参与梯度下降）

**对比**：
| 方式 | 是否参与训练 | 是否随模型移动 | 典型用途 |
|------|-------------|---------------|---------|
| `nn.Parameter()` | ✅ 会更新 | ✅ 会移动 | 可学习参数 |
| `register_buffer()` | ❌ 不更新 | ✅ 会移动 | 掩码、常量 |
| 普通属性 `self.x = tensor` | ❌ 不更新 | ❌ 不移动 | 临时变量 |

**`torch.tril`**：下三角矩阵

```python
torch.tril(torch.ones(4, 4))
# tensor([[1., 0., 0., 0.],
#         [1., 1., 0., 0.],
#         [1., 1., 1., 0.],
#         [1., 1., 1., 1.]])
```

这是因果掩码（Causal Mask）：**每个 token 只能看到自己和之前的 token**，不能"偷看"未来。

---

##### 3. 计算注意力分数

```python
wei = q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5)
```

**`transpose(-2, -1)`**：交换最后两个维度

```python
# q: (B, T, head_size)
# k: (B, T, head_size)
# k.transpose(-2, -1): (B, head_size, T)
# q @ k.transpose(-2, -1): (B, T, T)
```

结果是一个 `T×T` 的矩阵，表示每个 token 对其他 token 的关注程度。

**为什么要除以 `sqrt(head_size)`？**
- 防止点积值过大
- 点积值过大会导致 softmax 输出接近 one-hot（梯度消失）
- 这是论文 "Attention Is All You Need" 的标准做法

---

##### 4. `masked_fill`：掩码填充

```python
wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
```

**`masked_fill(mask, value)`**：将 mask 为 True 的位置填充为指定值

```python
# 假设 T=3
# tril[:3, :3]:
# [[1, 0, 0],
#  [1, 1, 0],
#  [1, 1, 1]]

# wei 原始值（假设）:
# [[0.5, 0.3, 0.2],
#  [0.4, 0.4, 0.2],
#  [0.3, 0.3, 0.4]]

# masked_fill 后:
# [[0.5, -inf, -inf],
#  [0.4, 0.4, -inf],
#  [0.3, 0.3, 0.4]]
```

**为什么用 `-inf`？**  
因为 `softmax(-inf) = 0`，这样上三角位置（未来 token）的注意力权重就会变成 0。

---

##### 5. `softmax`：归一化为概率

```python
wei = F.softmax(wei, dim=-1)
```

**`softmax(dim=-1)`**：对最后一个维度做归一化

```python
# 输入:
# [[0.5, -inf, -inf],
#  [0.4, 0.4, -inf],
#  [0.3, 0.3, 0.4]]

# 输出（每行和为 1）:
# [[1.0, 0.0, 0.0],
#  [0.5, 0.5, 0.0],
#  [0.33, 0.33, 0.34]]
```

现在每行都是一个概率分布，表示当前 token 对各历史 token 的关注权重。

---

##### 6. 加权求和

```python
out = wei @ v  # (B, T, T) @ (B, T, head_size) = (B, T, head_size)
```

每个 token 的输出 = 所有历史 token 的 Value 的加权和。

---

#### 完整流程图

```
输入 x: (B, T=3, n_embed=32)
        │
        ├──→ key(x)   → K: (B, 3, 32) "身份标签"
        ├──→ query(x) → Q: (B, 3, 32) "关注偏好"
        └──→ value(x) → V: (B, 3, 32) "实际内容"
                    │
                    ▼
        Q @ K.T / sqrt(d)  →  注意力分数 (B, 3, 3)
                    │
                    ▼
        masked_fill (下三角掩码)
                    │
                    ▼
        softmax  →  注意力权重 (B, 3, 3)
                    │
                    ▼
        权重 @ V  →  输出 (B, 3, 32)
```

---

#### 关键函数速查

| 函数 | 用法 | 示例 |
|------|------|------|
| `register_buffer(name, tensor)` | 注册不参与训练的张量 | `self.register_buffer('mask', torch.ones(10))` |
| `torch.tril(input)` | 取下三角矩阵 | `torch.tril(torch.ones(4,4))` |
| `tensor.transpose(dim1, dim2)` | 交换两个维度 | `x.transpose(-2, -1)` |
| `masked_fill(mask, value)` | 掩码填充 | `x.masked_fill(mask==0, float('-inf'))` |
| `F.softmax(x, dim)` | softmax 归一化 | `F.softmax(logits, dim=-1)` |

---

#### 为什么叫"自"注意力？

因为 **Key、Query、Value 都来自同一个输入 x**：

```
同一个 x ──┬──→ Key
           ├──→ Query
           └──→ Value
```

每个 token 既是"提问者"（Query），也是"被关注对象"（Key/Value），让序列内部的 token 相互关注、交换信息。
