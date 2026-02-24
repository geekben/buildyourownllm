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

#### 比喻：图书馆检索系统

想象你在图书馆里找资料：

| 概念 | 比喻 | 作用 |
|------|------|------|
| **Query** | 你的"查询问题" | 每本书都在问："我想找什么相关的书？" |
| **Key** | 书的"标签/关键词" | 每本书都有一个标签，描述它是什么内容 |
| **Value** | 书的"实际内容" | 你真正想读的东西 |
| **Attention** | "匹配度打分" | 查询问题和标签越匹配，越关注那本书 |

**核心思想**：
- Query 和 Key 做点积 → 得到"匹配度分数"
- 分数越高，越关注那本书的 Value
- 最终输出是所有书内容的加权组合

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
k: (B, T, head_size)  → "我是什么内容"（标签）
q: (B, T, head_size)  → "我在找什么内容"（查询）
v: (B, T, head_size)  → "我的实际信息"（内容）
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
        ├──→ key(x)   → K: (B, 3, 32) "每本书的标签"
        ├──→ query(x) → Q: (B, 3, 32) "每本书在找什么"
        └──→ value(x) → V: (B, 3, 32) "每本书的内容"
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
