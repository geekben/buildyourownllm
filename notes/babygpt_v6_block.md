# BabyGPT v6: Transformer Block 模块化封装

## 概述

v6 在 v5 的基础上引入了 **Block** 类，将 Multi-Head Attention 和 Feed-Forward Network 封装成一个可堆叠的模块，实现了 Transformer 的模块化设计。

## v5 vs v6 核心差异对比

### 1. 新增 Block 类

v6 新增了 `Block` 类，将 Attention 和 FFN 封装在一起：

```python
class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embed)

    def forward(self, x):
        x = self.sa(x)
        x = self.ffwd(x)
        return x
```

### 2. 新增 n_layer 参数

```python
n_layer = 3  # block的数量
```

### 3. BabyGPT 类的变化

**v5 `__init__`：**
```python
self.sa_heads = MultiHeadAttention(n_head, n_embed//n_head)
self.ffwd = FeedFoward(n_embed)
```

**v6 `__init__`：**
```python
self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
```

### 4. forward 函数的变化

**v5 forward：**
```python
x = tok_emb + pos_emb
x = self.sa_heads(x)  # 单层 attention
x = self.ffwd(x)       # 单层 ffwd
logits = self.lm_head(x)
```

**v6 forward：**
```python
x = tok_emb + pos_emb
x = self.blocks(x)     # 多层 block（每层包含 attention + ffwd）
logits = self.lm_head(x)
```

### 5. 架构对比

| 方面 | v5 | v6 |
|------|-----|-----|
| 层级结构 | 扁平：Attention → FFN | 模块化：Block × n_layer |
| 网络深度 | 1 层（1个 Attention + 1个 FFN） | n_layer 层（可配置） |
| 可扩展性 | 增加层数需要修改代码 | 只需调整 n_layer 参数 |

## 为什么需要 Block 封装？

### 问题：v5 的局限性

v5 中 Attention 和 FFN 是"扁平"串联的：

```
Embedding → Attention → FFN → LM Head
```

如果要增加层数，需要手动复制代码：

```python
# 假设要 3 层，v5 风格会很冗余
x = self.sa_heads_1(x)
x = self.ffwd_1(x)
x = self.sa_heads_2(x)
x = self.ffwd_2(x)
x = self.sa_heads_3(x)
x = self.ffwd_3(x)
```

### 解决方案：Block 封装

v6 将"Attention + FFN"封装成一个 Block：

```
Embedding → [Block] → [Block] → [Block] → LM Head
                ↑         ↑         ↑
              第1层     第2层     第3层
```

每个 Block 内部：

```
x → Multi-Head Attention → FFN → output
```

### 类比理解

可以把 Block 类比为：

| 类比 | 说明 |
|------|------|
| 乐高积木 | 每个 Block 是一块积木，可以堆叠任意数量 |
| 神经网络层 | 类似 CNN 中的 Conv+ReLU+Pool 组合 |
| 函数封装 | 将重复逻辑封装成可复用的单元 |

## nn.Sequential 的妙用

```python
self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
```

这行代码做了什么？

1. `[Block(...) for _ in range(n_layer)]` - 创建 n_layer 个 Block 实例的列表
2. `*` - 解包列表，将 `[b1, b2, b3]` 变成 `b1, b2, b3`
3. `nn.Sequential(...)` - 将这些 Block 串联成一个顺序容器

等价于：

```python
self.blocks = nn.Sequential(
    Block(n_embed, n_head=n_head),  # 第1层
    Block(n_embed, n_head=n_head),  # 第2层
    Block(n_embed, n_head=n_head),  # 第3层
)
```

## 深度的意义

### 单层 vs 多层

| 层数 | 模型能力 | 计算量 |
|------|----------|--------|
| 1 层 | 简单模式识别 | 低 |
| 3 层 | 中等复杂特征 | 中 |
| 12+ 层 | 复杂语义理解 | 高 |

### 类比：多轮会议

| 版本 | 类比 | 说明 |
|------|------|------|
| v5（1 层） | 团队只开了一次会 | 讨论完就做决定。对简单问题够用，但复杂问题想不透。 |
| v6（3 层） | 团队开了三轮会 | 第一轮收集基本信息，第二轮深入分析，第三轮综合决策。每一轮都是"先讨论（Attention）再独立思考（FFN）"，每一轮的理解都比上一轮更深。 |

### 为什么更深的网络更强？

每一层 Block 可以学习不同层次的抽象：

```
第1层: 学习局部特征（如：相邻词的关联）
第2层: 学习中层特征（如：短语结构）
第3层: 学习高层特征（如：语义关系）
```

这是深度学习的核心思想：**层次化特征提取**。

## 数据流对比

### v5 数据流

```
Input (B, T)
    ↓
Token Embedding (B, T, 32)
    ↓
Position Embedding (B, T, 32)
    ↓
Multi-Head Attention (B, T, 32)  ← 单层
    ↓
Feed-Forward (B, T, 32)          ← 单层
    ↓
LM Head (B, T, vocab_size)
```

### v6 数据流

```
Input (B, T)
    ↓
Token Embedding (B, T, 32)
    ↓
Position Embedding (B, T, 32)
    ↓
Block 1: Attention → FFN (B, T, 32)
    ↓
Block 2: Attention → FFN (B, T, 32)
    ↓
Block 3: Attention → FFN (B, T, 32)
    ↓
LM Head (B, T, vocab_size)
```

## 参数量对比

以 `n_embed=32, n_head=4, vocab_size=65, n_layer=3` 为例：

| 组件 | v5 参数量 | v6 参数量 |
|------|-----------|-----------|
| Token Embedding | 65 × 32 = 2080 | 2080 |
| Position Embedding | 8 × 32 = 256 | 256 |
| Block 1 (Attention + FFN) | 3072 + 1056 = 4128 | 4128 |
| Block 2 | - | 4128 |
| Block 3 | - | 4128 |
| LM Head | 65 × 32 = 2080 | 2080 |
| **总计** | **~8.5K** | **~17K** |

v6 通过堆叠 3 个 Block，参数量约为 v5 的 2 倍。

## 下一步

v6 实现了 Block 的模块化封装，但目前 Block 内部的连接是"直通"的：

```python
x = self.sa(x)
x = self.ffwd(x)
```

v7 将引入 **残差连接（Residual Connection）**，让梯度更容易流向深层：

```python
x = x + self.sa(x)   # 残差连接
x = x + self.ffwd(x) # 残差连接
```

## 总结

| 版本 | 核心变化 |
|------|----------|
| v5 | 前馈网络，引入非线性变换 |
| v6 | Block 封装，实现可堆叠的 Transformer 层 |

v6 的 Block 封装是 Transformer 架构的关键设计之一：

- **模块化**：将 Attention + FFN 封装成可复用单元
- **可配置深度**：通过 n_layer 参数控制网络层数
- **代码简洁**：用一行代码创建多层网络

这是从"玩具模型"向"真正可扩展的 Transformer"迈出的重要一步。
