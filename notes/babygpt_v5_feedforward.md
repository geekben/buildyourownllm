# BabyGPT v5: Feed-Forward Network 前馈网络

## 概述

v5 在 v4 的基础上引入了 **Feed-Forward Network（前馈网络）**，这是 Transformer 架构中与注意力机制同等重要的组件。

## v4 vs v5 核心差异对比

### 1. 新增 FeedForward 类

```python
class FeedFoward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)
```

### 2. BabyGPT 类的变化

**v4 forward 函数：**
```python
x = tok_emb + pos_emb
x = self.sa_heads(x)  # 只有多头注意力
logits = self.lm_head(x)
```

**v5 forward 函数：**
```python
x = tok_emb + pos_emb
x = self.sa_heads(x)  # 多头注意力
x = self.ffwd(x)       # 新增：前馈网络
logits = self.lm_head(x)
```

### 3. 架构对比

| 方面 | v4 | v5 |
|------|-----|-----|
| 组件数量 | 1 (Attention) | 2 (Attention + FFN) |
| 非线性变换 | 无（注意力是线性的） | 有（ReLU 激活函数） |
| 参数量 | 较少 | 新增 n_embed² 参数 |

## 为什么需要 Feed-Forward Network？

### 注意力机制的局限性

Self-Attention 本质上是**加权求和**操作：

```python
out = wei @ v  # wei 是注意力权重，v 是 value
```

这可以理解为：输出是输入的线性组合。即使有多个头，每个头的输出仍然是输入的线性变换。

**问题**：线性组合无法表达复杂的非线性关系。

### 前馈网络的作用

FFN 引入**非线性激活函数**（ReLU），让模型能够学习更复杂的映射：

```
输入 x (B, T, 32)
    ↓
Linear(32, 32)  # 线性变换
    ↓
ReLU()          # 非线性激活：把负值变成 0
    ↓
输出 (B, T, 32)
```

### 类比理解

可以把 Transformer 的结构类比为：

| 组件 | 类比 | 作用 |
|------|------|------|
| Attention | "信息收集员" | 收集上下文信息，了解"谁与谁相关" |
| FFN | "信息加工厂" | 对收集到的信息进行深度加工，提取复杂特征 |

**Attention 负责"通信"，FFN 负责"思考"。**

## ReLU 激活函数

```python
nn.ReLU()  # 把负值变为 0，正值不变
```

数学定义：
```
ReLU(x) = max(0, x)
```

示例：
```
输入: [-2, -1, 0, 1, 2]
输出: [ 0,  0, 0, 1, 2]
```

**为什么需要非线性？**

如果没有非线性激活函数，无论叠加多少层线性变换，最终都可以合并为一个线性变换：

```
W2(W1x) = (W2W1)x = Wx
```

非线性激活函数打破了这个限制，让神经网络能够逼近任意复杂函数。

## 维度分析

以 `n_embed = 32` 为例：

```
输入: (B, T, 32)
     ↓
Linear(32, 32): 32 * 32 + 32 = 1056 参数（含 bias）
     ↓
中间结果: (B, T, 32)
     ↓
ReLU: 无参数
     ↓
输出: (B, T, 32)
```

维度保持不变，确保可以串联多层。

## 完整数据流

```
Token Embedding (B, T, 32)
        ↓
Position Embedding (B, T, 32)
        ↓
Multi-Head Attention
  - 4 个头并行计算
  - 每个 head_size = 8
  - 输出拼接为 (B, T, 32)
        ↓
Feed-Forward Network
  - Linear(32, 32)  # *提供了新的线性变换矩阵，是可学习的参数来源*
  - ReLU            # *引入非线性，避免变成所有线性层坍缩成一个线性变换*
  - 输出 (B, T, 32)
        ↓
Linear Head (B, T, vocab_size)
```

## 标准 Transformer 的 FFN

在实际的 Transformer（如 GPT-2/3）中，FFN 通常有一个"扩展-收缩"的结构：

```python
# 标准实现（v5 中尚未使用）
self.net = nn.Sequential(
    nn.Linear(n_embed, 4 * n_embed),  # 扩展 4 倍
    nn.ReLU(),
    nn.Linear(4 * n_embed, n_embed),  # 收缩回来
)
```

这种结构让 FFN 有更大的"思考空间"。v5 使用简化版本，后续版本会完善。

## 参数量对比

| 组件 | v4 参数 | v5 新增参数 |
|------|---------|-------------|
| Token Embedding | vocab_size × 32 | - |
| Position Embedding | 8 × 32 | - |
| Multi-Head Attention | 3072 | - |
| **Feed-Forward** | - | **1056** |
| LM Head | vocab_size × 32 | - |

FFN 为模型增加了约 1000 个可学习参数，提升了模型的表达能力。

## 下一步

v5 引入了 FFN，但目前 Attention 和 FFN 是扁平串联的。v6 将把它们封装成 **Transformer Block**，形成可堆叠的模块化结构。

## 总结

| 版本 | 核心变化 |
|------|----------|
| v4 | 多头注意力，同时学习多种关联模式 |
| v5 | 前馈网络，引入非线性变换能力 |

FFN 和 Attention 是 Transformer 的两大支柱：
- **Attention**：让 token 之间交换信息
- **FFN**：对信息进行非线性加工

两者缺一不可，共同构成了 Transformer 的核心计算单元。
