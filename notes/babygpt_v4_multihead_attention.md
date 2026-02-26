# BabyGPT v4: Multi-Head Attention 多头注意力

## 概述

v4 在 v3 的基础上引入了 **Multi-Head Attention（多头注意力）**，这是 Transformer 架构的核心创新之一。

## v3 vs v4 核心差异对比

### 1. 新增超参数

```python
# v4 新增
n_head = 4  # 多头注意力的头数
```

### 2. 注意力模块变化

| 方面 | v3 (单头) | v4 (多头) |
|------|-----------|-----------|
| 类名 | `Head` | `MultiHeadAttention` |
| 头数 | 1 | 4 (`n_head`) |
| 每个头的大小 | `n_embed` (32) | `n_embed // n_head` (8) |
| 总计算量 | 相同 | 相同 |
| 输出维度 | `head_size` (32) | `head_size * n_head` (32) |

### 3. 代码变化

**v3 BabyGPT 初始化：**
```python
self.sa_head = Head(n_embed)  # 单头，head_size = n_embed = 32
```

**v4 BabyGPT 初始化：**
```python
self.sa_heads = MultiHeadAttention(n_head, n_embed // n_head)  # 4个头，每个head_size = 8
```

### 4. 新增 MultiHeadAttention 类

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)
```

工作流程：
1. 创建 `num_heads` 个独立的 `Head` 实例
2. 每个 Head 独立计算注意力，输出维度为 `head_size`
3. 将所有头的输出在最后一个维度上拼接（concatenate）

## 为什么需要多头注意力？

### 单头的局限性

单头注意力只能学习一种"关联模式"。例如，它可能只学会关注"语法结构"，而忽略"语义关联"。

### 多头的优势

多头注意力允许模型**同时关注不同类型的关联**：

```
Head 1: 关注语法结构（如主谓关系）
Head 2: 关注语义关联（如近义词）
Head 3: 关注位置信息（如相邻词）
Head 4: 关注长距离依赖（如句首句尾呼应）
```

每个头独立学习，最后拼接融合所有信息。

### 类比理解

可以把多头注意力想象成"多个专家会诊"：
- 每个专家（Head）有自己的专长视角
- 每个专家独立分析同一个问题
- 最后综合所有专家的意见得出结论

## 维度计算详解

假设 `n_embed = 32`, `n_head = 4`:

```
输入 x: (B, T, 32)
    ↓
┌─────────────────────────────────────┐
│  MultiHeadAttention(4, 8)           │
│                                     │
│  Head 0: (B, T, 32) → (B, T, 8)    │
│  Head 1: (B, T, 32) → (B, T, 8)    │
│  Head 2: (B, T, 32) → (B, T, 8)    │
│  Head 3: (B, T, 32) → (B, T, 8)    │
│                                     │
│  concat: (B, T, 8*4) = (B, T, 32)  │
└─────────────────────────────────────┘
    ↓
输出: (B, T, 32)
```

**关键洞察**：虽然头数变了，但输出维度保持不变（都是 32），这保证了后续层不需要修改。

## 参数量对比

### v3 单头
```
key:   Linear(32, 32) = 32 * 32 = 1024 参数
query: Linear(32, 32) = 32 * 32 = 1024 参数
value: Linear(32, 32) = 32 * 32 = 1024 参数
总计: 3072 参数
```

### v4 多头
```
每个头:
  key:   Linear(32, 8) = 32 * 8 = 256 参数
  query: Linear(32, 8) = 32 * 8 = 256 参数
  value: Linear(32, 8) = 32 * 8 = 256 参数
  每头小计: 768 参数

4个头总计: 768 * 4 = 3072 参数
```

**结论**：参数量相同！多头只是在结构上重新分配了参数，让不同头学习不同模式。

## 计算复杂度

| 操作 | 时间复杂度 |
|------|-----------|
| 单头注意力 | O(T² × d) |
| 多头注意力 | O(T² × d/n × n) = O(T² × d) |

**结论**：计算复杂度相同，多头不会增加计算量。

## 实际效果

多头注意力通常能获得更好的性能，因为：

1. **表达能力更强**：不同头可以捕获不同类型的关联
2. **泛化能力更好**：多头相当于一种隐式的 ensemble
3. **训练更稳定**：多个头的梯度更平滑

## 下一步

v4 的多头注意力已经引入，但模型还缺少一个重要组件：**Feed-Forward Network**。v5 将引入 FFN，为模型增加非线性变换能力。

## 总结

| 版本 | 核心变化 |
|------|----------|
| v3 | 引入 Self-Attention，token 之间可以交流 |
| v4 | 单头 → 多头，同时学习多种关联模式 |

多头注意力是 Transformer 的标志性创新之一，它让模型能够同时从多个角度理解 token 之间的关系。
