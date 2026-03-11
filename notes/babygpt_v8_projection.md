# BabyGPT v8: Projection Layer 投影层

## 概述

v8 在 v7 的基础上引入了两个重要改进：
1. **Multi-Head Attention 投影层**：在多头拼接后添加线性投影
2. **FFN 扩展-收缩结构**：采用标准的 4 倍中间层设计

## v7 vs v8 核心差异对比

### 1. MultiHeadAttention 类的变化

**v7：**
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)  # 直接拼接
```

**v8：**
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)  # 新增：投影层

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)  # 拼接后再投影
```

### 2. FeedForward 类的变化

**v7：**
```python
class FeedFoward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.ReLU(),
        )
```

**v8：**
```python
class FeedFoward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4),  # 扩展 4 倍
            nn.ReLU(),
            nn.Linear(n_embed * 4, n_embed),  # 收缩回来
        )
```

### 3. 架构对比

| 方面 | v7 | v8 |
|------|-----|-----|
| Multi-Head 输出 | 直接拼接 | 拼接 + 线性投影 |
| FFN 中间层 | n_embed | 4 × n_embed |
| 参数量 | 较少 | 增加约 5 倍（FFN）+ 投影层 |

## 为什么需要投影层？

### 问题：简单拼接的局限

v7 中，多头注意力的输出是简单拼接：

```
Head1 (B, T, 8) ─┐
Head2 (B, T, 8) ─┼─→ concat → (B, T, 32)
Head3 (B, T, 8) ─┤
Head4 (B, T, 8) ─┘
```

**问题：**
- 各头独立计算，缺乏"交流"
- 拼接只是机械组合，没有学习如何融合
- 不同头可能学到冗余信息

### 解决方案：线性投影

v8 在拼接后添加可学习的投影层：

```
concat (B, T, 32) → Linear(32, 32) → output (B, T, 32)
```

投影层让模型学习**如何组合各头的信息**：

- 可以强调重要的头
- 可以抑制冗余的头
- 可以学习头与头之间的交互

### 类比理解

| 方式 | 类比 |
|------|------|
| v7（拼接） | 4 个人各自写报告，直接钉在一起 |
| v8（投影） | 4 个人写报告后，由主编整合、提炼、重写 |

投影层就是这个"主编"。

## 为什么 FFN 需要 4 倍扩展？

### 标准 Transformer FFN 结构

```
输入 (B, T, 32)
    ↓
Linear(32, 128)  # 扩展 4 倍
    ↓
ReLU()
    ↓
Linear(128, 32)  # 收缩回来
    ↓
输出 (B, T, 32)
```

### 扩展的意义

| 阶段 | 维度变化 | 作用 |
|------|---------|------|
| 扩展 | 32 → 128 | 增加表示容量，在高维空间学习复杂特征 |
| 激活 | 128 → 128 | 引入非线性 |
| 收缩 | 128 → 32 | 压缩回原始维度，提取精华 |

### 类比理解

| 类比 | 说明 |
|------|------|
| 放大镜 | 先放大观察细节（扩展），再总结规律（收缩） |
| 头脑风暴 | 先发散思维产生 4 倍想法（扩展），再筛选精华（收缩） |
| 神经科学 | 类似大脑皮层的"稀疏编码"机制 |

### 为什么是 4 倍？

这是经验值，来自原始 Transformer 论文：

| 模型 | FFN 扩展倍数 |
|------|-------------|
| Transformer (2017) | 4× |
| GPT-2 | 4× |
| GPT-3 | 4× |
| LLaMA | 8/3× (约 2.67×) |

4 倍是平衡**表达能力**和**计算效率**的经典选择。

## 参数量对比

以 `n_embed = 32` 为例：

### MultiHeadAttention

| 组件 | v7 参数 | v8 参数 |
|------|---------|---------|
| 4 个 Head 的 K, Q, V | 4 × 3 × 32 × 8 = 3072 | 3072 |
| **Projection** | - | **32 × 32 = 1024** |
| **总计** | 3072 | **4096** |

### FeedForward

| 组件 | v7 参数 | v8 参数 |
|------|---------|---------|
| Linear 1 | 32 × 32 = 1024 | 32 × 128 = 4096 |
| Linear 2 | - | 128 × 32 = 4096 |
| **总计** | 1024 | **8192** |

### 单个 Block 总参数

| 组件 | v7 | v8 |
|------|-----|-----|
| MultiHeadAttention | 3072 | 4096 |
| FeedForward | 1024 | 8192 |
| **总计** | **4096** | **12288** |

v8 的单个 Block 参数量约为 v7 的 3 倍。

## 数据流对比

### v7 MultiHeadAttention

```
x (B, T, 32)
    ↓
┌───┴───┐
│ Head1 │→ (B, T, 8)
│ Head2 │→ (B, T, 8)
│ Head3 │→ (B, T, 8)
│ Head4 │→ (B, T, 8)
└───┬───┘
    ↓
concat → (B, T, 32)  ← 直接输出
```

### v8 MultiHeadAttention

```
x (B, T, 32)
    ↓
┌───┴───┐
│ Head1 │→ (B, T, 8)
│ Head2 │→ (B, T, 8)
│ Head3 │→ (B, T, 8)
│ Head4 │→ (B, T, 8)
└───┬───┘
    ↓
concat → (B, T, 32)
    ↓
Linear(32, 32) → (B, T, 32)  ← 投影后输出
```

### v8 FeedForward

```
x (B, T, 32)
    ↓
Linear(32, 128) → (B, T, 128)  # 扩展
    ↓
ReLU → (B, T, 128)
    ↓
Linear(128, 32) → (B, T, 32)   # 收缩
```

## 下一步

v8 完善了 Multi-Head Attention 和 FFN 的结构，但训练深层网络还需要**归一化**来稳定训练。

v9 将引入 **Layer Normalization**：

```python
# v9 Block（预告）
x = x + self.sa(self.ln1(x))   # LayerNorm + Attention + Residual
x = x + self.ffwd(self.ln2(x)) # LayerNorm + FFN + Residual
```

LayerNorm 确保每层的输入分布稳定，加速收敛。

## 总结

| 版本 | 核心变化 |
|------|----------|
| v7 | 残差连接，解决深层网络梯度消失 |
| v8 | 投影层 + FFN 扩展结构，增强表达能力 |

v8 的两个改进都是"标准配置"：

- **投影层**：让多头注意力学会如何融合信息
- **4 倍 FFN**：在更高维空间学习复杂特征

这些设计让 BabyGPT 更接近真正的 Transformer 架构。
