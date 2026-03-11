# BabyGPT v9: Layer Normalization 层归一化

## 概述

v9 在 v8 的基础上引入了 **Layer Normalization（层归一化）**，这是 Transformer 中稳定训练的关键技术。归一化确保每层输入的分布稳定，加速收敛并提升训练稳定性。

## v8 vs v9 核心差异对比

### 1. Block 类的变化

**v8 Block：**
```python
class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embed)

    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x
```

**v9 Block：**
```python
class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)  # 新增：Attention 前的 LayerNorm
        self.ln2 = nn.LayerNorm(n_embed)  # 新增：FFN 前的 LayerNorm

    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # Pre-Norm：先归一化，再做 Attention
        x = x + self.ffwd(self.ln2(x)) # Pre-Norm：先归一化，再做 FFN
        return x
```

### 2. BabyGPT 类的变化

**v8 BabyGPT `__init__`：**
```python
self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
self.lm_head = nn.Linear(n_embd, vocab_size)
```

**v9 BabyGPT `__init__`：**
```python
self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
self.ln_final = nn.LayerNorm(n_embed)  # 新增：最终归一化
self.lm_head = nn.Linear(n_embd, vocab_size)
```

### 3. forward 函数的变化

**v8 forward：**
```python
x = self.blocks(x)
logits = self.lm_head(x)
```

**v9 forward：**
```python
x = self.blocks(x)
x = self.ln_final(x)  # 新增：最终归一化
logits = self.lm_head(x)
```

### 4. 架构对比

| 方面 | v8 | v9 |
|------|-----|-----|
| Block 内部结构 | 残差连接 | 残差连接 + Pre-Norm |
| 归一化位置 | 无 | Attention 前、FFN 前、输出前 |
| 训练稳定性 | 一般 | 更稳定 |

## 什么是 Layer Normalization？

### 定义

LayerNorm 对每个样本的所有特征维度进行归一化：

```
对于输入 x = [x1, x2, ..., xd]
mean = (x1 + x2 + ... + xd) / d
var = ((x1-mean)² + (x2-mean)² + ... + (xd-mean)²) / d
output = [(x1-mean)/√var, (x2-mean)/√var, ..., (xd-mean)/√var] * γ + β
```

其中 γ 和 β 是可学习参数。

### PyTorch 实现

```python
nn.LayerNorm(n_embed)
```

对于输入 `(B, T, n_embed)`，LayerNorm 对最后一个维度（n_embed）进行归一化。

### 与 BatchNorm 的区别

| 特性 | BatchNorm | LayerNorm |
|------|-----------|-----------|
| 归一化维度 | 跨 batch | 跨特征维度 |
| 依赖 batch size | 是 | 否 |
| 适用场景 | CNN | Transformer、RNN |
| 训练/推理行为 | 不同（需要 running stats） | 相同 |

**图示：**

```
输入形状：(B, T, C) = (2, 3, 4)

BatchNorm：对每个特征通道，跨 batch 归一化
  - 所有样本的第1个特征一起归一化
  - 所有样本的第2个特征一起归一化
  - ...

LayerNorm：对每个样本的所有特征归一化
  - 样本1的所有特征一起归一化
  - 样本2的所有特征一起归一化
  - ...
```

## 为什么 Transformer 需要 LayerNorm？

### 问题：内部协变量偏移

深层网络中，每层的输入分布会随着训练不断变化：

```
Layer1 输出分布 → Layer2 输入分布变化 → 梯度不稳定 → 训练困难
```

这导致：
- 学习率需要设置得很小
- 训练不稳定，容易震荡
- 初始化敏感

### 解决方案：归一化

LayerNorm 将每层的输入"标准化"到稳定的分布：

```
输入 → LayerNorm → 均值≈0, 方差≈1 → 稳定的输入分布
```

### 类比理解：考试成绩标准化

想象 3 轮考试（3 层 Block），每轮的分数标准不同：

| 版本 | 类比 | 问题 |
|------|------|------|
| **v8（无 Norm）** | 第一轮满分 100，第二轮满分 1000，第三轮满分 10 | 直接把三轮分数加起来，第二轮的分数完全主导了总分，第三轮几乎没有影响 |
| **v9（有 Norm）** | 每轮考完后，先把分数标准化（转换为 Z-score），再传给下一轮 | 每一轮的贡献都是公平的，不会因为量纲不同而失衡 |

**更精确的理解**：

```
v8：第1层输出 → 第2层接收（但"单位"已变）→ 第2层难以学习
v9：第1层输出 → LayerNorm（统一"单位"）→ 第2层在"同一起跑线"上学习
```

本质是**稳定每层的输入分布**，让每层都在相同的数值范围内工作。

**其他类比**：

| 类比 | 说明 |
|------|------|
| 音量调节 | 把每层输出的"音量"调到统一水平 |
| 营养配餐 | 不管之前吃了什么，每顿饭都保证营养均衡 |

## Pre-Norm vs Post-Norm

### 两种结构

**Post-Norm（原始 Transformer）：**
```python
x = self.ln1(x + self.sa(x))
x = self.ln2(x + self.ffwd(x))
```

**Pre-Norm（GPT-2/3 使用，v9 采用）：**
```python
x = x + self.sa(self.ln1(x))
x = x + self.ffwd(self.ln2(x))
```

### 对比

| 方面 | Post-Norm | Pre-Norm |
|------|-----------|----------|
| 归一化位置 | 残差之后 | 残差之前 |
| 梯度流动 | 需要通过 LayerNorm | 有"干净"的残差通道 |
| 训练稳定性 | 较差 | 较好 |
| 深层网络 | 容易出问题 | 更稳定 |

### Pre-Norm 的优势

在 Pre-Norm 中，残差连接的 `+ x` 完全保留了原始输入，梯度可以直接流过：

```
∂(x + sa(ln(x)))/∂x = 1 + ∂sa/∂x × ∂ln/∂x
                      ↑
                   直通通道不受 LayerNorm 影响
```

这也是为什么 GPT-2 之后的主流模型都采用 Pre-Norm。

## 数据流对比

### v8 Block 数据流

```
x (B, T, 32)
    │
    ├──────────────────────┐
    ↓                      ↓
Attention(x)              x
    ↓                      │
    (+) ←──────────────────┘
    ↓
x + Attention(x)
    │
    ├──────────────────────┐
    ↓                      ↓
FFN(...)                  x + Attention(x)
    ↓                      │
    (+) ←──────────────────┘
    ↓
x + Attention(x) + FFN(...)
```

### v9 Block 数据流（Pre-Norm）

```
x (B, T, 32)
    │
    ├──────────────────────┐
    ↓                      │
LayerNorm                 │
    ↓                      │
Attention                 │
    ↓                      │
    (+) ←──────────────────┘
    ↓
x + Attention(LN(x))
    │
    ├──────────────────────┐
    ↓                      │
LayerNorm                 │
    ↓                      │
FFN                       │
    ↓                      │
    (+) ←──────────────────┘
    ↓
x + Attention(LN(x)) + FFN(LN(x))
```

### v9 完整数据流

```
Input → Token Embedding + Position Embedding
    ↓
Block 1: LN → Attention → Residual
         LN → FFN → Residual
    ↓
Block 2: LN → Attention → Residual
         LN → FFN → Residual
    ↓
Block 3: LN → Attention → Residual
         LN → FFN → Residual
    ↓
LayerNorm (final)
    ↓
LM Head → Output Logits
```

## 参数量对比

LayerNorm 每层有 2 × n_embed 个参数（γ 和 β）：

| 组件 | 新增参数 |
|------|---------|
| 每个 Block 的 ln1 | 2 × 32 = 64 |
| 每个 Block 的 ln2 | 2 × 32 = 64 |
| ln_final | 2 × 32 = 64 |
| **3 个 Block 总计** | **64 × 2 × 3 + 64 = 448** |

参数增加量很小，但带来的训练稳定性提升显著。

## LayerNorm 的可学习参数

```python
nn.LayerNorm(n_embed)
# 包含两个可学习参数：
# - weight (γ): 缩放因子，初始化为 1
# - bias (β): 偏移因子，初始化为 0
```

为什么需要 γ 和 β？

- 归一化后数据被"标准化"到均值0、方差1
- 但网络可能需要不同的分布
- γ 和 β 让网络"学习"最优的分布

```
output = normalized × γ + β
```

如果 γ=1, β=0，就是标准归一化。网络可以学习调整这两个参数。

## 下一步

v9 引入了 LayerNorm 稳定训练，但模型仍可能过拟合。v10 将引入 **Dropout**：

```python
# v10 Block（预告）
self.dropout = nn.Dropout(dropout)

def forward(self, x):
    x = x + self.dropout(self.sa(self.ln1(x)))
    x = x + self.dropout(self.ffwd(self.ln2(x)))
    return x
```

Dropout 在训练时随机"关闭"部分神经元，防止过拟合。

## 总结

| 版本 | 核心变化 |
|------|----------|
| v8 | 投影层 + FFN 扩展结构，增强表达能力 |
| v9 | LayerNorm，稳定训练过程 |

LayerNorm 是 Transformer 的"稳定器"：

- **Pre-Norm 结构**：归一化在残差之前，梯度流动更顺畅
- **最终归一化**：确保 LM Head 的输入分布稳定
- **小代价大收益**：参数增加极少，训练稳定性显著提升

至此，BabyGPT 的核心架构已经完整：Embedding → Blocks（Attention + FFN + Residual + LayerNorm）→ LM Head。后续版本主要是正则化和工程优化。
