# BabyGPT v10: Dropout 正则化

## 概述

v10 在 v9 的基础上引入了 **Dropout（随机失活）**，这是深度学习中防止过拟合的经典正则化技术。训练时随机"关闭"部分神经元，迫使网络学习更鲁棒的特征。

## v9 vs v10 核心差异对比

### 1. 新增 dropout 参数

```python
dropout = 0.2  # dropout的比例
```

### 2. Head 类的变化

**v9 Head：**
```python
def forward(self, x):
    ...
    wei = F.softmax(wei, dim=-1)
    out = wei @ v
    return out
```

**v10 Head：**
```python
def __init__(self, head_size):
    ...
    self.dropout = nn.Dropout(dropout)  # 新增

def forward(self, x):
    ...
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)  # 新增：对注意力权重做 dropout
    out = wei @ v
    return out
```

### 3. MultiHeadAttention 类的变化

**v9 MultiHeadAttention：**
```python
def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    return self.proj(out)
```

**v10 MultiHeadAttention：**
```python
def __init__(self, num_heads, head_size):
    ...
    self.dropout = nn.Dropout(dropout)  # 新增

def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.proj(out)
    out = self.dropout(out)  # 新增：对投影输出做 dropout
    return out
```

### 4. FeedForward 类的变化

**v9 FeedForward：**
```python
self.net = nn.Sequential(
    nn.Linear(n_embed, n_embed * 4),
    nn.ReLU(),
    nn.Linear(n_embed * 4, n_embed),
)
```

**v10 FeedForward：**
```python
self.net = nn.Sequential(
    nn.Linear(n_embed, n_embed * 4),
    nn.ReLU(),
    nn.Linear(n_embed * 4, n_embed),
    nn.Dropout(dropout),  # 新增：在最后添加 dropout
)
```

### 5. 架构对比

| 方面 | v9 | v10 |
|------|-----|-----|
| 正则化 | 无 | Dropout |
| Dropout 位置 | - | 注意力权重、投影输出、FFN 输出 |
| 过拟合风险 | 较高 | 较低 |
| 训练时行为 | 全部神经元参与 | 部分神经元随机"关闭" |

## 什么是 Dropout？

### 定义

Dropout 在训练时以概率 `p` 随机将神经元输出置为 0：

```python
nn.Dropout(p=0.2)  # 20% 的神经元被置为 0
```

### 工作原理

**训练时：**
```
输入: [0.5, 0.3, 0.8, 0.2]
Dropout(0.5): [0.5, 0.0, 0.8, 0.0]  # 随机置 0，剩余值放大
              [1.0, 0.0, 1.6, 0.0]  # 除以 (1-p) 保持期望值不变
```

**推理时：**
```
输入: [0.5, 0.3, 0.8, 0.2]
Dropout: [0.5, 0.3, 0.8, 0.2]  # 不做任何处理，直接传递
```

### PyTorch 自动处理

```python
model.train()  # Dropout 生效
model.eval()   # Dropout 关闭
```

## 为什么需要 Dropout？

### 问题：过拟合

当模型参数量较大、训练数据较少时，模型容易"记住"训练数据：

```
训练集：loss 很低，准确率很高
验证集：loss 很高，准确率很低
```

这就是**过拟合**：模型学到了训练数据的"噪声"而非真正的规律。

### 解决方案：Dropout

Dropout 通过随机"关闭"神经元，迫使网络：

1. **不依赖单一神经元**：每个特征都需要被多个神经元表达
2. **学习冗余表示**：增强模型的鲁棒性
3. **隐式集成**：相当于训练了无数个子网络的集成

### 深入理解：捷径学习 vs 真正学习

神经网络有一个"懒惰"倾向：倾向于找"捷径"——用最少的计算量降低损失。

| 学习方式 | 行为 | 泛化能力 |
|---------|------|---------|
| **捷径学习** | 记住"这个句子以'春'开头，下一个词通常是'江'" | 差，稍变就不会 |
| **真正学习** | 学习"春江"作为词组的搭配规律 | 强，能举一反三 |

**神经网络的具体表现：**

| 捷径学习 | 真正学习 |
|---------|---------|
| 记住特定的 token 组合 | 学习多种上下文关联方式 |
| 某几个神经元"包办"所有判断 | 多个神经元协作，互有备份 |
| 依赖某个特定的注意力模式 | 学习多种等价的注意力模式 |

**Dropout 的作用就是打断这种"捷径"：**

```
无 Dropout：
  神经元A → "我发现一个捷径，不用管别人了"
  结果：模型记住特定模式，泛化差

有 Dropout：
  神经元A → "我发现了捷径"
  Dropout → "对不起，A 今天休息"
  神经元B、C → "我们被迫学会这个规律"
  结果：多个神经元都掌握了规律，泛化强
```

**本质**：Dropout 强迫模型**充分从各个维度学习数据集能提供的知识**，而不是仅仅找到正确答案的捷径就草草了事。这样模型才能真正掌握数据集的内在规律，而不是忽略它们。

### 类比理解

| 类比 | 说明 |
|------|------|
| 团队协作 | 每次开会随机缺勤 20% 成员，迫使团队不依赖任何个人 |
| 考试准备 | 不能只依赖某个知识点，必须全面掌握 |
| 备份系统 | 关键岗位有备份，任何一人缺席都不影响运作 |

**更具体的类比：球队训练**

| 方式 | 类比 |
|------|------|
| v9（无 Dropout） | 每次训练全员到场，比赛中某主力受伤就崩盘 |
| v10（有 Dropout） | 每次训练随机缺勤 20%，每个人都要能顶替别人，比赛中任何人员变动都能应对 |

## Dropout 在 Transformer 中的位置

v10 在三个位置添加了 Dropout：

```
                    ┌─────────────────────────────────────┐
                    │           Block 内部                 │
                    └─────────────────────────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
        ▼                            ▼                            ▼
   ┌─────────┐               ┌──────────────┐            ┌─────────────┐
   │  Head   │               │ MultiHeadAtt │            │   FFN       │
   └─────────┘               └──────────────┘            └─────────────┘
        │                            │                            │
        ▼                            ▼                            ▼
  Softmax 后                   Proj 后                      最后
  Dropout(wei)                 Dropout(out)                Dropout
```

### 1. Head 内的 Dropout

```python
wei = F.softmax(wei, dim=-1)
wei = self.dropout(wei)  # 对注意力权重做 dropout
```

**作用**：防止模型过度依赖某些特定的注意力模式。

### 2. MultiHeadAttention 的 Dropout

```python
out = self.proj(out)
out = self.dropout(out)  # 对多头输出做 dropout
```

**作用**：防止模型过度依赖某些特定的头。

### 3. FeedForward 的 Dropout

```python
nn.Linear(n_embed * 4, n_embed),
nn.Dropout(dropout),  # FFN 输出做 dropout
```

**作用**：防止 FFN 过拟合。

## 数据流对比

### v9 数据流

```
Attention:
  softmax(wei) → wei @ v → concat → proj → output

FFN:
  Linear → ReLU → Linear → output
```

### v10 数据流

```
Attention:
  softmax(wei) → Dropout → wei @ v → concat → proj → Dropout → output

FFN:
  Linear → ReLU → Linear → Dropout → output
```

## Dropout 率的选择

| Dropout 率 | 适用场景 |
|-----------|---------|
| 0.0 | 无正则化 |
| 0.1 ~ 0.3 | 常用范围（v10 使用 0.2） |
| 0.5 | 较强正则化，常用于全连接层 |
| > 0.5 | 过强，可能影响学习 |

**经验法则**：
- 小模型、小数据：可以用较高的 dropout
- 大模型、大数据：使用较低的 dropout
- Transformer 通常使用 0.1 ~ 0.3

## Dropout 与 BatchNorm 的关系

| 特性 | Dropout | BatchNorm |
|------|---------|-----------|
| 目的 | 防止过拟合 | 稳定训练 |
| 训练/推理行为 | 不同（关闭） | 不同（使用 running stats） |
| 适用位置 | 全连接层、注意力 | 每层输入 |

在 Transformer 中，Dropout 和 LayerNorm 配合使用，共同提升训练稳定性。

## 常见问题

### Q: 为什么推理时要关闭 Dropout？

**A**: Dropout 是训练时的正则化手段。推理时需要确定性输出，不应该有随机性。

```python
model.eval()  # 自动关闭 Dropout
```

### Q: Dropout 会影响模型性能吗？

**A**: 训练时可能略微降低训练集准确率，但能提升验证集/测试集表现。这是预期行为——"牺牲"训练表现换取更好的泛化。

### Q: 为什么要在注意力权重上加 Dropout？

**A**: 防止模型学习"作弊"——只依赖某些特定的 token 组合。Dropout 强制模型学习多种注意力模式。

## 下一步

v10 引入了 Dropout 防止过拟合，至此 BabyGPT 的架构已基本完整。v11 将进行 **超参数调优**：

- 增加网络深度：`n_layer = 6`
- 增加模型宽度：`n_embed = 384`
- 增加注意力头：`n_head = 6`
- 扩大上下文窗口：`block_size = 256`

## 总结

| 版本 | 核心变化 |
|------|----------|
| v9 | LayerNorm，稳定训练过程 |
| v10 | Dropout，防止过拟合 |

Dropout 是深度学习的"保险机制"：

- **简单有效**：一行代码，显著提升泛化能力
- **广泛使用**：几乎所有现代神经网络都采用
- **无额外参数**：只是训练时的随机操作

至此，BabyGPT 已具备现代 Transformer 的所有核心组件：

```
Token Embedding + Position Embedding
    ↓
Block × n_layer:
  - LayerNorm
  - Multi-Head Attention + Dropout
  - Residual Connection
  - LayerNorm
  - Feed-Forward + Dropout
  - Residual Connection
    ↓
LayerNorm → LM Head
```
