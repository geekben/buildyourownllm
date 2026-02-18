# torch.clamp 与 torch.multinomial 详解

## torch.clamp

### 函数签名

```python
torch.clamp(input, min=None, max=None)
```

### 作用

将张量中的值限制在 `[min, max]` 范围内：
- 小于 `min` 的值 → 变为 `min`
- 大于 `max` 的值 → 变为 `max`
- 范围内的值 → 保持不变

### 示例

```python
import torch

x = torch.tensor([-2, 0, 3, 5, 10])
print(torch.clamp(x, min=1, max=8))
# 输出: tensor([1, 1, 3, 5, 8])
```

### 在生成代码中的应用

```python
probs = logits / torch.clamp(logits.sum(dim=-1, keepdim=True), min=1.0)
```

**目的：防止除零错误**

```
logits: [0, 0, 0, 0]  (从未见过的 token 转移)
sum:    0
直接除: [0/0, 0/0, 0/0, 0/0] → [nan, nan, nan, nan] ❌

clamp 后:
sum:    max(0, 1.0) = 1.0
除法:   [0, 0, 0, 0] / 1.0 = [0, 0, 0, 0] ✓
```

---

## torch.multinomial

### 函数签名

```python
torch.multinomial(input, num_samples, replacement=True)
```

### 作用

根据输入的概率分布进行**多项式采样**（加权随机抽样）。

### 参数说明

| 参数 | 说明 |
|------|------|
| `input` | 概率分布张量，形状 `(n,)` 或 `(B, n)` |
| `num_samples` | 采样次数 |
| `replacement` | 是否放回采样 |

### 示例

```python
import torch

# 单个概率分布采样
probs = torch.tensor([0.1, 0.3, 0.6])  # 概率和为 1
print(torch.multinomial(probs, num_samples=1))
# 约 10% 概率返回 0
# 约 30% 概率返回 1
# 约 60% 概率返回 2

# 批量采样
batch_probs = torch.tensor([
    [0.1, 0.3, 0.6],
    [0.7, 0.2, 0.1]
])
print(torch.multinomial(batch_probs, num_samples=1))
# 形状: (2, 1)，每行独立采样
```

### 在生成代码中的应用

```python
next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
```

**流程图：**

```
probs (B, vocab_size)
        │
        ▼
┌─────────────────────┐
│  torch.multinomial  │  按概率随机采样
│  (加权随机选择)      │
└─────────────────────┘
        │
        ▼
next_token (B, 1)
```

---

## 完整流程

```python
# logits 是转移计数值（整数）
logits = model(idx)[:, -1, :]          # (B, vocab_size)

# 步骤1: 归一化为概率分布
probs = logits / torch.clamp(logits.sum(dim=-1, keepdim=True), min=1.0)

# 步骤2: 按概率采样下一个 token
next_token = torch.multinomial(probs, num_samples=1)
```

**为什么不使用 softmax？**

| 方法 | 适用场景 |
|------|----------|
| `softmax` | 神经网络输出的原始分数，需要转换为概率 |
| 直接归一化 | 已有计数值，直接转为最大似然概率 |

本代码中 `logits` 是统计的转移次数（计数矩阵），直接归一化即可得到概率分布。

---

## 总结

| 函数 | 作用 | 关键点 |
|------|------|--------|
| `torch.clamp` | 限制数值范围 | 防止除零，确保数值稳定 |
| `torch.multinomial` | 概率采样 | 按权重随机选择，支持批量操作 |
