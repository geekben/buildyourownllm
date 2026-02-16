# PyTorch 训练机制详解

## 问题引入

在 `pytorch_5min.py` 中，训练过程看起来只是一堆函数调用，有些函数甚至没有显式传入模型参数：

```python
model = SimpleNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(epochs):
    y_pred = model(x_train)       # 模型调用，没有显式传入模型参数
    loss = criterion(y_pred, y_train)
    
    optimizer.zero_grad()         # 清除梯度，没有传入模型参数
    loss.backward()               # 反向传播，没有传入模型参数
    optimizer.step()              # 更新参数，没有传入模型参数
```

这些函数调用是如何联系在一起，作用于 `SimpleNet` 实例的呢？

---

## 核心机制：PyTorch 的"魔法"

### 1. nn.Module 的 __call__ 方法

当你调用 `model(x_train)` 时，实际上调用的是 `nn.Module` 的 `__call__` 方法：

```python
# 你写的代码
y_pred = model(x_train)

# 实际发生的事情（简化版）
def __call__(self, *args, **kwargs):
    # 1. 前置钩子处理
    # 2. 调用 forward 方法
    result = self.forward(*args, **kwargs)
    # 3. 后置钩子处理
    return result
```

**关键点**：`model(x)` 等价于 `model.forward(x)`，但 `__call__` 会自动处理钩子函数。

### 2. model.parameters() 的魔法

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

`model.parameters()` 返回模型中所有可学习参数的迭代器：

```
SimpleNet
├── linear (nn.Linear)
│   ├── weight: tensor([[...]])  # 形状 (1, 1)
│   └── bias: tensor([...])      # 形状 (1,)
```

**原理**：
- `nn.Module` 内部维护一个 `_parameters` 字典
- 当你在 `__init__` 中定义 `self.linear = nn.Linear(1, 1)` 时
- `nn.Linear` 的 `weight` 和 `bias` 会被自动注册到父模块
- 调用 `model.parameters()` 会递归收集所有子模块的参数

### 3. Optimizer 如何"记住"模型参数

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

Optimizer 内部存储了参数的引用：

```python
class SGD(Optimizer):
    def __init__(self, params, lr):
        # params 是参数迭代器
        self.param_groups = []  # 存储参数引用
        for param in params:
            self.param_groups.append({
                'params': [param],  # 这里存的是引用！
                'lr': lr
            })
```

**关键**：Optimizer 存的是参数的引用，不是副本。所以：
- `optimizer.step()` 直接修改 `model.linear.weight` 和 `model.linear.bias`
- 不需要把模型传给 optimizer

### 4. 梯度是如何流动的

```python
y_pred = model(x_train)  # 前向传播
loss = criterion(y_pred, y_train)  # 计算损失
loss.backward()  # 反向传播
```

**计算图构建过程**：

```
x_train (输入)
    │
    ▼
linear.weight, linear.bias (参数)
    │
    ▼
y_pred = x @ weight + bias  (前向传播)
    │
    ▼
loss = MSELoss(y_pred, y_train)  (损失计算)
    │
    ▼
loss.backward()  (反向传播)
    │
    ├── linear.weight.grad = ∂loss/∂weight
    └── linear.bias.grad = ∂loss/∂bias
```

**关键**：每个 tensor 都有 `.grad` 属性，`backward()` 会自动计算梯度并存入这个属性。

### 5. optimizer.step() 如何更新参数

```python
optimizer.step()
```

等价于：

```python
# 伪代码
for param in model.parameters():
    if param.grad is not None:
        param = param - lr * param.grad  # SGD 更新规则
```

---

## 数据流向图解

```
┌─────────────────────────────────────────────────────────────────┐
│                        初始化阶段                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  model = SimpleNet()                                            │
│     │                                                           │
│     ├── self.linear = nn.Linear(1, 1)                          │
│     │       ├── weight: tensor([[?]])  # 随机初始化             │
│     │       └── bias: tensor([?])                               │
│     │                                                           │
│  optimizer = SGD(model.parameters(), lr=0.01)                  │
│     │       │                                                   │
│     │       └── 保存 weight, bias 的引用                        │
│     │                                                           │
│  criterion = MSELoss()                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        训练循环                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  for epoch in range(epochs):                                   │
│      │                                                          │
│      │  ┌─────────────────────────────────────────┐            │
│      │  │ 前向传播                                 │            │
│      │  ├─────────────────────────────────────────┤            │
│      │  │                                         │            │
│      │  │  y_pred = model(x_train)               │            │
│      │  │     └── y_pred = x @ weight + bias     │            │
│      │  │                                         │            │
│      │  │  loss = criterion(y_pred, y_train)     │            │
│      │  │     └── 构建计算图                      │            │
│      │  └─────────────────────────────────────────┘            │
│      │                                                          │
│      │  ┌─────────────────────────────────────────┐            │
│      │  │ 反向传播                                 │            │
│      │  ├─────────────────────────────────────────┤            │
│      │  │                                         │            │
│      │  │  optimizer.zero_grad()                 │            │
│      │  │     └── weight.grad = 0                │            │
│      │  │     └── bias.grad = 0                  │            │
│      │  │                                         │            │
│      │  │  loss.backward()                       │            │
│      │  │     └── weight.grad = ∂loss/∂weight    │            │
│      │  │     └── bias.grad = ∂loss/∂bias        │            │
│      │  │                                         │            │
│      │  │  optimizer.step()                      │            │
│      │  │     └── weight -= lr * weight.grad     │            │
│      │  │     └── bias -= lr * bias.grad         │            │
│      │  └─────────────────────────────────────────┘            │
│      │                                                          │
│      └── 重复直到收敛                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 关键概念总结

| 概念 | 说明 |
|------|------|
| `nn.Module` | 所有神经网络的基类，自动管理子模块和参数 |
| `parameters()` | 返回模型所有可学习参数的迭代器 |
| `__call__` | 让模型可以像函数一样调用，自动调用 forward |
| 计算图 | PyTorch 自动构建的有向无环图，记录运算关系 |
| `.grad` | 每个 tensor 的梯度属性，backward() 会填充它 |
| Optimizer | 持有参数引用，根据梯度更新参数 |

---

## 为什么看起来"没有联系"？

**因为 PyTorch 使用了隐式状态管理**：

1. **参数引用**：Optimizer 存储参数引用，不是参数值
2. **梯度属性**：梯度存在 tensor 的 `.grad` 属性中，不需要传递
3. **计算图**：前向传播自动构建计算图，反向传播自动追踪

这种设计的优点：
- 代码简洁，不需要手动传递大量参数
- 灵活，可以轻松修改模型结构
- 自动化程度高，减少出错机会

---

## 更多细节

### nn.Linear 内部结构

```python
class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Parameter(torch.randn(out_features, in_features))
        self.bias = Parameter(torch.randn(out_features))
    
    def forward(self, x):
        return x @ self.weight.T + self.bias
```

`Parameter` 是 `Tensor` 的子类，会被 `parameters()` 自动收集。

### MSELoss 的工作方式

```python
criterion = nn.MSELoss()
loss = criterion(y_pred, y_train)
```

等价于：

```python
loss = ((y_pred - y_train) ** 2).mean()
```

---

## 参考文件

- `/cn_nfs/luoben/buildyourownllm/pytorch_5min.py` - 本笔记解释的源代码

---

## SGD 优化器详解

### 什么是 SGD？

SGD (Stochastic Gradient Descent，随机梯度下降) 是最基础的优化算法。

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

### 更新公式

对于每个参数 θ：
```
θ_new = θ_old - lr × ∂loss/∂θ
```

具体到 `pytorch_5min.py` 中的例子：
```python
# 假设某次迭代：
# weight = 1.5, weight.grad = -0.8 (梯度)
# bias = 0.5, bias.grad = -0.3
# lr = 0.01

# 更新后：
weight_new = 1.5 - 0.01 × (-0.8) = 1.5 + 0.008 = 1.508
bias_new = 0.5 - 0.01 × (-0.3) = 0.5 + 0.003 = 0.503

# 梯度为负说明增加参数能降低loss，所以参数增大
```

### SGD 的三个核心方法

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 1. zero_grad() - 清除梯度
optimizer.zero_grad()
# 等价于：
for param in model.parameters():
    param.grad = None

# 2. step() - 更新参数
optimizer.step()
# 等价于：
for param in model.parameters():
    param.data -= lr * param.grad

# 3. state_dict() / load_state_dict() - 保存/加载优化器状态
```

### 为什么叫"随机"梯度下降？

| 名称 | 数据使用 | 特点 |
|------|---------|------|
| **GD** (梯度下降) | 每次用全部数据 | 准确但慢 |
| **SGD** (随机梯度下降) | 每次用单个样本 | 快但噪声大 |
| **Mini-batch SGD** | 每次用一小批数据 | 平衡速度和稳定性 |

PyTorch 的 `SGD` 实际是 Mini-batch SGD，因为训练时通常传入一批数据。

---

## MSELoss 损失函数详解

### 什么是 MSE？

MSE (Mean Squared Error，均方误差) 衡量预测值和真实值的差距。

```python
criterion = nn.MSELoss()
loss = criterion(y_pred, y_train)
```

### 数学公式

```
MSE = (1/n) × Σ(y_pred - y_true)²
```

### 示例计算

```python
y_pred = torch.tensor([2.0, 4.0, 6.0])   # 预测值
y_true = torch.tensor([1.8, 4.2, 5.5])   # 真实值

# MSE 计算：
# 差值: [0.2, -0.2, 0.5]
# 平方: [0.04, 0.04, 0.25]
# 均值: (0.04 + 0.04 + 0.25) / 3 = 0.11

loss = ((y_pred - y_true) ** 2).mean()  # 0.11
```

### 为什么用 MSE 而不是其他损失？

| 损失函数 | 公式 | 适用场景 |
|---------|------|---------|
| **MSE** | (y_pred - y_true)² | 回归问题，对大误差惩罚重 |
| **MAE** | \|y_pred - y_true\| | 回归问题，对异常值不敏感 |
| **CrossEntropy** | -Σy_true×log(y_pred) | 分类问题 |

在 `pytorch_5min.py` 中，我们学习的是线性关系 `y = 2x + 1`，这是一个回归问题，所以用 MSE。

### MSE 的梯度

```python
# MSE = (1/n) × Σ(y_pred - y_true)²
# 对 y_pred 求导：
# ∂MSE/∂y_pred = (2/n) × (y_pred - y_true)
```

这就是 `loss.backward()` 会计算并存入 `y_pred.grad` 的值，然后通过链式法则传播到 `weight.grad` 和 `bias.grad`。

---

## 本例中的参数：w 和 b

在 `pytorch_5min.py` 中：

```python
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # 内部定义了 weight 和 bias
```

`nn.Linear(1, 1)` 创建了：
- `weight`: 形状为 (1, 1) 的张量，初始随机值
- `bias`: 形状为 (1,) 的张量，初始随机值

训练目标是让：
- `weight` → 2.0 (真实斜率)
- `bias` → 1.0 (真实截距)

```
初始状态:    weight ≈ 0.5,  bias ≈ -0.2  (随机)
↓ 训练1000轮后
最终状态:    weight ≈ 2.0,  bias ≈ 1.0   (接近真实值)
```

---

## 总结：联系的纽带

```
model = SimpleNet()
    │
    ├── model.linear.weight (tensor, w)
    └── model.linear.bias (tensor, b)
            │
            ▼ (引用传递)
optimizer = SGD(model.parameters(), lr=0.01)
    │
    └── 保存了 w, b 的引用
    
训练循环:
    y_pred = w × x + b           # 前向传播
    loss = MSELoss(y_pred, y)    # 计算 loss
    loss.backward()              # 计算梯度 → w.grad, b.grad
    optimizer.step()             # 更新 → w -= lr × w.grad
                                 #      → b -= lr × b.grad
```

**核心纽带**：`tensor` 的引用传递 + `.grad` 属性的隐式存储。
