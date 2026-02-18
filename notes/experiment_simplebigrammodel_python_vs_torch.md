# 实验：Python 列表 vs PyTorch 张量性能对比

## 实验背景

对比 `simplebigrammodel_with_comments.py`（Python 列表）和 `simplebigrammodel_torch.py`（PyTorch 张量）的性能差异。

---

## 实验结果

### Python 列表版本

```bash
time python3 simplebigrammodel_with_comments.py
```

```
春江红紫霄效颦。

怎。
兰修月。
两个事对西风酒伴寄我登临，看雪惊起步，总不与泪满南园春来。
最关上阅。
信断，名姝，夜正坐认旧武仙 朱弦。

岁，回。


看一丝竹。
愿皇受风，当。

妆一笑时，不堪
----------
往事多闲田舍、十三楚珪
酒困不须紫芝兰花痕皱，青步虹。
暗殿人物华高层轩者，临江渌池塘。
三峡。
天、彩霞冠
燕翻云垂杨、一声羌笛罢瑶觥船窗幽园春生阵。
长桥。
无恙，中有心期。

开处。
燕姹绿遍，烂□
----------

real    0m6.318s
user    0m6.015s
sys     0m0.295s
```

### PyTorch 张量版本

```bash
time python3 simplebigrammodel_torch.py
```

```
春江句，一声断多酌玻璃。
香透，醉魂处里思无名呼蟾飞。
念丹榭画不早梅，拼一曲回旋温神京镗
到前、江月，不怕销凝。

画。
满庭院悄怯，只把人暗向月。


上有飞动笙。
饮加焉美东风进奉高，楚腰身元干。

----------
往事，情极星斗转盼。
娉婷。
天消息。

应萝。
指，驻金门系，地。

又成休。
还休说憔悴老。
杨花嫩。
还是欢声，万斛、鹊楼前回歌罢熏香尘清夜半春，荣钟相并壬。
西。
却。
中春心授、才情争圆光映阑、
----------

real    0m51.604s
user    1m2.649s
sys     0m6.726s
```

### 时间对比

| 版本 | 时间 | 相对倍数 |
|------|------|----------|
| Python 列表 | 6.3s | 1x |
| PyTorch 张量 | 51.6s | **8.2x 慢** |

---

## 原因分析

### 为什么 PyTorch 版本更慢？

**当前代码 `device='cpu'`，不存在 CPU-GPU 数据传输问题。**

性能差距的主要原因：

#### 1. 张量索引开销

```python
# Python 列表：直接指针访问
value = list_matrix[x][y]  # 两次指针跳转，极快

# PyTorch 张量：计算 + 检查
value = tensor_matrix[x, y]  
# 内部流程：
# 1. 计算 offset = x * stride[0] + y * stride[1]
# 2. 边界检查
# 3. 数据类型检查
# 4. 返回张量视图（创建新对象）
```

#### 2. Python 循环无法利用向量化

```python
# 当前代码结构（两个版本都是）
for i in range(batch_size):      # Python 循环
    for j in range(block_size):  # Python 循环
        model.transition[x, y] += 1  # 单点操作
```

**GPU 的优势在于大规模并行**，但 Python 循环逐个操作完全无法利用：
- 每次循环迭代都是串行的
- 无法批量并行处理
- PyTorch 的向量化能力未使用

#### 3. 额外的框架开销

| 操作 | Python 列表 | PyTorch 张量 |
|------|-------------|--------------|
| 索引返回 | 原始值 | 张量视图对象 |
| 增量操作 | 原地修改 | 可能创建临时张量 |
| 内存管理 | 简单 | 引用计数 + 自动求导追踪准备 |

---

## 如果使用 GPU 会怎样？

**当前代码结构下，用 GPU 会更慢。**

```python
# 假设 device='cuda'
for i in range(batch_size):
    for j in range(block_size):
        model.transition[x, y] += 1
        # 每次 += 1 都会：
        # 1. CPU 发起请求
        # 2. 数据传到 GPU（如果需要）
        # 3. GPU 执行微小操作
        # 4. 结果同步回 CPU（如果需要读取）
```

**GPU 开销来源：**
- 内核启动开销：每次操作 ~微秒级（CPU 单次操作 ~纳秒级）
- 数据传输：CPU ↔ GPU 通信延迟
- 同步等待：Python 循环需要等待 GPU 完成

---

## 输出结果不同的原因

两个版本使用了**不同的随机采样机制**：

```python
# Python 版本
random.seed(42)
next_token = random.choices(range(vocab_size), weights=logits, k=1)[0]

# PyTorch 版本
torch.manual_seed(42)
next_token = torch.multinomial(probs, num_samples=1)
```

| 对比项 | Python 版本 | PyTorch 版本 |
|--------|-------------|--------------|
| 随机数生成器 | `random` 模块 | `torch` 随机模块 |
| 采样函数 | `random.choices()` | `torch.multinomial()` |
| 种子设置 | `random.seed(42)` | `torch.manual_seed(42)` |

即使种子相同，不同实现的随机算法不同，产生的序列自然不同。

**结论：输出不同是符合预期的。**

---

## 如何发挥 PyTorch/GPU 优势？

### 向量化训练

```python
# 原始：Python 循环（慢）
for i in range(batch_size):
    for j in range(block_size):
        model.transition[x_batch[i,j], y_batch[i,j]] += 1

# 优化：向量化操作（快）
flat_x = x_batch.flatten()
flat_y = y_batch.flatten()
indices = flat_x * vocab_size + flat_y
model.transition.view(-1).scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.float))
```

### 向量化 forward

```python
# 原始：Python 循环
for b in range(B):
    for t in range(T):
        result[b][t] = self.transition[idx[b][t]]

# 优化：直接索引（PyTorch 自动并行）
result = self.transition[idx]  # shape: (B, T, vocab_size)
```

---

## 总结

| 问题 | 答案 |
|------|------|
| PyTorch 版本为什么更慢？ | Python 循环 + 张量单点操作开销，无法利用向量化 |
| GPU 能加速吗？ | 当前代码结构下不能，反而更慢（内核启动开销） |
| 输出不同正常吗？ | 正常，使用了不同的随机采样机制 |
| 何时有优势？ | 使用向量化操作，批量处理数据时 |

**核心教训**：PyTorch/GPU 的优势来自向量化并行计算，Python 循环逐个操作无法利用这些优势。
