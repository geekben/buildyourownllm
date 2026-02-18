# PyTorch 张量 vs Python 列表

## 数据结构对比

### Python 嵌套列表

```python
transition = [[0 for _ in range(vocab_size)]
              for _ in range(vocab_size)]
```

**内存布局：**

```
transition ──┐
             ▼
           ┌───┐
           │ * │──→ [0, 0, 0, ...]  ← 独立对象
           ├───┤
           │ * │──→ [0, 0, 0, ...]  ← 独立对象
           ├───┤
           │ * │──→ [0, 0, 0, ...]  ← 独立对象
           └───┘

每个元素都是独立的 Python 对象，内存分散
访问需要多次指针跳转
```

### PyTorch 张量

```python
transition = torch.zeros((vocab_size, vocab_size))
```

**内存布局：**

```
transition ──┐
             ▼
           ┌─────────────────────────────────┐
           │ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 │  ← 连续内存块
           └─────────────────────────────────┘

所有数据在连续内存中存储
直接通过偏移量访问
```

---

## 索引方式对比

### Python 列表：两次索引

```python
value = transition[x][y]
```

**执行过程：**

```
1. transition[x]  → 取第 x 行，返回一个列表对象
2. [y]            → 在返回的列表中取第 y 个元素

共两次内存访问 + 一次临时对象创建
```

### PyTorch 张量：单次索引

```python
value = transition[x, y]
```

**执行过程：**

```
1. 直接计算偏移量: offset = x * vocab_size + y
2. 一次内存访问取值

更高效，无需创建中间对象
```

---

## 性能对比

```python
import torch
import time

vocab_size = 10000

# Python 列表
list_matrix = [[0] * vocab_size for _ in range(vocab_size)]

# PyTorch 张量
tensor_matrix = torch.zeros((vocab_size, vocab_size))

# 测试写入性能
iterations = 1000000

# Python 列表
start = time.time()
for _ in range(iterations):
    i, j = divmod(_, vocab_size)
    list_matrix[i % vocab_size][j] += 1
list_time = time.time() - start

# PyTorch 张量
start = time.time()
for _ in range(iterations):
    i, j = divmod(_, vocab_size)
    tensor_matrix[i % vocab_size, j] += 1
tensor_time = time.time() - start

print(f"Python 列表: {list_time:.3f}s")
print(f"PyTorch 张量: {tensor_time:.3f}s")
print(f"加速比: {list_time / tensor_time:.2f}x")
```

---

## 训练代码对比

### simplebigrammodel_with_comments.py（Python 列表）

```python
class BigramLanguageModel():
    def __init__(self, vocab_size: int):
        self.transition = [[0 for _ in range(vocab_size)]
                          for _ in range(vocab_size)]

# 训练
for iter in range(max_iters):
    x_batch, y_batch = get_batch(tokens, batch_size, block_size)
    for i in range(len(x_batch)):
        for j in range(len(x_batch[i])):
            x = x_batch[i][j]      # 两次索引
            y = y_batch[i][j]      # 两次索引
            model.transition[x][y] += 1  # 两次索引
```

### simplebigrammodel_torch.py（PyTorch 张量）

```python
class BigramLanguageModel():
    def __init__(self, vocab_size: int):
        self.transition = torch.zeros((vocab_size, vocab_size), device=device)

# 训练
for iter in range(max_iters):
    x_batch, y_batch = get_batch(tokens, batch_size, block_size)
    for i in range(batch_size):
        for j in range(block_size):
            x = x_batch[i, j]      # 单次索引
            y = y_batch[i, j]      # 单次索引
            model.transition[x, y] += 1  # 单次索引
```

---

## 优化点总结

| 方面 | Python 列表 | PyTorch 张量 |
|------|-------------|--------------|
| **内存布局** | 分散，指针引用 | 连续，紧凑存储 |
| **索引方式** | `arr[i][j]` 两次 | `arr[i, j]` 单次 |
| **内存访问** | 多次指针跳转 | 直接偏移计算 |
| **临时对象** | 每次索引创建 | 无 |
| **GPU 支持** | 无 | 支持 CUDA |
| **向量化操作** | 无 | 支持 |

---

## GPU 加速潜力

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
transition = torch.zeros((vocab_size, vocab_size), device=device)
```

当数据量大时，GPU 并行计算可带来 **10-100x** 加速。

---

## 总结

将 Python 列表替换为 PyTorch 张量的核心优势：

1. **内存效率**：连续存储，减少内存碎片
2. **访问速度**：单次索引，避免临时对象
3. **GPU 支持**：为大规模计算加速打基础
4. **生态兼容**：可直接使用 PyTorch 丰富的 API
