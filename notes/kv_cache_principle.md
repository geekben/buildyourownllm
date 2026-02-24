### KV Cache 原理：推理优化的关键洞察

在自回归生成（如 GPT 推理）中，每生成一个新 token 都需要重新计算注意力。但有一个关键观察可以大幅优化计算量。

---

#### 关键洞察

##### 1. Query 只需要最后一行

预测下一个 token 时，我们只需要**最后一个位置的 Query** 与所有历史 Key 的点积：

```python
# generate 中的代码
logits, _ = self(idx)           # 计算整个序列
logits = logits[:, -1, :]       # 只取最后一个token的输出
```

所以中间位置的 Q 计算是冗余的。

##### 2. K 和 V 可以缓存

每生成一个新 token 时：
- **Query**：只需要计算新 token 的 Q（一行）
- **Key**：历史 K 不变，只需要计算新 token 的 K，拼接到缓存
- **Value**：历史 V 不变，只需要计算新 token 的 V，拼接到缓存

---

#### 对比：有/无 KV Cache

| 操作 | 无 KV Cache | 有 KV Cache |
|------|------------|-------------|
| 第 1 轮 | 计算 Q[0], K[0], V[0] | 计算 Q[0], K[0], V[0]，缓存 K,V |
| 第 2 轮 | 计算 Q[0:1], K[0:1], V[0:1] | 计算 Q[1], K[1], V[1]，拼接缓存 |
| 第 3 轮 | 计算 Q[0:2], K[0:2], V[0:2] | 计算 Q[2], K[2], V[2]，拼接缓存 |
| 第 n 轮 | 计算 O(n²) | 计算 O(n)，累积缓存 |

---

#### 示意图

```
无 KV Cache（每次重新计算全部）:
────────────────────────────────────
轮次  Token序列    计算量
1     [A]          Q[0], K[0], V[0]
2     [A,B]        Q[0,1], K[0,1], V[0,1]      ← 重复计算 A
3     [A,B,C]      Q[0,1,2], K[0,1,2], V[0,1,2] ← 重复计算 A,B
...
n     [A..N]       Q[0..n-1], K[0..n-1], V[0..n-1]  ← 总计算量 O(n²)

有 KV Cache（增量计算）:
────────────────────────────────────
轮次  Token序列    计算量                缓存状态
1     [A]          Q[0], K[0], V[0]     K_cache=[K0], V_cache=[V0]
2     [A,B]        Q[1], K[1], V[1]     K_cache=[K0,K1], V_cache=[V0,V1]
3     [A,B,C]      Q[2], K[2], V[2]     K_cache=[K0,K1,K2], V_cache=[V0,V1,V2]
...
n     [A..N]       Q[n-1], K[n-1], V[n-1]   ← 总计算量 O(n)
```

---

#### 计算复杂度对比

| 指标 | 无 KV Cache | 有 KV Cache |
|------|------------|-------------|
| 计算量 | O(n²) | O(n) |
| 显存占用 | O(block_size) | O(n) 随序列增长 |
| 每轮延迟 | 随序列增长 | 恒定 |

---

#### 为什么叫 KV Cache？

因为缓存的是 **Key** 和 **Value** 矩阵：

```
输入 token → Linear → K, Q, V
              ↓
         K_cache ───► 缓存
         Q          ───► 只计算当前 token，不缓存
         V_cache ───► 缓存
```

---

#### 实现示例

项目中 `babygpt_sample_with_kvcache.py` 展示了完整实现，核心修改：

```python
class Head(nn.Module):
    def __init__(self, head_size):
        ...
        self.cache_k = None  # KV Cache
        self.cache_v = None

    def forward(self, x, use_cache=False):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        if use_cache:
            if self.cache_k is None:
                # 首次，初始化缓存
                self.cache_k = k
                self.cache_v = v
            else:
                # 拼接历史缓存
                self.cache_k = torch.cat([self.cache_k, k], dim=1)
                self.cache_v = torch.cat([self.cache_v, v], dim=1)
            k = self.cache_k
            v = self.cache_v
        
        # 后续注意力计算不变...
```

---

#### 局限性

KV Cache 也有代价：
- **显存线性增长**：序列越长，缓存越大
- 长序列推理时可能成为瓶颈

解决方案：
- **PagedAttention**（vLLM）：分页管理 KV Cache
- **MQA / GQA**：多头共享 KV，减少缓存大小
- **Sliding Window**：限制缓存窗口大小
