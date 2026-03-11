# BabyGPT v12: Wandb 实验跟踪

## 概述

v12 在 v11 的基础上引入了 **Wandb（Weights & Biases）实验跟踪**，将训练过程可视化、云端同步、便于对比实验。这是从"玩具项目"走向"工程实践"的关键一步。

## v11 vs v12 核心差异对比

### 1. 新增 Wandb 导入和初始化

**v11：**
```python
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List
import time
```

**v12：**
```python
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List
import time
import wandb  # 新增

...

wandb.init(
    project="babygpt",
    config={
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "block_size": block_size,
        "n_embed": n_embed,
        "n_head": n_head,
        "n_layer": n_layer,
        "dropout": dropout,
    }
)
```

### 2. 训练循环中的日志记录

**v11：**
```python
if iter % eval_interval == 0:
    losses = estimate_loss(model, data, batch_size, block_size, eval_iters)
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
```

**v12：**
```python
if iter % eval_interval == 0:
    losses = estimate_loss(model, data, batch_size, block_size, eval_iters)
    wandb.log({
        "train_loss": losses['train'],
        "val_loss": losses['val'],
        "tokens_per_sec": tokens_per_sec,
        "iteration": iter
    })
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, speed: {tokens_per_sec:.2f} tokens/sec, time: {int(elapsed_mins)}m {elapsed_secs:.1f}s")
```

### 3. 模型保存

**v11：** 无模型保存

**v12：**
```python
save_path = 'model.pth'
torch.save(model.state_dict(), save_path)
print(f"模型已保存到{save_path}")
```

### 4. 参数调整

| 参数 | v11 | v12 |
|------|-----|-----|
| `eval_interval` | 200 | 50 |

更频繁的评估间隔，便于 Wandb 绘制更平滑的曲线。

### 5. 架构对比

| 方面 | v11 | v12 |
|------|-----|-----|
| 实验跟踪 | 无（仅终端打印） | Wandb 云端可视化 |
| 模型保存 | 无 | 保存到 model.pth |
| 评估间隔 | 200 步 | 50 步 |
| 时间显示 | 总时间 | 分:秒格式 |

## 什么是 Wandb？

### 定义

Wandb（Weights & Biases）是一个机器学习实验跟踪平台，提供：

- **实时可视化**：训练曲线、指标对比
- **超参数记录**：自动保存实验配置
- **云端同步**：多机实验统一管理
- **团队协作**：共享实验结果
- **模型版本管理**：追踪模型迭代历史

### 使用流程

```
1. pip install wandb
2. wandb login  # 输入 API key
3. 代码中 wandb.init() 初始化
4. wandb.log() 记录指标
5. 访问 wandb.ai 查看结果
```

### Wandb 界面示例

```
┌─────────────────────────────────────────────────────────────┐
│  BabyGPT - wandb.ai/simpxx/babygpt                          │
├─────────────────────────────────────────────────────────────┤
│  Charts                          │  Runs                    │
│  ┌──────────────────────────┐    │  ┌─────────────────────┐ │
│  │    train_loss  val_loss  │    │  │ absurd-frog-1      │ │
│  │  8.0 ─                    │    │  │ lr: 3e-4           │ │
│  │  6.0 ────\                │    │  │ n_layer: 6         │ │
│  │  4.0      ────\__________ │    │  │ best_val: 3.98     │ │
│  │  2.0                      │    │  │                     │ │
│  │     0   1000  2000  5000  │    │  │ romantic-dog-2     │ │
│  └──────────────────────────┘    │  │ lr: 1e-4 ...       │ │
│                                   │  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 为什么需要实验跟踪？

### 问题：实验管理的混乱

在机器学习实验中，常见问题：

| 问题 | 后果 |
|------|------|
| 忘记记录超参数 | 不知道哪个配置效果好 |
| 终端日志丢失 | 无法回顾训练过程 |
| 手动记录表格 | 效率低、易出错 |
| 多机实验 | 结果分散，难以对比 |

### 解决方案：Wandb

```python
# 一次配置，自动记录
wandb.init(project="babygpt", config={...})
wandb.log({"train_loss": ..., "val_loss": ...})

# 云端自动：
# - 保存所有超参数
# - 绘制训练曲线
# - 对比多次实验
# - 生成报告
```

### 类比理解

| 方式 | 类比 |
|------|------|
| v11（无跟踪） | 在草稿纸上记笔记，容易丢失 |
| v12（Wandb） | 用 GitHub 管理代码，有完整历史记录 |

## Wandb 核心功能

### 1. 实时训练曲线

```
Loss 曲线：
8.0 ─┐
     │\
6.0 ─│ \        训练集 (蓝色)
     │  \___    验证集 (橙色)
4.0 ─│      ────\____
     │                ────
2.0 ─│                     ───
     └──────────────────────────
     0   1k   2k   3k   4k   5k
```

**观察重点**：
- 曲线是否收敛？
- 训练集和验证集差距（过拟合指标）
- 是否有异常波动？

### 2. 超参数对比

```
┌─────────────────────────────────────────────┐
│  Run              │ LR    │ Layers │ Best Val │
├─────────────────────────────────────────────┤
│  absurd-frog-1    │ 3e-4  │ 6      │ 3.98     │
│  romantic-dog-2   │ 1e-4  │ 6      │ 4.12     │
│  happy-cat-3      │ 3e-4  │ 4      │ 4.35     │
└─────────────────────────────────────────────┘
```

一目了然看出哪个配置最优。

### 3. 系统监控

Wandb 自动记录：
- GPU 使用率
- 内存占用
- 训练速度（tokens/sec）

### 4. 模型版本

配合 `torch.save()`，可以追踪模型迭代历史。

## v12 的完整训练输出示例

```
wandb: Using wandb-core as the SDK backend.
wandb: Currently logged in as: simpxx (simpxx-zhejiang-university).
wandb: Tracking run with wandb version 0.18.3
wandb: Syncing run absurd-frog-1
wandb: ⭐️ View project at https://wandb.ai/simpxx/babygpt
wandb: 🚀 View run at https://wandb.ai/.../runs/ysgr3tei

step 0: train loss 8.0529, val loss 8.0512, speed: 55304.00 tokens/sec, time: 0m 0.3s
step 50: train loss 5.9337, val loss 6.0072, speed: 102707.49 tokens/sec, time: 0m 8.1s
...
step 4950: train loss 2.8944, val loss 3.9963, speed: 104340.95 tokens/sec, time: 12m 57.4s

模型已保存到 model.pth
wandb: 🚀 View run absurd-frog-1 at https://wandb.ai/...
```

## 模型保存与加载

### 保存模型

```python
torch.save(model.state_dict(), 'model.pth')
```

`state_dict()` 返回模型所有参数的字典：

```
{
    'token_embedding_table.weight': tensor(...),
    'blocks.0.sa.heads.0.key.weight': tensor(...),
    ...
}
```

### 加载模型

```python
model = BabyGPT(vocab_size, block_size, n_embed)
model.load_state_dict(torch.load('model.pth'))
model.eval()  # 切换到推理模式
```

### 为什么要保存模型？

| 场景 | 说明 |
|------|------|
| 推理部署 | 训练后加载模型生成文本 |
| 继续训练 | 从 checkpoint 恢复训练 |
| 实验对比 | 保存不同配置的最佳模型 |
| 模型分享 | 与他人共享训练好的模型 |

## eval_interval 调整的意义

| 值 | 优点 | 缺点 |
|----|------|------|
| 200 | 评估开销少 | 曲线粗糙，难以观察细节 |
| 50 | 曲线平滑，细节清晰 | 评估开销增加 4 倍 |

对于 Wandb 可视化，更频繁的评估可以绘制更平滑的曲线，便于观察训练动态。

## 实验跟踪的最佳实践

### 1. 有意义的运行名称

```python
wandb.init(
    project="babygpt",
    name=f"lr_{learning_rate}_layers_{n_layer}",  # 自动命名
    config={...}
)
```

### 2. 记录所有超参数

```python
config = {
    # 模型架构
    "n_embed": n_embed,
    "n_head": n_head,
    "n_layer": n_layer,
    
    # 训练配置
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "max_iters": max_iters,
    
    # 正则化
    "dropout": dropout,
}
```

### 3. 记录关键指标

```python
wandb.log({
    "train_loss": train_loss,
    "val_loss": val_loss,
    "learning_rate": current_lr,  # 如果使用学习率调度
    "epoch": epoch,
})
```

### 4. 保存最佳模型

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), 'best_model.pth')
    wandb.save('best_model.pth')  # 同步到 Wandb 云端
```

## Wandb vs 其他工具

| 工具 | 优点 | 缺点 |
|------|------|------|
| **Wandb** | 功能全面、云端同步、团队协作 | 需要网络、免费版有限制 |
| TensorBoard | 本地运行、无网络依赖 | 多机实验不便 |
| MLflow | 开源、可自建服务器 | 功能较少 |
| 手动记录 | 完全可控 | 效率低、易出错 |

Wandb 是目前最流行的实验跟踪工具之一，被 OpenAI、DeepMind 等团队使用。

## 总结

| 版本 | 核心变化 |
|------|----------|
| v11 | 超参数调优，模型规模扩大 |
| v12 | Wandb 实验跟踪，模型保存 |

v12 引入 Wandb，标志着 BabyGPT 从学习项目迈向工程实践：

- **可视化训练过程**：实时观察 loss 曲线、发现训练问题
- **实验对比**：快速找出最优超参数配置
- **云端同步**：实验记录永不丢失
- **模型保存**：训练成果可以复用

**至此，BabyGPT 的核心演进路线已完成**：

```
v1-v5:   核心组件（Embedding、Attention、FFN）
v6-v7:   架构优化（Block、残差连接）
v8-v10:  细节完善（投影层、LayerNorm、Dropout）
v11:     规模扩展（超参数调优）
v12:     工程实践（实验跟踪、模型保存）
```

从 v1 的 2000 参数到 v12 的 1280 万参数，从 8 token 上下文到 256 token 上下文，这是一条完整的 LLM 学习路径。

后续还可以探索：
- **KV Cache**：推理优化（`babygpt_sample_with_kvcache.py`）
- **分布式训练**：多 GPU 并行
- **更多数据**：扩大训练语料
- **更优架构**：RoPE、SwiGLU、Flash Attention
