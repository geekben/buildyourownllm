# Build Your Own LLM 🚀

从零开始，一步步构建一个类 GPT 的语言模型。使用宋词作为训练语料，从最简单的统计模型逐步演进到完整的 Transformer 架构，每个版本只引入一个新概念，帮助你真正理解 LLM 的核心原理。

## 快速开始

```bash
pip install -r requirements.txt
python babygpt_v11_hyper_params.py
```

训练完成后，模型会根据 prompt（如"春江"、"往事"）自动生成宋词风格的文本。

## 学习路线图

```
simplemodel.py                    # 最简单的 Bigram 统计模型
       ↓
simplebigrammodel.py              # 加入 Tokenizer 封装
       ↓
simplebigrammodel_torch.py        # PyTorch 张量重写
       ↓
pytorch_5min.py                   # PyTorch 梯度下降入门
       ↓
babygpt_v1.py                     # Embedding + Linear（神经网络起点）
       ↓
babygpt_v2.py ~ babygpt_v10.py    # 逐步添加 GPT 组件
       ↓
babygpt_v11_hyper_params.py       # 完整配置（推荐运行）
       ↓
babygpt_v12_wandb.py              # Wandb 实验跟踪
       ↓
babygpt_sample_with_kvcache.py    # KV Cache 推理优化
```

## 演进路线

项目按版本递进，每一步只引入**一个新概念**：

### 阶段一：统计模型

| 文件 | 说明 | 学习笔记 |
|------|------|----------|
| `simplemodel.py` | 最简单的 Bigram 统计模型，纯 Python 实现 | |
| `simplemodel_with_comments.py` | `simplemodel.py` 的详细注释版本 | `notes/random_choices_and_shuffle.md` |
| `simplebigrammodel.py` | 加入 Tokenizer 封装，结构更清晰 | |
| `simplebigrammodel_with_comments.py` | `simplebigrammodel.py` 的详细注释版本 | `notes/batch_size_and_block_size.md` |
| `simplebigrammodel_torch.py` | 用 PyTorch 张量重写统计模型 | `notes/pytorch_vs_python_list.md`<br>`notes/torch_clamp_multinomial.md`<br>`notes/experiment_simplebigrammodel_python_vs_torch.md` |
| `pytorch_5min.py` | PyTorch 梯度下降入门 | `notes/pytorch_training_mechanism.md` |

### 阶段二：神经网络模型（BabyGPT）

| 文件 | 引入的新概念 | 学习笔记 |
|------|-------------|----------|
| `babygpt_v1.py` | **Embedding + Linear**，从统计计数转向神经网络，引入梯度下降训练 | `notes/babygpt_v1_vs_simplebigrammodel.md`<br>`notes/estimate_loss_and_cross_entropy.md` |
| `babygpt_v2_position.py` | **Position Embedding**，让模型感知 token 的位置信息 | `notes/babygpt_v2_position_embedding.md` |
| `babygpt_v3_self_attention.py` | **Self-Attention**，token 之间可以互相"交流" | `notes/babygpt_v3_head_class_explained.md`<br>`notes/babygpt_v3_head_size_vs_n_embed.md`<br>`notes/babygpt_v3_self_attention_and_block_size.md` |
| `babygpt_v4_multihead_attention.py` | **Multi-Head Attention**，多个注意力头并行捕捉不同模式 | |
| `babygpt_v5_feedforward.py` | **Feed-Forward Network**，增加非线性变换能力 | |
| `babygpt_v6_block.py` | **Transformer Block**，将 Attention + FFN 封装为可堆叠的模块 | |
| `babygpt_v7_residual_connection.py` | **残差连接**，缓解深层网络的梯度消失问题 | |
| `babygpt_v8_projection.py` | **投影层**，Multi-Head 输出映射回原始维度 | |
| `babygpt_v9_layer_norm.py` | **Layer Normalization**，稳定训练过程 | |
| `babygpt_v10_dropout.py` | **Dropout**，正则化防止过拟合 | |
| `babygpt_v11_hyper_params.py` | **超参数调优**，6 层 6 头 384 维的完整配置 | `notes/experiment_babygpt_v11_on_T4_GPU.md` |
| `babygpt_v12_wandb.py` | **Wandb 集成**，可视化训练过程 | |

### 阶段三：推理优化

| 文件 | 说明 | 学习笔记 |
|------|------|----------|
| `babygpt_sample_with_kvcache.py` | **KV Cache** 推理优化，加载训练好的模型进行交互式生成 | |

## 最终模型架构

```
Input → Token Embedding + Position Embedding
      → Transformer Block × 6
          ├── Layer Norm → Multi-Head Attention (6 heads) → Residual + Dropout
          └── Layer Norm → Feed-Forward (384 → 1536 → 384) → Residual + Dropout
      → Layer Norm → Linear → Output Logits
```

## 学习笔记

`notes/` 目录包含学习过程中的笔记和实验记录：

**概念解析**
- `batch_size_and_block_size.md` - batch_size 与 block_size 概念解释
- `pytorch_training_mechanism.md` - PyTorch 训练机制详解
- `estimate_loss_and_cross_entropy.md` - estimate_loss 函数与交叉熵损失详解
- `random_choices_and_shuffle.md` - Python 随机采样函数对比

**模型演进**
- `babygpt_v1_vs_simplebigrammodel.md` - BabyGPT v1 与 Bigram 模型对比分析
- `babygpt_v2_position_embedding.md` - v2 位置编码原理详解
- `babygpt_v3_head_class_explained.md` - v3 Head 类详解：自注意力机制的实现
- `babygpt_v3_head_size_vs_n_embed.md` - head_size 与 n_embed 的关系
- `babygpt_v3_self_attention_and_block_size.md` - 自注意力与 block_size（上下文窗口）

**实验记录**
- `experiment_simplebigrammodel_python_vs_torch.md` - Python vs PyTorch 实现对比实验
- `experiment_babygpt_v11_on_T4_GPU.md` - T4 GPU 训练实验记录

**工具用法**
- `torch_clamp_multinomial.md` - torch.clamp 和 torch.multinomial 用法
- `pytorch_vs_python_list.md` - PyTorch 张量 vs Python 列表的优化细节

## 训练语料

- `ci.txt`：提取自 [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) 项目中的宋词和南唐词，经过格式化处理
- 使用字符级 Tokenizer（每个汉字/标点为一个 token）

## 依赖

- Python 3.8+
- PyTorch
- wandb（可选，仅 v12 使用）

## 参考资料

- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) - Andrej Karpathy
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Jay Alammar
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - 原始 Transformer 论文
