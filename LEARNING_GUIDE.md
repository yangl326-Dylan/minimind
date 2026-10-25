# MiniMind 学习指导（本机实操版）

> 项目：`/Users/dylan/code/minimind`
> 机器：Intel Mac（i5-6287U，2 物理核 / 4 线程，16GB 内存，无 CUDA、无 MPS）
> 环境：conda `minimind`（Python 3.12.11 + torch 2.5.1 CPU）
> 已有：全部 Python 依赖、`./minimind-3` 推理权重、训练数据（`dataset/*.jsonl`）

这份指导不是 README 的复述，而是回答三个问题：

1. **代码该按什么顺序读？**（每块在做什么、对应哪个文件）
2. **在本机 CPU 上能做什么、不能做什么？**（含实测速度）
3. **怎样用最小成本把整条训练链路跑通？**（玩具模型路线）

---

## 目录

- [0. 一分钟看全局](#0-一分钟看全局)
- [1. 环境使用](#1-环境使用)
- [2. 第一站：推理](#2-第一站推理)
- [3. 代码地图与阅读顺序](#3-代码地图与阅读顺序)
- [4. 模型结构逐块拆解](#4-模型结构逐块拆解)
- [5. 数据与 loss mask](#5-数据与-loss-mask)
- [6. 训练主线：每个阶段在做什么](#6-训练主线每个阶段在做什么)
- [7. 本机 CPU 的现实与玩具模型路线](#7-本机-cpu-的现实与玩具模型路线)
- [8. 四周学习计划](#8-四周学习计划)
- [9. 十个可以动手的实验](#9-十个可以动手的实验)
- [10. 排错手册](#10-排错手册)
- [11. 命令速查表](#11-命令速查表)
- [12. 术语表](#12-术语表)

---

## 0. 一分钟看全局

MiniMind 是一条**从零训练小语言模型**的完整流水线，所有核心算法都用原生 PyTorch 实现：

```
预训练语料 {"text": ...}
        │
        ▼
   train_pretrain.py          # 下一 token 预测，学语言能力
        │  out/pretrain_768.pth
        ▼
   train_full_sft.py          # 对话格式 SFT，只对 assistant 回答算 loss
        │  out/full_sft_768.pth
        ├──────────────► train_lora.py      # 低成本领域适配（只存 LoRA 权重）
        ├──────────────► train_dpo.py       # 偏好对齐（chosen / rejected）
        ├──────────────► train_distillation.py  # 大模型教小模型（KL 蒸馏）
        └──────────────► train_ppo.py / train_grpo.py / train_agent.py  # RL
        │
        ▼
   scripts/convert_model.py   # 转成 HuggingFace/Qwen3 结构
        │  ./minimind-3
        ▼
   eval_llm.py / web_demo.py / serve_openai_api.py
```

核心代码量只有约 3500 行（不含 README），非常适合逐行读。

**模型规格（minimind-3）**

| 项目 | 值 |
|---|---|
| 参数量 | 63.9M（README 标称 64M） |
| hidden_size | 768 |
| 层数 | 8 |
| Q / KV heads | 8 / 4（GQA，2 个 Q head 共享 1 组 KV） |
| head_dim | 96 |
| FFN 中间维度 | 2432（SwiGLU） |
| 词表 | 6400 |
| 最大位置 | 32768 |
| RoPE theta | 1e6 |
| 词嵌入 | 与 lm_head 共享（tied） |
| MoE 版本 | 4 experts / top-1，198M-A64M |

---

## 1. 环境使用

```bash
conda activate minimind
cd /Users/dylan/code/minimind
python -c "import torch; print(torch.__version__)"   # 2.5.1
```

关键版本（已与 `requirements.txt` 对齐）：

| 包 | 版本 | 说明 |
|---|---|---|
| torch | 2.5.1 | conda 安装的 CPU 版；Intel macOS 上 pip 最高只有 2.2.2 |
| transformers | 4.57.6 | 模型加载、chat template |
| datasets | 3.6.0 | 读 JSONL |
| pyarrow | 19.0.0 | **手动固定**：Intel macOS 上 `pyarrow>=20` 没有 wheel，会触发 Rust 编译失败 |
| numpy | 1.26.4 | 项目 pin |
| trl | 0.13.0 | 项目锁定但实际训练脚本并未使用（算法全部原生实现） |

> ⚠️ **必须从 `trainer/` 目录运行训练脚本**。脚本内部大量使用 `../dataset/...`、`../out/...`、`../model` 这类相对路径。

```bash
cd trainer
python train_pretrain.py --help
```

---

## 2. 第一站：推理

先建立直觉：模型能做什么、速度如何、两种权重格式有什么区别。

### 2.1 两种权重格式

| 格式 | 路径 | 加载方式 | 用途 |
|---|---|---|---|
| 原生 MiniMind | `out/*.pth` | `--load_from model` + `--hidden_size/--num_hidden_layers` | **训练**、LoRA、续训 |
| HuggingFace/Qwen3 | `./minimind-3` | `AutoModelForCausalLM.from_pretrained` | **推理**、部署、生态兼容 |

`./minimind-3` 是 `scripts/convert_model.py` 转换出来的，`config.json` 里写的是 `Qwen3ForCausalLM`。它**不能**直接用于仓库内的训练脚本；训练要用原生 `.pth`。

### 2.2 跑一次推理

```bash
conda activate minimind
cd /Users/dylan/code/minimind

# 自动测试 8 个问题
python eval_llm.py --load_from ./minimind-3

# 手动对话
python eval_llm.py --load_from ./minimind-3
# 输入 1 进入手动模式
```

本机实测：约 **5.5–6.4 tokens/s**（CPU，fp16 推理）。
短回答几秒，长回答要等几十秒，这是正常的。

常用参数：

```bash
python eval_llm.py --load_from ./minimind-3 \
  --max_new_tokens 256 \
  --temperature 0.85 \
  --top_p 0.95 \
  --open_thinking 1        # 开启  thinking 输出
```

### 2.3 Web UI 与 API

```bash
# Web UI：需要先把 HF 模型目录放进 scripts/
cp -r minimind-3 scripts/
cd scripts && streamlit run web_demo.py

# OpenAI 兼容 API（端口 8998）
cd /Users/dylan/code/minimind
python scripts/serve_openai_api.py --load_from ./minimind-3
```

### 2.4 下载官方原生权重（可选，用于微调）

```bash
modelscope download --model gongjy/minimind-3-pytorch \
  full_sft_768.pth --local_dir ./out

# 用原生权重推理
python eval_llm.py --load_from model --weight full_sft
```

首发权重文件对照：

| 文件 | 用途 |
|---|---|
| `pretrain_768.pth` | 预训练产物，SFT 的起点 |
| `full_sft_768.pth` | 对话 SFT 产物，最常用的基座 |
| `dpo_768.pth` / `grpo_768.pth` / `ppo_actor_768.pth` | 各对齐阶段产物 |
| `lora_identity_768.pth` / `lora_medical_768.pth` / `lora_exam_768.pth` | LoRA 增量权重（约 0.8MB） |
| `*_768_moe.pth` | MoE 版本对应权重 |

---

## 3. 代码地图与阅读顺序

按下面的顺序读，每一步都建立在前一步之上：

| 顺序 | 文件 | 行数 | 读什么 |
|---|---|---|---|
| 1 | `model/model_minimind.py` | 292 | 模型结构、RoPE、GQA、MoE、loss、generate |
| 2 | `dataset/lm_dataset.py` | 260 | 四种数据集、chat template、label mask |
| 3 | `trainer/trainer_utils.py` | 209 | 学习率、checkpoint、初始化、参数统计 |
| 4 | `trainer/train_pretrain.py` | 163 | 最干净的训练循环，先看它 |
| 5 | `trainer/train_full_sft.py` | 164 | 和预训练几乎相同，差别在数据集 |
| 6 | `model/model_lora.py` + `trainer/train_lora.py` | 65 + 176 | LoRA 实现与冻结策略 |
| 7 | `trainer/train_dpo.py` | 218 | 参考模型、DPO loss |
| 8 | `trainer/train_distillation.py` | 238 | CE + KL 混合损失 |
| 9 | `trainer/train_grpo.py` / `train_ppo.py` / `train_agent.py` | 326 / 447 / 485 | rollout、奖励、策略更新 |
| 10 | `scripts/convert_model.py`、`serve_openai_api.py` | 144 / 252 | 格式转换与部署 |

建议边读边回答：

- `train_pretrain.py` 里哪一行真正更新了参数？
- `loss = res.loss + res.aux_loss` 中的 `aux_loss` 什么时候不为 0？
- 为什么 SFT 的 loss 只算 assistant 部分？代码在哪里实现？
- `lm_checkpoint` 保存了什么？`--from_resume 1` 恢复的是什么？
- `generate()` 里 KV cache 是怎样避免重复计算的？

---

## 4. 模型结构逐块拆解

文件：`model/model_minimind.py`

### 4.1 RMSNorm

```python
x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
```

比 LayerNorm 少去均值、少 bias，计算更省。注意 `forward` 里先转 `float()` 再转回原 dtype，是 fp16 数值稳定性的常见写法。

### 4.2 RoPE 与 YaRN

`precompute_freqs_cis()` 预计算每个位置的 cos/sin：

- 普通 RoPE：`freqs = 1 / (theta ** (arange(0, dim, 2) / dim))`
- `inference_rope_scaling=True` 时启用 YaRN，把频率按 `beta_fast/beta_slow` 分段拉伸，用于超出训练长度外推（默认 4 倍）

`apply_rotary_pos_emb()` 对 q/k 做旋转，q 和 k 用同一个位置编码。

### 4.3 GQA Attention

```python
q_proj: 768 → 8 × 96
k_proj: 768 → 4 × 96
v_proj: 768 → 4 × 96
repeat_kv(k, n_rep=2)
```

4 组 KV 被复制 2 份，和 8 个 Q head 对齐。好处：KV cache 显存减半，推理更快。

细节：

- `q_norm` / `k_norm`：对每个 head 做 RMSNorm（QK-Norm），提升训练稳定性。
- 训练且无 cache 时走 `F.scaled_dot_product_attention`；否则手写 causal mask + softmax。
- `past_key_value` 拼接实现 KV cache，`generate()` 每次只输入新 token。

### 4.4 SwiGLU FFN

```python
down_proj( silu(gate_proj(x)) * up_proj(x) )
```

三个线性层，中间维度 2432。`intermediate_size` 默认由 `ceil(hidden * π / 64) * 64` 推出。

### 4.5 MoE（可选）

`MOEFeedForward`：

1. gate 输出每个 token 对 4 个专家的分数；
2. 取 top-1；
3. 只把 token 送给命中的专家；
4. 训练时计算负载均衡辅助损失 `aux_loss`；
5. `num_experts_per_tok=1` 时用 straight-through 让路由权重在前向为 1.0、梯度仍可回传。

注意实现是 `for` 循环遍历专家，**教学清晰但速度慢**，这也是 README 说 MoE 实际训练反而更重的原因之一。

### 4.6 Block 与残差

Pre-Norm 结构：

```
x = x + Attention(RMSNorm(x))
x = x + FFN(RMSNorm(x))
```

### 4.7 输出层与 loss

```python
logits = lm_head(hidden_states)
loss = cross_entropy(logits[..., :-1, :], labels[..., 1:], ignore_index=-100)
```

- 输入第 `t` 个 token，预测第 `t+1` 个 token；
- `tie_word_embeddings=True` 时 `lm_head.weight` 与 `embed_tokens.weight` 是同一块参数（省 4.9M 参数）；
- `ignore_index=-100` 让 padding / prompt 部分不产生梯度。

### 4.8 手写 generate

`MiniMindForCausalLM.generate()` 没调 `transformers` 的 generate，而是自己实现：

- temperature 缩放 logits；
- top-k 截断；
- top-p（nucleus）累积概率截断；
- repetition penalty；
- KV cache 增量解码；
- `TextStreamer` 流式输出；
- 遇到 eos 提前结束。

读这个方法可以完整理解采样策略。

### 4.9 参数量怎么来的

以 64M 为例（tied embedding 只算一次）：

| 部分 | 计算 | 参数量 |
|---|---|---|
| 词嵌入 | 6400 × 768 | 4.92M |
| 每层 Attention | q/o: 2×768×768；k/v: 2×768×384 | ≈1.77M |
| 每层 FFN | 3 × 768 × 2432 | ≈5.61M |
| 每层小计 | Attention + FFN | ≈7.38M |
| 8 层 | 7.38M × 8 | ≈59.0M |
| 合计 | 59.0M + 4.9M | **≈63.9M** |

可以用 `get_model_params()` 打印验证。MoE 版本把每层 FFN 换成 4 个专家，总参数 198M，但每 token 只激活 1 个专家，所以标成 `198M-A64M`。

---

## 5. 数据与 loss mask

文件：`dataset/lm_dataset.py`

### 5.1 预训练数据 `PretrainDataset`

JSONL 每行：`{"text": "..."}`

处理：

1. tokenize，截断到 `max_length - 2`；
2. 首尾加 bos/eos；
3. pad 到固定长度；
4. labels 复制 input_ids，pad 位置设为 `-100`。

**没有做样本拼接（packing）**，每条样本独立 padding，所以短样本会浪费算力。

### 5.2 SFT 数据 `SFTDataset`

JSONL 每行：`{"conversations": [{"role": ..., "content": ...}, ...]}`

关键点：**只对 assistant 的回答算 loss**。

实现方式：

1. `apply_chat_template` 生成完整对话文本；
2. 找到 `<|im_start|>assistant\n` 的位置，之后的内容才标为有效 label；
3. 用户问题、system prompt 全部是 `-100`。

为什么要 mask？

- 模型需要学的是「给定上下文如何回答」，不是「如何提问」；
- 不 mask 会让模型也学会复述用户输入，且 loss 被大量 prompt token 稀释。

另外两个工程细节：

- `pre_processing_chat`：以 20% 概率给没有 system 的样本加系统提示，增加鲁棒性；
- `post_processing_chat`：以 80% 概率删除空的 ` thinking\n\n</think>`，避免模型学到无意义思考标签。

### 5.3 DPO 数据 `DPODataset`

每行包含 `chosen` 和 `rejected` 两段完整对话。

- 两段都做 chat template；
- 各自生成 loss mask（只算 assistant）；
- 返回 `x_chosen/y_chosen/mask_chosen` 和 rejected 版本；
- 训练时 batch 里前一半是 chosen、后一半是 rejected（`dpo_loss` 依赖这个顺序）。

### 5.4 RLAIF / Agent 数据

- `RLAIFDataset`：只返回 prompt，答案由模型在线采样，再用奖励模型打分；
- `AgentRLDataset`：返回 `messages`、`tools`、`gt`，用于多步工具调用任务。

---

## 6. 训练主线：每个阶段在做什么

> 下表默认是 **minimind-3 / 64M / 单卡 3090** 的官方配置与耗时（README 数据），用于建立量级概念。

| 阶段 | 脚本 | 数据 | 关键默认参数 | 产物 | 3090 单 epoch |
|---|---|---|---|---|---|
| 预训练 | `train_pretrain.py` | `pretrain_t2t_mini.jsonl` | seq 340, bs 32, accum 8, lr 5e-4, ep 2 | `out/pretrain_768.pth` | ≈1.21h |
| 全参 SFT | `train_full_sft.py` | `sft_t2t_mini.jsonl` | seq 768, bs 16, lr 1e-5, ep 2 | `out/full_sft_768.pth` | ≈1.10h |
| LoRA | `train_lora.py` | `lora_medical.jsonl` 等 | seq 340, bs 32, lr 1e-4, ep 10 | `out/lora_*_768.pth` | ≈0.9h |
| DPO | `train_dpo.py` | `dpo.jsonl` | seq 1024, bs 4, lr 4e-8, β 0.15, ep 1 | `out/dpo_768.pth` | ≈1.1h |
| 蒸馏 | `train_distillation.py` | `sft_t2t_mini.jsonl` | α 0.5, T 1.5, ep 6 | `out/full_dist_768.pth` | — |
| PPO | `train_ppo.py` | `rlaif.jsonl` + reward model | lr 3e-7, clip 0.2, KL 0.02 | `out/ppo_actor_768.pth` | — |
| GRPO | `train_grpo.py` | `rlaif.jsonl` + reward model | 每 prompt 采 6 条, β 0.1 | `out/grpo_768.pth` | — |
| Agent RL | `train_agent.py` | `agent_rl.jsonl` | 多步工具调用 + 规则奖励 | `out/agent_768.pth` | — |

### 6.1 预训练

```bash
cd trainer
python train_pretrain.py
```

它做的是最标准的语言建模：给一段文本，预测下一个 token。

观察点：

- `loss` 应从约 `ln(6400) ≈ 8.76` 开始下降；
- `logits_loss` 是主损失，`aux_loss` 只有 MoE 才非 0；
- 学习率用余弦衰减：`lr * (0.1 + 0.45 * (1 + cos(π * step / total)))`；
- 每 `save_interval` 步保存 `out/pretrain_768.pth`（fp16）；
- `checkpoints/pretrain_768_resume.pth` 保存模型 + 优化器 + scaler + step，用于断点续训。

### 6.2 全参 SFT

```bash
python train_full_sft.py --from_weight pretrain
```

和预训练代码几乎一样，区别：

- 数据集换成 `SFTDataset`；
- 默认 lr 从 5e-4 降到 1e-5（避免破坏预训练知识）；
- `from_weight=pretrain` 加载上一步权重。

产出模型才能按对话格式回答。

### 6.3 LoRA

```bash
python train_lora.py --lora_name lora_medical --data_path ../dataset/lora_medical.jsonl
```

实现要点（`model/model_lora.py`）：

- 只给 **in_features == out_features 的 Linear** 加 LoRA；
- `B` 初始化为 0，保证训练开始时输出不变；
- rank 默认 16；
- 只保存 LoRA 的 A/B 权重（约 0.8MB），基座不动；
- 合并公式：`W' = W + B @ A`，`merge_lora()` 已实现。

推理时加载：

```bash
python eval_llm.py --load_from model --weight full_sft --lora_weight lora_medical
```

适合：领域知识注入（医疗、考试、身份），不想全参微调时。

### 6.4 DPO

```bash
python train_dpo.py --from_weight full_sft --beta 0.15
```

DPO 不需要单独训练奖励模型，直接用偏好对：

```
loss = -logsigmoid( β * [ (logπ(chosen) - logπ(rejected))
                        - (logπ_ref(chosen) - logπ_ref(rejected)) ] )
```

代码里初始化了两个模型：

- `model`：策略模型，参与训练；
- `ref_model`：参考模型，冻结，只算 logprob。

`beta` 越大越保守（更贴近参考模型），越小越敢偏离。lr 默认只有 `4e-8`，因为 DPO 很容易训崩。

### 6.5 蒸馏

```bash
python train_distillation.py
```

总损失：

```
loss = α * CE(student, labels) + (1 - α) * KL(student/T || teacher/T) * T²
```

- 教师默认是 MoE（`teacher_use_moe=1`），学生是 Dense；
- `alpha=0.5` 表示一半学硬标签、一半学教师的软分布；
- 温度 `T=1.5` 让软分布更平滑，携带更多“暗知识”。

### 6.6 RL：PPO / GRPO / Agent

这三个脚本都需要**奖励模型**：

```bash
--reward_model_path ../../internlm2-1_8b-reward
```

即 InternLM2-1.8B Reward，需要单独下载。流程：

1. 策略模型对 prompt 采样多条回答（rollout）；
2. 奖励模型 / 规则函数打分；
3. PPO：用 critic 估计优势，做 clip 更新；
4. GRPO：同一 prompt 的多条回答组内归一化得到优势，不需要 critic；
5. Agent：多步工具调用，按最终结果和工具执行情况给奖励。

GRPO 还包含一些工程奖励整形（长度约束、重复惩罚、` thinking` 格式），见 `calculate_rewards()`。

> 本机 16GB CPU **不建议实际跑 RL**：仅奖励模型就 1.8B，且每条样本要多次生成。把这三份代码当阅读理解即可。

### 6.7 转 HF 格式与部署

```bash
cd scripts
python convert_model.py
```

脚本默认读取 `../out/full_sft_768.pth`，输出 `../minimind-3`（Qwen3 结构）。转换后可以用 `transformers`、vLLM、ollama 等生态工具。

---

## 7. 本机 CPU 的现实与玩具模型路线

### 7.1 实测速度（本机，torch 默认 2 线程）

| 配置 | 每步耗时 |
|---|---|
| 64M，batch 1，seq 128 | **1.79 s/step** |
| 64M，batch 1，seq 340 | **3.06 s/step** |
| 玩具 5M，batch 8，seq 128 | **1.02 s/step** |
| 64M 推理 | **≈6 tokens/s** |

补充：本机 `torch.get_num_threads()` 默认是 2（按物理核数），**不要手动改成 4**——我实测改成 4 线程反而从 3.06s 降到 3.93s/step（超线程争抢）。

### 7.2 为什么不能按默认配置跑完整训练

`pretrain_t2t_mini.jsonl` 有 1.18GB。以 batch 32 计算，一个 epoch 约几万步；即使按最乐观的 20–30 s/step 估计，也要**数十天**。SFT 数据 1.66GB，同理。

所以本机的定位是：

- ✅ 推理、读代码、跑通流程
- ✅ 用**子集 + 小模型**做实验
- ❌ 复现官方 64M 完整训练

### 7.3 玩具模型路线（推荐）

思路：把模型缩到 5M、数据缩到几千条，让一次完整训练在 10–30 分钟内结束。

**第一步：造子集**（在项目根目录）

```bash
head -n 5000 dataset/pretrain_t2t_mini.jsonl > dataset/pretrain_toy.jsonl
head -n 3000 dataset/sft_t2t_mini.jsonl    > dataset/sft_toy.jsonl
wc -l dataset/pretrain_toy.jsonl dataset/sft_toy.jsonl
```

**第二步：训练玩具预训练模型**

```bash
cd trainer
python train_pretrain.py \
  --data_path ../dataset/pretrain_toy.jsonl \
  --hidden_size 256 --num_hidden_layers 4 \
  --batch_size 8 --max_seq_len 128 --epochs 1 \
  --save_weight pretrain_toy \
  --save_interval 200 --log_interval 20 \
  --num_workers 2
```

预期：约 625 步，**10–12 分钟**跑完，产出 `../out/pretrain_toy_256.pth`。
loss 会从 8.7 左右下降，但模型仍然不会说人话——它只学了 5000 条样本、1 个 epoch，这是正常的。

**第三步：玩具 SFT**

```bash
python train_full_sft.py \
  --data_path ../dataset/sft_toy.jsonl \
  --hidden_size 256 --num_hidden_layers 4 \
  --batch_size 4 --max_seq_len 256 --epochs 1 \
  --from_weight pretrain_toy --save_weight full_sft_toy \
  --save_interval 200 --log_interval 20 \
  --num_workers 2
```

约 750 步，20–30 分钟，产出 `../out/full_sft_toy_256.pth`。

**第四步：验证**

```bash
cd /Users/dylan/code/minimind
python eval_llm.py --load_from model \
  --weight full_sft_toy \
  --hidden_size 256 --num_hidden_layers 4 \
  --max_new_tokens 64
```

你会看到一个“会套对话格式、但内容很弱”的模型。**重点不是效果，而是整条链路（数据 → tokenizer → 训练 → 保存 → 加载 → 推理）全部跑通。**

**第五步（可选）：玩具 LoRA / DPO**

```bash
cd trainer

# LoRA：先造 1000 条医疗子集
head -n 1000 ../dataset/lora_medical.jsonl > ../dataset/lora_toy.jsonl
python train_lora.py \
  --data_path ../dataset/lora_toy.jsonl \
  --lora_name lora_toy \
  --hidden_size 256 --num_hidden_layers 4 \
  --batch_size 4 --max_seq_len 192 --epochs 3 \
  --from_weight full_sft_toy --save_interval 100 --log_interval 10

# DPO：先造 500 条偏好对
head -n 500 ../dataset/dpo.jsonl > ../dataset/dpo_toy.jsonl
python train_dpo.py \
  --data_path ../dataset/dpo_toy.jsonl \
  --hidden_size 256 --num_hidden_layers 4 \
  --batch_size 1 --max_seq_len 512 --epochs 1 \
  --from_weight full_sft_toy --save_weight dpo_toy \
  --save_interval 100 --log_interval 10
```

### 7.4 玩具路线的三个注意点

1. **文件名编码了 hidden_size**：`pretrain_toy_256.pth`。换模型宽度时记得同步改 `--hidden_size`，否则加载会 size mismatch。
2. **`--save_weight` 是输出名，`--from_weight` 是输入名**。两者不要混。
3. **Mac 上 `--num_workers` 不要开太大**：2 就够了，设 8 会因进程启动开销变慢。

---

## 8. 四周学习计划

### 第 1 周：推理 + 模型结构

目标：能不看代码画出 MiniMind 的结构图，并解释每个模块。

- [ ] 跑通 `eval_llm.py` 自动测试与手动对话
- [ ] 启动一次 Web UI 或 OpenAI API
- [ ] 精读 `model/model_minimind.py`，手画 Block 图
- [ ] 手算 64M 参数量，与 `get_model_params()` 输出对照
- [ ] 改 `temperature` / `top_p` / `repetition_penalty`，观察输出差异
- [ ] 用 `tokenizer.apply_chat_template()` 打印一条完整对话文本，找到 assistant 边界

自检问题：

- 为什么 KV heads 比 Q heads 少？省了什么？
- RoPE 为什么不用可学习位置编码？
- `tie_word_embeddings` 省了多少参数？

### 第 2 周：数据与预训练

目标：理解数据格式、loss mask，跑通玩具预训练。

- [ ] 各取 3 行 `pretrain_t2t_mini.jsonl` / `sft_t2t_mini.jsonl` 对照格式
- [ ] 跑 `train_tokenizer.py` 的 `get_texts()`，理解 BPE 语料来源（不要覆盖 `model/`）
- [ ] 造子集，跑玩具预训练（第 7.3 节）
- [ ] 观察 loss 曲线，解释为什么起点约为 `ln(6400)`
- [ ] 手动中断训练，用 `--from_resume 1` 续训，确认 step 恢复
- [ ] 读 `lm_checkpoint()`，说清 `*_resume.pth` 里有什么

### 第 3 周：SFT、LoRA、DPO

目标：理解三种对齐/微调方式的区别。

- [ ] 跑玩具 SFT，用 `eval_llm.py` 对话
- [ ] 打印 `SFTDataset.generate_labels()` 的结果，标出哪些 token 是 `-100`
- [ ] 跑玩具 LoRA，对比加载前后输出
- [ ] 用 `merge_lora()` 合并权重，验证合并后不加载 LoRA 也能复现效果
- [ ] 跑玩具 DPO，观察 `dpo_loss` 是否下降
- [ ] 对比全参 SFT / LoRA / DPO 的显存、参数量、数据需求

### 第 4 周：蒸馏、RL、部署与实验

目标：读懂高级算法，完成一次格式转换和部署。

- [ ] 读 `train_distillation.py`，手推 `α*CE + (1-α)*KL*T²`
- [ ] 读 `train_grpo.py` 的 `calculate_rewards()`，列出所有奖励项
- [ ] 读 `train_agent.py`，理解多步工具调用如何进入训练
- [ ] 把玩具模型转成 HF 格式（修改 `convert_model.py` 的路径与 hidden_size）
- [ ] 用 `serve_openai_api.py` 起服务，用 curl 调一次
- [ ] 从第 9 节挑 2 个实验，写一页结论

---

## 9. 十个可以动手的实验

每个实验都建议：**只改一个变量 → 固定种子 → 跑玩具规模 → 记录 loss/速度/输出**。

### 实验 1：GQA 到底省了什么

把 `num_key_value_heads` 改成 8（等价 MHA）、4（默认）、2、1，比较参数量与推理速度。

- 预期：KV cache 随 KV heads 线性减小；参数量减少主要来自 k/v 投影。
- 观察点：`k_proj/v_proj` 的输出维度。

### 实验 2：RoPE theta

把 `rope_theta` 从 `1e6` 改成 `1e4`，重新训练玩具模型。

- 预期：小模型 + 短序列差异不明显；长序列外推时 1e6 更好。
- 进阶：打开 `inference_rope_scaling`，比较 2048 长度下的困惑度。

### 实验 3：tied embedding

把 `tie_word_embeddings` 设为 `False`，打印参数量。

- 预期：多出 `6400 × 768 ≈ 4.9M`。
- 思考：小模型共享嵌入的利弊。

### 实验 4：SFT 的 label mask

临时把 `generate_labels()` 改成“全部 token 都算 loss”，训练 1 个 epoch。

- 预期：loss 更低，但模型更容易复述问题，回答质量下降。
- 这是理解“loss 低 ≠ 效果好”的最好例子。

### 实验 5：LoRA rank

把 `apply_lora(rank=16)` 改成 4 / 16 / 64。

- 观察 LoRA 参数量占比（训练脚本会打印）。
- 预期：rank 越大拟合能力越强，但更容易过拟合小数据集。

### 实验 6：DPO beta

在玩具 DPO 上试 `beta=0.05 / 0.15 / 0.5`。

- 预期：beta 小 → 偏离参考模型多，风险高；beta 大 → 更新保守。
- 观察 `dpo_loss` 与生成风格。

### 实验 7：采样策略

固定 prompt，比较：

- `temperature=0.3` vs `0.85` vs `1.2`
- `top_p=0.5` vs `0.95`
- `top_k=1`（贪心）vs `50`

记录多样性、重复率、事实性。

### 实验 8：MoE vs Dense（玩具规模）

用 `--use_moe 1` 训练玩具 MoE（4 experts），对比：

- 总参数量 / 激活参数量；
- 每步耗时（预计明显变慢）；
- `aux_loss` 是否收敛。

### 实验 9：序列长度与吞吐

玩具模型下比较 `max_seq_len=128 / 256 / 340` 的 s/step 与最终 loss。

- 预期：耗时近似线性增长；短序列更快但长依赖学不到。

### 实验 10：蒸馏温度

`temperature=1.0 / 1.5 / 2.0`，`alpha=0.2 / 0.5 / 0.8`。

- 预期：温度越高软标签越平滑；alpha 越高越依赖真实标签。
- 观察：学生模型在验证集上的表现与输出多样性。

---

## 10. 排错手册

| 现象 | 原因 | 解决 |
|---|---|---|
| `ModuleNotFoundError: datasets` | 没激活环境 | `conda activate minimind` |
| `FileNotFoundError: ../out/pretrain_768.pth` | 从错误目录运行 / 权重名不对 | `cd trainer`；检查 `--from_weight` |
| `size mismatch for ...` | `--hidden_size` 与权重文件不匹配 | 文件名里的数字就是 hidden_size |
| `RuntimeError: MPS backend` / CUDA 报错 | 本机无 GPU | 显式 `--device cpu` |
| `MemoryError` / 系统卡死 | batch/seq 太大 | bs 降到 1，seq 降到 128，用梯度累积 |
| loss 变 NaN | lr 太大 / fp16 溢出 | 降 lr；CPU 上默认 fp32，不要设 `--dtype float16` |
| 训练极慢 | CPU 2 核，正常 | 用子集 + 小模型；不要开 `--use_compile` |
| `use_wandb` 报登录错误 | swanlab 未登录 | 去掉 `--use_wandb` |
| `pyarrow` 编译报 Rust 错误 | Intel macOS 无新版 wheel | 已固定 `pyarrow==19.0.0` |
| tokenizer 加载失败 | `./model/tokenizer.json` 缺失 | 确认在项目根目录运行推理 |
| `--from_resume 1` 没恢复 | `checkpoints/*_resume.pth` 不存在 | 先正常训练并触发一次 save |
| HF 模型下载慢/失败 | 网络 | 用 `modelscope download` |

补充：

- 训练脚本的 `--save_interval` 同时也是 resume checkpoint 的保存间隔；
- `--use_compile 1` 在 CPU 上收益有限，LoRA 还会自动关闭；
- `train_lora.py` 的 `--epochs` 默认 10，小数据集很容易过拟合，建议 1–3。

---

## 11. 命令速查表

```bash
# ===== 环境 =====
conda activate minimind
cd /Users/dylan/code/minimind

# ===== 推理 =====
python eval_llm.py --load_from ./minimind-3
python eval_llm.py --load_from model --weight full_sft
python eval_llm.py --load_from model --weight full_sft --lora_weight lora_medical
python scripts/serve_openai_api.py --load_from ./minimind-3

# ===== 下载 =====
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3
modelscope download --model gongjy/minimind-3-pytorch full_sft_768.pth --local_dir ./out
modelscope download --dataset gongjy/minimind_dataset pretrain_t2t_mini.jsonl sft_t2t_mini.jsonl --local_dir ./dataset

# ===== 正式训练（需 GPU，从 trainer/ 运行）=====
cd trainer
python train_pretrain.py
python train_full_sft.py --from_weight pretrain
python train_lora.py --lora_name lora_medical
python train_dpo.py --from_weight full_sft
python train_distillation.py
python train_ppo.py    # 需 reward model
python train_grpo.py   # 需 reward model
python train_agent.py  # 需 reward model + 工具环境

# ===== 玩具训练（本机 CPU）=====
python scripts/train_ui.py   # 训练控制台 WebUI: http://localhost:8502

python train_pretrain.py --data_path ../dataset/pretrain_toy.jsonl \
  --hidden_size 256 --num_hidden_layers 4 \
  --batch_size 8 --max_seq_len 128 --epochs 1 \
  --save_weight pretrain_toy --save_interval 200 --log_interval 20 --num_workers 2

python train_full_sft.py --data_path ../dataset/sft_toy.jsonl \
  --hidden_size 256 --num_hidden_layers 4 \
  --batch_size 4 --max_seq_len 256 --epochs 1 \
  --from_weight pretrain_toy --save_weight full_sft_toy \
  --save_interval 200 --log_interval 20 --num_workers 2

# ===== 转换 =====
cd ../scripts && python convert_model.py
```

---

## 12. 术语表

| 术语 | 含义 | 最直白的一句话 | 在本项目中的位置 |
|---|---|---|---|
| Pretrain | 下一 token 预测，学通用语言能力 | 让模型疯狂做"接下半句"填空题，先把语言本身学会 | `train_pretrain.py` |
| SFT | Supervised Fine-Tuning，学对话格式 | 拿标准问答示范教模型"别人问你我该怎么答" | `train_full_sft.py` |
| Loss mask | 只对部分 token 计算损失 | 题目不计分、答案才计分——只惩罚模型该说却说错的部分 | `SFTDataset.generate_labels()` |
| LoRA | 低秩增量微调，只训练 A/B 矩阵 | 不改原书内容，只在页边贴小抄条，训练完贴上去就是新书 | `model/model_lora.py` |
| DPO | 直接用偏好对优化策略，无需奖励模型 | 摆出好答案和坏答案，让模型学会"更偏爱好的那个" | `train_dpo.py` |
| PPO | 带 critic 的策略梯度 | 边写边打分：写得好就加强这种写法，写得差就抑制 | `train_ppo.py` |
| GRPO | 组内相对优势，无需 critic | 同一道题让模型答 6 遍，组内比一比，答得最好的加强 | `train_grpo.py` |
| CISPO | 一种更稳定的策略优化变体，本项目可选 | GRPO 的改良 loss，让更新步子更稳、不容易训崩 | `--loss_type cispo` |
| Distillation | 用教师模型的软分布教学生 | 大老师把"每个词的可能性打分表"抄给小学生，而不只抄答案 | `train_distillation.py` |
| RoPE | 旋转位置编码 | 给每个词盖一个"位置邮戳"，模型靠角度差知道谁在前谁在后 | `precompute_freqs_cis()` |
| YaRN | RoPE 的长度外推方案 | 只读过短文的模型也能读长文的"望远镜"技巧 | `inference_rope_scaling` |
| GQA | Grouped Query Attention，多个 Q 共享 KV | 好几个提问头共用一本笔记，省显存又不怎么掉效果 | `repeat_kv()` |
| KV cache | 缓存历史 K/V，避免重复计算 | 把已经想过的内容记在小本上，别每蹦一个字都从头背一遍 | `generate()` |
| MoE | 混合专家，每 token 只激活部分专家 | 公司里有很多专家，每个问题只请一位专家上手干活 | `MOEFeedForward` |
| SwiGLU | `silu(gate(x)) * up(x)` 的 FFN | 一种"先开闸再放水"的全连接层写法，效果更好还省算力 | `FeedForward` |
| aux loss | MoE 负载均衡辅助损失 | 防止所有问题都挤去找同一位专家的"分流罚款" | `MOEFeedForward.forward()` |
| rollout | RL 中让模型现场生成样本 | 先让模型真刀真枪写一遍答案，再看写得怎么样来打分 | `train_grpo.py` |
| temperature | 采样温度，越大越随机 | 掷骰子的"疯狂程度"：0 就是死板背答案，1 以上开始放飞 | `generate()` |
| top-p | 核采样，保留累积概率前 p 的 token | 只从"最有戏的候选词"里抽签，防止蹦出离谱的词 | `generate()` |
| packing | 把多条短样本拼进一个序列 | 短样本拼车出行，一车装满不浪费座位（本项目没用） | 本项目未使用 |

---

## 附录：产物与文件对照

| 路径 | 内容 |
|---|---|
| `model/tokenizer.json` | 6400 词表 BPE tokenizer（不要覆盖） |
| `model/tokenizer_config.json` | chat template 与特殊 token 配置 |
| `dataset/*.jsonl` | 训练数据（已加入 `.gitignore`） |
| `out/pretrain_768.pth` | 预训练权重 |
| `out/full_sft_768.pth` | SFT 权重 |
| `out/lora_*_768.pth` | LoRA 增量权重（约 0.8MB） |
| `out/dpo_768.pth` / `grpo_768.pth` / `ppo_actor_768.pth` | 对齐阶段权重 |
| `checkpoints/*_resume.pth` | 断点续训状态（模型 + 优化器 + step） |
| `minimind-3/` | HF/Qwen3 格式权重（已加入 `.gitignore`） |
| `scripts/minimind-3/` | Web UI 扫描的模型目录（需要时手动复制） |

---

> 学习建议：**每读一个模块，就改一个参数、跑一次最小实验、记录一个结论。**
> 在这个项目里，跑通 5M 玩具模型的完整链路，比在纸上读完 64M 的全部代码更有价值。
