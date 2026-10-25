# AGENTS.md

> 给 AI 编码代理的项目速览。学习路线图见 `LEARNING_GUIDE.md`（HTML 版 `LEARNING_GUIDE.html`）。

## 项目是什么

MiniMind：一个从零可训练的极简大语言模型仓库，Qwen3 架构（RoPE / GQA / SwiGLU / MoE 可选），
打通完整链路：**预训练 → SFT → LoRA/DPO/PPO/GRPO/蒸馏 → 推理/评测**。
64M 参数（768 维 × 8 层）可从零训，也有 64M~300M 的公开权重（`minimind-3` 为 transformers 格式的成品权重）。

## 目录结构

```
model/          模型定义 model_minimind.py（LLaMA 式结构）、model_lora.py、tokenizer
trainer/        训练脚本：train_pretrain / train_full_sft / train_lora / train_dpo /
                train_ppo / train_grpo / train_distillation / train_agent、trainer_utils.py
dataset/        lm_dataset.py（Pretrain/SFT/DPO/RL 数据集类）、*.jsonl 数据（不入库）
scripts/        web_demo.py（聊天 WebUI）、train_ui.py（训练控制台，本仓库新增）、
                serve_openai_api.py / chat_api.py（OpenAI 兼容 API）、eval_toolcall.py
eval_llm.py     命令行对话 / 评测入口
out/            训练产出权重 out/{name}_{hidden}.pth（不入库）
minimind-3/     HF 格式推理权重（不入库）
```

## 环境

- 本机（Intel Mac，无 CUDA/MPS，纯 CPU）：`conda activate minimind`（Python 3.12，
  torch 2.5.1 用 conda CPU 版，其余 pip 装，清华镜像）
- `requirements.txt` 固定 `pyarrow==19.0.0`（Intel macOS 无新版 wheel，别升级）
- GPU 机器：直接 `pip install -r requirements.txt`，训练脚本 `--device` 默认自动检测 cuda

## 常用命令

```bash
# 推理（minimind-3 为 transformers 格式，用 --load_from；model/ 为原生 torch 格式走默认）
python eval_llm.py --load_from ./minimind-3

# 训练：必须在 trainer/ 目录下运行（脚本内相对路径为 ../dataset、../out、../model）
cd trainer
python train_pretrain.py --data_path ../dataset/pretrain_toy.jsonl \
  --hidden_size 256 --num_hidden_layers 4 --batch_size 8 --max_seq_len 128 \
  --epochs 1 --save_weight pretrain_toy --save_interval 200 --log_interval 20 --num_workers 2
python train_full_sft.py --data_path ../dataset/sft_toy.jsonl --from_weight pretrain_toy ...

# UI / 服务
streamlit run scripts/web_demo.py      # 聊天 WebUI，端口 8501（会扫描 scripts/ 下的模型目录）
streamlit run scripts/train_ui.py      # 训练控制台，端口 8502（本仓库新增）
python scripts/serve_openai_api.py --load_from ./minimind-3   # OpenAI 兼容 API，端口 8998
```

权重产物命名：`out/{save_weight}_{hidden_size}.pth`；下一阶段用 `--from_weight` 引用（如 SFT 需要 `pretrain_toy`）。
续训：`--from_resume 1` 会自动找 `out/{name}_{hidden}_resume.pth`。

## 数据

| 文件 | 规模 | 用途 |
|---|---|---|
| `dataset/pretrain_t2t_mini.jsonl` | 1.2G / 127 万条，字段 `text` | 预训练 |
| `dataset/sft_t2t_mini.jsonl` | 1.6G / 90 万条，字段 `conversations`（含 thinking、tool_calls） | SFT |
| `dataset/pretrain_toy.jsonl` / `sft_toy.jsonl` | 5000 / 3000 条子集 | 本机快速实验 |

下载：`modelscope download --dataset gongjy/minimind_dataset <file> --local_dir ./dataset`（断点续传，建议 `caffeinate -i` 前缀防休眠中断）。

## 本机（CPU）实测参考

| 配置 | 速度 |
|---|---|
| 64M（768×8）bs1 seq340 | 3.06 s/step |
| 64M（768×8）bs1 seq128 | 1.79 s/step |
| 玩具 256×4 bs8 seq128 | 1.02 s/step |
| 推理 | ~6 tok/s |

**官方全量数据（预训练 127 万条 × 2 epochs ≈ 254 万步）在 CPU 上需约 90 天，必须用 GPU。**
本机只能跑玩具子集（数千~2 万条）。`torch.set_num_threads()` 实测保持默认（=2）最快，设 4 反而更慢。

## 代理注意事项

1. 训练脚本只能在 `trainer/` 下执行，改路径要同步 `../` 前缀。
2. 新增/修改 `dataset/lm_dataset.py` 的 label mask 逻辑后，务必验证：prompt 段 label 应为 `-100`，只有 assistant 回答参与 loss。
3. Streamlit 服务跑在后台（8501 聊天、8502 训练台）；训练台日志落盘 `out/ui_train.log`，页面刷新不丢。
4. 下载物一律不入库：`minimind-3/`、`dataset/*.jsonl`、`out`、`checkpoints/` 已在 `.gitignore`。
5. 生成/重建 HTML 指导：改 `LEARNING_GUIDE.md` 后需跑一遍 md→html 脚本（保留中文锚点 slug）。
