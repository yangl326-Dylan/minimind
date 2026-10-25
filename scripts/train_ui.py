"""
MiniMind 本地训练控制台（简易 WebUI）

启动:
    conda activate minimind
    python scripts/train_ui.py          # 或 streamlit run scripts/train_ui.py
浏览器访问 http://localhost:8502

功能: 选择训练脚本与参数 → 启动/停止 → 实时日志 + loss/lr 曲线。
日志写入 ../out/ui_train.log，页面刷新/重开后仍可回看上次训练记录。
"""
import os
import re
import sys
import time
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parent.parent
TRAINER = REPO / "trainer"
DATASET = REPO / "dataset"
OUT_DIR = REPO / "out"
LOG_PATH = OUT_DIR / "ui_train.log"

st.set_page_config(page_title="MiniMind 训练台", page_icon="🛠️", layout="wide")

# 支持的脚本与玩具预设参数（save=输出权重名，From=输入权重名，lr=None 表示用脚本默认）
SCRIPTS = {
    "预训练 pretrain": dict(
        file="train_pretrain.py", flag_save="--save_weight", save="pretrain_toy",
        From=None, data="pretrain_toy.jsonl", epochs=1, bs=8, seq=128, lr=None,
        hint="从零学语言，产出可被 SFT 使用的底座权重"),
    "全参SFT full_sft": dict(
        file="train_full_sft.py", flag_save="--save_weight", save="full_sft_toy",
        From="pretrain_toy", data="sft_toy.jsonl", epochs=1, bs=4, seq=256, lr=None,
        hint="对话格式微调，需先有预训练权重"),
    "LoRA": dict(
        file="train_lora.py", flag_save="--lora_name", save="lora_toy",
        From="full_sft_toy", data="lora_toy.jsonl", epochs=3, bs=4, seq=192, lr=None,
        hint="只训练增量小矩阵，需先有 full_sft 权重"),
    "DPO": dict(
        file="train_dpo.py", flag_save="--save_weight", save="dpo_toy",
        From="full_sft_toy", data="dpo.jsonl", epochs=1, bs=1, seq=512, lr=None,
        hint="偏好对齐（chosen/rejected），需先有 full_sft 权重"),
}
LOSS_RE = re.compile(
    r"Epoch:\[(\d+)/(\d+)\]\((\d+)/(\d+)\), loss: ([\d.]+).*?(?:lr|learning_rate): ([\d.eE+-]+)")

ss = st.session_state
if "proc" not in ss:
    ss.proc, ss.meta = None, {}


def build_cmd(key, p):
    """根据页面参数拼出训练命令（与指导文档第 7.3 节一致）"""
    cfg = SCRIPTS[key]
    cmd = [sys.executable, f"trainer/{cfg['file']}",
           "--data_path", f"../dataset/{p['data']}",
           "--hidden_size", str(p["hidden"]), "--num_hidden_layers", str(p["layers"]),
           "--batch_size", str(p["bs"]), "--max_seq_len", str(p["seq"]),
           "--epochs", str(p["epochs"]),
           "--num_workers", "2",
           "--log_interval", str(p["log_iv"]), "--save_interval", str(p["save_iv"])]
    if cfg["From"]:
        cmd += ["--from_weight", p["from"]]
    cmd += [cfg["flag_save"], p["save"]]
    if p["lr"] is not None:
        cmd += ["--learning_rate", str(p["lr"])]
    return cmd


def parse_log():
    """从日志中解析 (step, loss, lr) 点位与最近日志行"""
    if not LOG_PATH.exists():
        return pd.DataFrame(), []
    rows = []
    for m in LOSS_RE.finditer(LOG_PATH.read_text(encoding="utf-8", errors="ignore")):
        ep, ept, step, total, loss, lr = m.groups()
        rows.append(dict(step=int(step), epoch=int(ep), total_step=int(total), loss=float(loss), lr=float(lr)))
    tail = LOG_PATH.read_text(encoding="utf-8", errors="ignore").strip().split("\n")[-40:]
    return pd.DataFrame(rows), tail


# ============ 侧边栏：配置与启动 ============
with st.sidebar:
    st.title("🛠️ MiniMind 训练台")
    key = st.selectbox("训练脚本", list(SCRIPTS.keys()))
    cfg = SCRIPTS[key]
    st.caption(cfg["hint"])

    preset = st.radio("参数预设", ["玩具预设（本机 CPU）", "官方 64M（需 GPU，本机极慢）"], horizontal=False)
    toy = preset.startswith("玩具")

    data_files = sorted(f.name for f in DATASET.glob("*.jsonl")) or ["(无数据文件)"]
    p = dict(
        data=st.selectbox("数据文件 (dataset/)", data_files,
                          index=data_files.index(cfg["data"]) if cfg["data"] in data_files else 0),
        hidden=st.number_input("hidden_size", 32, 4096, 256 if toy else 768, step=32),
        layers=st.number_input("num_hidden_layers", 1, 64, 4 if toy else 8),
        bs=st.number_input("batch_size", 1, 512, cfg["bs"] if toy else max(cfg["bs"], 8)),
        seq=st.number_input("max_seq_len", 32, 8192, cfg["seq"] if toy else max(cfg["seq"], 340), step=32),
        epochs=st.number_input("epochs", 1, 100, cfg["epochs"] if toy else 2),
        save=st.text_input("输出权重名", cfg["save"]),
        from_=st.text_input("输入权重名", cfg["From"] or "none"),
        lr_on=st.checkbox("使用脚本默认学习率", value=True),
        lr=None,
    )
    if not p["lr_on"]:
        p["lr"] = st.number_input("learning_rate", 1e-9, 1e-2, 1e-4, format="%e")
    p["log_iv"] = st.number_input("log_interval", 1, 1000, 20 if toy else 100)
    p["save_iv"] = st.number_input("save_interval", 1, 5000, 200 if toy else 1000)

    running = ss.proc is not None and ss.proc.poll() is None
    if running:
        st.warning("已有训练在运行，先停止再启动新的。")
        if st.button("⛔ 停止当前训练", type="primary", use_container_width=True):
            ss.proc.terminate()
            try:
                ss.proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                ss.proc.kill()
            ss.proc, ss.meta = None, {}
            st.rerun()
    else:
        weight_file = OUT_DIR / f"{p['from_']}_{p['hidden']}.pth"
        if cfg["From"] and not weight_file.exists():
            st.error(f"缺少输入权重: out/{p['from_']}_{p['hidden']}.pth\n\n请先训练/下载该阶段权重。")
        if not (DATASET / p["data"]).exists():
            st.error(f"缺少数据文件: dataset/{p['data']}")
        if st.button("🚀 开始训练", type="primary", use_container_width=True,
                     disabled=(cfg["From"] and not weight_file.exists()) or not (DATASET / p["data"]).exists()):
            OUT_DIR.mkdir(exist_ok=True)
            cmd = build_cmd(key, p)
            log_f = open(LOG_PATH, "w", encoding="utf-8")
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            ss.proc = subprocess.Popen(cmd, cwd=TRAINER, stdout=log_f, stderr=subprocess.STDOUT, env=env)
            ss.meta = dict(cmd=" ".join(cmd), t0=time.time(), script=key)
            st.rerun()

    st.divider()
    st.caption(f"设备: {'GPU (CUDA)' if __import__('torch').cuda.is_available() else 'CPU'}"
               f" · 日志: out/ui_train.log")

# ============ 主区域：状态 / 指标 / 曲线 / 日志 ============
st.title("训练监控")
running = ss.proc is not None and ss.proc.poll() is None

if ss.meta:
    st.code(ss.meta["cmd"], language="bash")
elif LOG_PATH.exists():
    st.info("当前无运行中的任务，下面显示的是最近一次训练的记录（只读）。")

df, tail = parse_log()

if df.empty:
    st.caption("暂无日志。启动训练后，这里会出现 loss 曲线与实时输出。")
else:
    c1, c2, c3, c4 = st.columns(4)
    last = df.iloc[-1]
    c1.metric("最新 loss", f"{last['loss']:.4f}")
    c2.metric("学习率", f"{last['lr']:.2e}")
    c3.metric("进度 (step)", f"{int(last['step'])} / {int(last['total_step'])}")
    if running and ss.meta:
        c4.metric("已运行", f"{(time.time() - ss.meta['t0']) / 60:.1f} 分钟")
    else:
        c4.metric("状态", "已结束" if not running else "运行中")
    st.line_chart(df.set_index("step")[["loss"]], height=280)

st.subheader("实时日志")
if tail:
    status = "🟢 运行中" if running else "⚪ 已停止"
    st.caption(f"{status} · 最近 {len(tail)} 行")
    st.code("\n".join(tail), language=None)
else:
    st.caption("暂无输出。")

# 运行中则自动刷新
if running:
    time.sleep(2)
    st.rerun()
