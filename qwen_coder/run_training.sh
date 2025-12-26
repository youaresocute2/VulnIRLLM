#!/bin/bash

# ===============================================================
# 🔧 基础环境配置
# ===============================================================
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ===============================================================
# 🚀 极简启动 (完全依赖 training.py)
# ===============================================================

# 1. 确保日志目录存在 (即使我们不知道具体 output_dir，先建个 logs 文件夹也行，
#    或者直接让 nohup 写在当前目录，或者用 Python 内部的 output_dir)
#    为了简单且健壮，我们先读一下 python 里的 output_dir 或者是写死一个 log 根目录
mkdir -p ./logs

# 2. 定义日志文件
CURRENT_TIME=$(date "+%Y%m%d_%H%M%S")
LOG_FILE="./logs/train_run_${CURRENT_TIME}.log"

echo "--> Launching Training..."
echo "--> Configuration is loaded strictly from: qwen_coder/configs/training.py"
echo "--> Logs will be saved to: $LOG_FILE"

# 3. 执行 nohup
nohup python -m qwen_coder.finetuning > "$LOG_FILE" 2>&1 &

PID=$!
echo ""
echo "Training started in BACKGROUND!"
echo "PID: $PID"
echo "Log Monitor: tail -f $LOG_FILE"
echo "Kill: kill $PID"
echo ""