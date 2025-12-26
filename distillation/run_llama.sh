#!/bin/bash

# ==========================================
# Llama-3 VLLM Server 启动脚本
# 功能: 启动一个兼容 OpenAI API 的推理服务
# 硬件: RTX 4090 (24GB)
# ==========================================

# 1. 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=0

# 2. 设置模型路径 (请根据实际情况修改)
MODEL_PATH="/home/daiwenju/Llama-3.1-8B-Instruct-AWQ"

# 3. 设置服务参数
# --max-model-len: Llama-3 支持长文本，设为 8192 防止截断
# --gpu-memory-utilization: 预留一点显存给系统，设为 0.9 或 0.95
# --dtype: 4090 推荐使用 bfloat16
# --served-model-name: 客户端调用时使用的模型名称
PORT=8000

echo "Starting vLLM API Server..."
echo "Model: $MODEL_PATH"
echo "Port: $PORT"

# 启动 vLLM Server
# 注意: 第一次运行可能需要下载 tokenizer，请保持网络连接或提前下载好
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "Llama-3.1" \
    --tensor-parallel-size 1 \
    --max-model-len 18000 \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --port $PORT \
    --trust-remote-code