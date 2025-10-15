#!/bin/bash

# OpenPI 训练启动脚本
# 使用方法: bash toollllllll/start_training.sh

echo "=========================================="
echo "OpenPI 训练脚本"
echo "=========================================="
echo ""
echo "配置信息:"
echo "  - 配置名称: pi0_custom_robot"
echo "  - 实验名称: grab_bottle_experiment"
echo "  - 任务: grab_bottle (抓瓶子)"
echo "  - 数据集: caesar/my_robot_demo"
echo "  - 训练步数: 30,000"
echo "  - Batch size: 16"
echo ""
echo "Checkpoints 保存位置:"
echo "  checkpoints/pi0_custom_robot/grab_bottle_experiment/"
echo ""
echo "=========================================="
echo ""

# 设置 JAX 内存使用
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# 开始训练
echo "开始训练..."
uv run scripts/train.py pi0_custom_robot \
  --exp-name=grab_bottle_experiment \
  --overwrite

echo ""
echo "=========================================="
echo "训练完成或已终止"
echo "=========================================="
