#!/bin/bash

# --- 网格搜索设置 ---
# 设置你的 Python 脚本文件名
PYTHON_SCRIPT="main.py" 
# 设置固定的实验参数
TOPOLOGY="semidfl"
ROUNDS=100
DISTILLATION_MODE="normal" 
DISTILL_SOURCE="original"

# --- 定义要搜索的权重组合 ---
# 每个字符串代表一次实验的 --teacher_weights 参数。
declare -a WEIGHT_SETS=(
    "0.25 0.25 0.25 0.25"  # 1. 平均权重 (基线)
    "0.7 0.1 0.1 0.1"      # 2. 一个主导的教师模型
    "0.1 0.7 0.1 0.1"      # 3. 另一个主导的教师模型 (检查位置偏差)
    "0.4 0.4 0.1 0.1"      # 4. 两个较强的教师模型
    "0.5 0.3 0.1 0.1"      # 5. 权重递减的场景
)

# --- 执行循环 ---
echo "--- 开始网格搜索 (模式: $DISTILLATION_MODE) ---"
echo "Python 脚本: $PYTHON_SCRIPT"
echo "拓扑结构: $TOPOLOGY"
echo "通信轮数: $ROUNDS"
echo "---------------------------"

# 遍历权重组合数组
for weights in "${WEIGHT_SETS[@]}"; do
    echo "" 
    echo "*** 正在执行权重组合: [ $weights ] ***"
    echo "--------------------------------------------------"

    # 构建并执行命令
    python3 "$PYTHON_SCRIPT" \
        --topology "$TOPOLOGY" \
        --rounds "$ROUNDS" \
        --distillation_data_mode "$DISTILLATION_MODE" \
        --teacher_mode "all" \
        --teacher_weights $weights \
        --distill_data_source "$DISTILL_SOURCE"

    if [ $? -ne 0 ]; then
        echo "!!! 警告: 上一个权重组合 [ $weights ] 的实验运行失败。 !!!"
    fi

    echo "--- 权重 [ $weights ] 的实验已完成 ---"
done

echo ""
echo "--- 网格搜索全部完成 (模式: $DISTILLATION_MODE) ---"