#!/bin/bash

# ==============================================================================
# === 1. 全局配置 (在此处修改通用参数) =======================================
# ==============================================================================

# 设置要运行的 Python 脚本
PYTHON_SCRIPT="main.py"

# 指定要使用的 GPU ID
GPU_ID=1

# 创建结果输出目录，-p 参数确保目录已存在时不会报错
mkdir -p log

# ==============================================================================
# === 2. 实验运行辅助函数 (无需修改) ========================================
# ==============================================================================

# 定义一个函数来执行单个实验，它会自动处理日志记录和屏幕输出
# 参数1: 日志文件名
# 参数2及之后: 所有要传递给 python 脚本的参数
run_experiment() {
    LOG_FILE_NAME="log/$1"
    shift # 移除第一个参数(日志文件名)，剩下的都是python脚本的参数

    echo "======================================================================"
    echo "▶️  开始实验 | 日志文件: ${LOG_FILE_NAME}"
    echo "▶️  参数: $@"
    echo "======================================================================"

    # 执行 Python 脚本，并使用 tee 命令同时将输出打印到屏幕和日志文件
    python "$PYTHON_SCRIPT" "$@" | tee "$LOG_FILE_NAME"

    # 检查上一个命令的退出状态
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "❌ 警告: 上一个实验运行失败。"
    else
        echo "✅ 实验成功结束。"
    fi
    echo ""
}

# ==============================================================================
# === 3. 定义不同的实验组 (在此处设计您的实验) ===============================
# ==============================================================================

# 实验组1: 对比不同的“教师选择”策略
compare_teacher_selectors() {
    echo "********* 开始运行 [教师选择策略] 对比实验 *********"
    # 固定其他变量，只改变 --teacher_selector
    FIXED_PARAMS="--gpu_id $GPU_ID --distill_data_source original --sample_filter none --distill_mode dkd"

    run_experiment "log_teacher_all.txt" $FIXED_PARAMS --teacher_selector all
    run_experiment "log_teacher_fedd3a.txt" $FIXED_PARAMS --teacher_selector fedd3a
    run_experiment "log_teacher_top1.txt" $FIXED_PARAMS --teacher_selector top1
    run_experiment "log_teacher_top2.txt" $FIXED_PARAMS --teacher_selector top2
    run_experiment "log_teacher_top3.txt" $FIXED_PARAMS --teacher_selector top3
}

# 实验组2: 对比不同的“样本筛选”策略 (消融实验)
compare_sample_filters() {
    echo "********* 开始运行 [样本筛选策略] 对比实验 *********"
    # 固定教师选择策略为效果较好的 top2，对比有无困难样本筛选的区别
    FIXED_PARAMS="--gpu_id $GPU_ID --distill_data_source original --teacher_selector top2 --distill_mode dkd"
    
    # 基线：不进行任何样本筛选
    run_experiment "log_filter_none.txt" $FIXED_PARAMS --sample_filter none

    # 实验组：筛选出30%最困难的样本
    run_experiment "log_filter_confidence_0.3.txt" $FIXED_PARAMS --sample_filter confidence --hard_sample_ratio 0.3 --distill_batch_size 107

    # 实验组：筛选出50%最困难的样本
    run_experiment "log_filter_confidence_0.5.txt" $FIXED_PARAMS --sample_filter confidence --hard_sample_ratio 0.5 --distill_batch_size 64
}

# 实验组3: LTE 机制的消融实验
ablation_study_lte() {
    echo "********* 开始运行 [LTE机制] 消融实验 *********"
    FIXED_PARAMS="--gpu_id $GPU_ID --distill_data_source original --teacher_selector fedd3a --sample_filter none"

    # 对照组：关闭LTE机制 (lambda_ltc = 0)
    run_experiment "log_lte_off.txt" $FIXED_PARAMS --lambda_ltc 0

    # 实验组：开启LTE机制 (lambda_ltc = 1.0)
    run_experiment "log_lte_on.txt" $FIXED_PARAMS --lambda_ltc 1.0
}

# 实验组4: 数据来源的对比实验
compare_data_sources() {
    echo "********* 开始运行 [数据来源] 对比实验 *********"
    FIXED_PARAMS="--gpu_id $GPU_ID --teacher_selector fedd3a --sample_filter none --distill_mode dkd"

    # 使用公开数据集
    run_experiment "log_source_original.txt" $FIXED_PARAMS --distill_data_source original

    # 使用模型生成数据 (普通模式)
    run_experiment "log_source_generated_normal.txt" $FIXED_PARAMS --distill_data_source generated --distillation_data_mode normal

    # 使用模型生成数据 (稀缺优先模式)
    run_experiment "log_source_generated_scarcity.txt" $FIXED_PARAMS --distill_data_source generated --distillation_data_mode hard_scarcity
}

grid_search_dkd() {
    echo "********* 开始运行 [DKD 超参数 alpha & beta] 网格搜索 *********"
    
    # --- 在这里定义您想要搜索的 alpha 和 beta 值 ---
    # 根据 DKD 原论文和 FedDKDGen 的实践，beta 通常大于 alpha
    ALPHA_VALUES=(0.5 1.0 2.0)
    BETA_VALUES=(2.0 4.0 8.0 16.0)

    # 固定其他实验参数，以确保只测试 alpha 和 beta 的影响
    FIXED_PARAMS="--gpu_id $GPU_ID --distill_data_source generated --rounds 100 --teacher_selector fedd3a --sample_filter hard_scarcity --distill_mode dkd"

    # 使用嵌套循环遍历所有 alpha 和 beta 的组合
    for alpha in "${ALPHA_VALUES[@]}"; do
        for beta in "${BETA_VALUES[@]}"; do
            # 为每个组合创建一个唯一的日志文件名
            LOG_NAME="log_dkd_alpha_${alpha}_beta_${beta}.txt"
            
            # 运行实验
            run_experiment "$LOG_NAME" $FIXED_PARAMS --dkd_alpha "$alpha" --dkd_beta "$beta"
        done
    done
}


# ==============================================================================
# === 4. 主执行入口 (在此处选择要运行哪个实验组) =============================
# ==============================================================================

echo "实验脚本启动..."

# --- 您可以在这里选择要运行的实验组 ---
# --- 取消您想运行的实验组前面的注释符 '#' 即可 ---

# compare_teacher_selectors
# compare_sample_filters
# ablation_study_lte
# compare_data_sources
grid_search_dkd

echo "所有选定的实验组均已执行完毕。"