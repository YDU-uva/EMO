#!/bin/bash

# EMO (Expected Model Output) 训练和测试脚本
# 基于ARML代码实现的期望输出优化方法

echo "================================================"
echo "EMO模型训练和测试脚本"
echo "================================================"

# 设置基本参数
DATASOURCE="plainmulti"  # 可选: 2D, plainmulti, artmulti
NUM_CLASSES=5
META_BATCH_SIZE=25
UPDATE_BATCH_SIZE=5
NUM_UPDATES=1
META_LR=0.001
UPDATE_LR=0.001
METATRAIN_ITERATIONS=15000

# EMO特有参数
NUM_ENSEMBLE_MODELS=5
UNCERTAINTY_WEIGHT=0.1
EXPECTED_OUTPUT_WEIGHT=1.0

# 创建日志目录
mkdir -p logs_emo

echo "开始EMO模型训练..."
echo "数据源: $DATASOURCE"
echo "集成模型数量: $NUM_ENSEMBLE_MODELS"
echo "不确定性权重: $UNCERTAINTY_WEIGHT"

# 训练命令
python main_emo.py \
    --datasource=$DATASOURCE \
    --num_classes=$NUM_CLASSES \
    --meta_batch_size=$META_BATCH_SIZE \
    --update_batch_size=$UPDATE_BATCH_SIZE \
    --num_updates=$NUM_UPDATES \
    --meta_lr=$META_LR \
    --update_lr=$UPDATE_LR \
    --metatrain_iterations=$METATRAIN_ITERATIONS \
    --num_ensemble_models=$NUM_ENSEMBLE_MODELS \
    --uncertainty_weight=$UNCERTAINTY_WEIGHT \
    --expected_output_weight=$EXPECTED_OUTPUT_WEIGHT \
    --train=True \
    --log=True

echo "训练完成!"

echo "开始EMO模型测试..."

# 测试命令
python main_emo.py \
    --datasource=$DATASOURCE \
    --num_classes=$NUM_CLASSES \
    --meta_batch_size=1 \
    --update_batch_size=$UPDATE_BATCH_SIZE \
    --num_updates_test=20 \
    --num_test_task=1000 \
    --num_ensemble_models=$NUM_ENSEMBLE_MODELS \
    --uncertainty_weight=$UNCERTAINTY_WEIGHT \
    --expected_output_weight=$EXPECTED_OUTPUT_WEIGHT \
    --train=False \
    --test_set=True

echo "测试完成!"
echo "结果保存在 logs_emo/ 目录中" 