#!/bin/bash



echo "================================================"
echo "EMO train and test script"
echo "================================================"


DATASOURCE="plainmulti"  # options: 2D, plainmulti, artmulti
NUM_CLASSES=5
META_BATCH_SIZE=25
UPDATE_BATCH_SIZE=5
NUM_UPDATES=1
META_LR=0.001
UPDATE_LR=0.001
METATRAIN_ITERATIONS=15000

# EMO specific parameters
NUM_ENSEMBLE_MODELS=5
UNCERTAINTY_WEIGHT=0.1
EXPECTED_OUTPUT_WEIGHT=1.0

# create log directory
mkdir -p logs_emo

echo "Start EMO model training..."
echo "数据源: $DATASOURCE"
echo "集成模型数量: $NUM_ENSEMBLE_MODELS"
echo "不确定性权重: $UNCERTAINTY_WEIGHT"

# train command
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

echo "Training completed!"

echo "Start EMO model testing..."

# test command
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

echo "Testing completed!"
echo "Results saved in logs_emo/ directory" 