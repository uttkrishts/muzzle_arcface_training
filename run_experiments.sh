#!/bin/bash

# Ensure experiments directory exists
mkdir -p experiments

echo "Starting parallel experiments..."

# Experiment 1: SGD High LR (0.1)
echo "Launching Experiment 1: SGD LR=0.1..."
nohup python training/train_arcface.py \
    --optimizer sgd \
    --model_lr 0.1 \
    --loss_lr 0.1 \
    --weight_decay 5e-4 \
    --output_dir experiments/sgd_0.1 \
    > experiments/sgd_0.1.log 2>&1 &
PID1=$!

# Experiment 2: SGD Mid LR (0.01)
echo "Launching Experiment 2: SGD LR=0.01..."
nohup python training/train_arcface.py \
    --optimizer sgd \
    --model_lr 0.01 \
    --loss_lr 0.01 \
    --weight_decay 5e-4 \
    --output_dir experiments/sgd_0.01 \
    > experiments/sgd_0.01.log 2>&1 &
PID2=$!

# Experiment 3: SGD Low LR (0.001)
echo "Launching Experiment 3: SGD LR=0.001..."
nohup python training/train_arcface.py \
    --optimizer sgd \
    --model_lr 0.001 \
    --loss_lr 0.001 \
    --weight_decay 5e-4 \
    --output_dir experiments/sgd_0.001 \
    > experiments/sgd_0.001.log 2>&1 &
PID3=$!

# Experiment 4: AdamW Control (LR=3e-4) - The previous winner
# Note: Using Loss LR of 0.001 (1e-3) as per previous success
echo "Launching Experiment 4: AdamW LR=3e-4..."
nohup python training/train_arcface.py \
    --optimizer adamw \
    --model_lr 0.0003 \
    --loss_lr 0.001 \
    --weight_decay 0.01 \
    --output_dir experiments/adamw_control \
    > experiments/adamw_control.log 2>&1 &
PID4=$!

echo "Experiments running with PIDs: $PID1, $PID2, $PID3, $PID4"
echo "Monitor logs at experiments/*.log"
echo "Waiting for completion..."

wait
echo "All experiments finished."
