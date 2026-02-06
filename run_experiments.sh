#!/bin/bash

# Ensure experiments directory exists
mkdir -p experiments

echo "Starting parallel experiments..."
echo "Killing any existing training processes..."
pkill -f train_arcface.py || true

# ============================================================
# PARALLEL EXPERIMENTS - Baseline vs CBAM Attention
# ============================================================

# Experiment 1: Baseline (No Attention)
# echo "Launching Experiment 1: Baseline (No Attention)..."
# nohup python training/train_arcface.py \
#     --optimizer sgd \
#     --model_lr 0.1 \
#     --loss_lr 0.1 \
#     --weight_decay 5e-4 \
#     --output_dir experiments/baseline_no_attention \
#     --num_workers 4 \
#     --epochs 50 \
#     > experiments/baseline_no_attention.log 2>&1 &
# PID1=$!
# echo "PID: $PID1"

# Experiment 2: CBAM Attention (Default)
echo "Launching Experiment 2: CBAM Attention..."
nohup python training/train_arcface.py \
    --optimizer sgd \
    --model_lr 0.1 \
    --loss_lr 0.1 \
    --weight_decay 5e-4 \
    --output_dir experiments/cbam_attention \
    --num_workers 4 \
    --epochs 50 \
    > experiments/cbam_attention.log 2>&1 &
PID2=$!
echo "PID: $PID2"

# Experiment 3: CBAM + Lower LR (0.01)
echo "Launching Experiment 3: CBAM + LR=0.01..."
nohup python training/train_arcface.py \
    --optimizer sgd \
    --model_lr 0.01 \
    --loss_lr 0.01 \
    --weight_decay 5e-4 \
    --output_dir experiments/cbam_lr_0.01 \
    --num_workers 4 \
    --epochs 50 \
    > experiments/cbam_lr_0.01.log 2>&1 &
PID3=$!
echo "PID: $PID3"

# Experiment 4: AdamW + CBAM
echo "Launching Experiment 4: AdamW + CBAM..."
nohup python training/train_arcface.py \
    --optimizer adamw \
    --model_lr 0.0003 \
    --loss_lr 0.001 \
    --weight_decay 0.01 \
    --output_dir experiments/adamw_cbam \
    --num_workers 4 \
    --epochs 50 \
    > experiments/adamw_cbam.log 2>&1 &
PID4=$!
echo "PID: $PID4"

echo ""
echo "============================================================"
echo "Experiments running with PIDs: $PID1, $PID2, $PID3, $PID4"
echo "============================================================"
echo ""
echo "Monitor logs:"
# echo "  tail -f experiments/baseline_no_attention.log"
echo "  tail -f experiments/cbam_attention.log"
echo "  tail -f experiments/cbam_lr_0.01.log"
echo "  tail -f experiments/adamw_cbam.log"
echo ""
echo "Waiting for completion..."

wait
echo "All experiments finished."
