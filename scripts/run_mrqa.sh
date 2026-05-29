#!/bin/bash

set -e

OUTPUT_DIR="./outputs_mrqa"
DATASET_NAME="MRQA"
DEVICE="cuda:2"

echo "[HiVe] Training on MRQA..."
python train.py \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE"

echo "[HiVe] Evaluating on MRQA (out-of-domain)..."
python evaluate.py \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE"

echo "[HiVe] Done. Results saved to $OUTPUT_DIR"