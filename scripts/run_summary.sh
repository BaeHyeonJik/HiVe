#!/bin/bash
# ------------------------------------------
# HiVe: summary full reproduction script
# ------------------------------------------

set -e

OUTPUT_DIR="./outputs_summary"
DATASET_NAME="summary"
DEVICE="cuda:0"

echo "[HiVe] Training on summary..."
python train.py \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE"

echo "[HiVe] Evaluating on summary (out-of-domain)..."
python evaluate.py \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE"

echo "[HiVe] Done. Results saved to $OUTPUT_DIR"