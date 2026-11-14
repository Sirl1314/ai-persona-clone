#!/usr/bin/env bash
# One-shot training entrypoint.
#
# Training is delegated to LLaMA-Factory, not hand-written Trainer code.
# The whole hyperparameter set lives in configs/train_qlora.yaml; this script
# only wires the dataset registry and calls the CLI.
#
# Requirements:
#   pip install llamafactory[metrics]  (>= 0.9.5)
#   a GPU with >= 24GB VRAM for a 7B model at 4-bit
#
# Usage:
#   bash scripts/train.sh
#   bash scripts/train.sh configs/train_qlora.yaml
set -euo pipefail

CONFIG="${1:-configs/train_qlora.yaml}"

# LLaMA-Factory resolves dataset names through this registry.
export LLAMAFACTORY_DATASET_DIR="${LLAMAFACTORY_DATASET_DIR:-samples}"
cp -n configs/dataset_info.json "$LLAMAFACTORY_DATASET_DIR/dataset_info.json" 2>/dev/null || true

llamafactory-cli train "$CONFIG"

echo "done. adapter written to the output_dir in $CONFIG"
