#!/usr/bin/env bash
# Merge a LoRA adapter into the base model and write sharded safetensors.
#
# Exporting on CPU keeps peak host memory low. export_size 4 splits the output
# into 4 shards, which matters when the disk is tight (a 7B model in bf16 is
# about 15.5GB).
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
ADAPTER="${ADAPTER:-saves/qwen25-7b-lora}"
OUT_DIR="${OUT_DIR:-merged}"

llamafactory-cli export \
  --model_name_or_path "$BASE_MODEL" \
  --adapter_name_or_path "$ADAPTER" \
  --template qwen \
  --finetuning_type lora \
  --export_dir "$OUT_DIR" \
  --export_size 4 \
  --export_device cpu \
  --export_legacy_format false

echo "merged model written to $OUT_DIR"
ls -lh "$OUT_DIR"
