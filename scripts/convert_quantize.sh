#!/usr/bin/env bash
# Convert merged HF weights to GGUF and quantize to Q4_K_M.
#
# IMPORTANT: go straight from f16 to Q4_K_M. Do not quantize to q8_0 first
# and then re-quantize down to Q4_K_M. That second pass measurably destroys
# the weights: the model keeps its tone but stops answering the question
# ("1+1=" -> "大于2 / 可能等于2"). It took a full day to trace.
#
#   bf16 merged -> f16 gguf -> Q4_K_M gguf
#
# Requires llama.cpp's convert_hf_to_gguf.py and llama-quantize on PATH.
set -euo pipefail

MERGED="${MERGED:-merged}"
F16="${F16:-model-f16.gguf}"
Q4="${Q4:-persona-q4km.gguf}"

python convert_hf_to_gguf.py "$MERGED" --outtype f16 --outfile "$F16"
llama-quantize "$F16" "$Q4" Q4_K_M

ls -lh "$F16" "$Q4"
md5sum "$Q4"

# Sanity check, not optional: a broken quantisation answers these wrong.
#   1+1 = ?            -> should be 2
#   首都是哪里          -> should be 北京
# If either one rambles, the file is bad, not the prompt.
