#!/bin/bash
# VQAv2 test-dev (EvalAI submission) | LLaVA-1.5-7B
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/vqav2.sh <VISUAL_TOKENS> [METHOD]
#
#   VISUAL_TOKENS : 192 | 128 | 64  (out of 576)
#   METHOD        : stpruner (default) | spectral | none (full-token baseline)
#
# Set CKPT to a local path to use a downloaded checkpoint.

CKPT=${CKPT:-liuhaotian/llava-v1.5-7b}
TOKEN=${1:-64}
METHOD=${2:-stpruner}

bash scripts/eval.sh ${CKPT} vqav2 ${TOKEN} ${METHOD}
