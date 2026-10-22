#!/bin/bash
# MMBench-EN dev (circular accuracy) | LLaVA-NeXT-7B (LLaVA-1.6)
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts/v1_6/eval/mmbench.sh <VISUAL_TOKENS> [METHOD]
#
#   VISUAL_TOKENS : 640 | 320 | 160  (out of 2880 = 5 views x 576; 128 / 64 / 32 per view)
#   METHOD        : stpruner (default) | spectral | none (full-token baseline)
#
# Set CKPT to a local path to use a downloaded checkpoint.

CKPT=${CKPT:-liuhaotian/llava-v1.6-vicuna-7b}
TOKEN=${1:-320}
METHOD=${2:-stpruner}

bash scripts/eval.sh ${CKPT} mmbench ${TOKEN} ${METHOD}
