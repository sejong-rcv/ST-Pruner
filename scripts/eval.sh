#!/bin/bash
# Shared runner used by scripts/v1_5/eval/*.sh and scripts/v1_6/eval/*.sh.
#
#   bash scripts/eval.sh <CKPT> <BENCHMARK> <VISUAL_TOKENS> [METHOD]
#
#   CKPT          : Hugging Face id or local path of the LLaVA checkpoint
#   BENCHMARK     : vqav2 | gqa | vizwiz | sqa | textvqa | pope | mme | mmbench | mmbench_cn
#   VISUAL_TOKENS : retained visual tokens (LLaVA-1.5: 192/128/64, LLaVA-NeXT: 640/320/160)
#   METHOD        : stpruner (default) | spectral | none
#
# Inference is split over all GPUs in CUDA_VISIBLE_DEVICES, then the chunks are merged and scored.
# Environment overrides: DATA_DIR (default ./playground/data/eval), OUTPUT_DIR (default ./results).

set -e

CKPT=$1
BENCH=$2
TOKEN=$3
METHOD=${4:-stpruner}

DATA_DIR=${DATA_DIR:-./playground/data/eval}
OUTPUT_DIR=${OUTPUT_DIR:-./results}
NAME=$(basename "$CKPT")

if [ "$METHOD" = "none" ]; then
    TAG="full"
    TOKEN_ARGS=""
else
    TAG="${METHOD}_${TOKEN}"
    TOKEN_ARGS="--visual-tokens ${TOKEN}"
fi
OUT=${OUTPUT_DIR}/${NAME}/${BENCH}
mkdir -p ${OUT}/${TAG}

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.infer \
        --model-path ${CKPT} \
        --benchmark ${BENCH} \
        --data-dir ${DATA_DIR} \
        --method ${METHOD} ${TOKEN_ARGS} \
        --answers-file ${OUT}/${TAG}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks ${CHUNKS} \
        --chunk-idx ${IDX} \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done
wait

RESULT=${OUT}/${TAG}.jsonl
> ${RESULT}
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${OUT}/${TAG}/${CHUNKS}_${IDX}.jsonl >> ${RESULT}
done

if [ "$BENCH" = "vqav2" ]; then
    python scripts/convert_vqav2_for_submission.py \
        --result-file ${RESULT} \
        --upload-file ${OUT}/upload/${TAG}.json \
        --test-split ${DATA_DIR}/vqav2/llava_vqav2_mscoco_test2015.jsonl
elif [ "$BENCH" = "vizwiz" ]; then
    python scripts/convert_vizwiz_for_submission.py \
        --result-file ${RESULT} \
        --upload-file ${OUT}/upload/${TAG}.json \
        --test-split ${DATA_DIR}/vizwiz/llava_test.jsonl
    python -m llava.eval.score --benchmark ${BENCH} --result-file ${RESULT} --data-dir ${DATA_DIR}
else
    python -m llava.eval.score --benchmark ${BENCH} --result-file ${RESULT} --data-dir ${DATA_DIR}
fi
