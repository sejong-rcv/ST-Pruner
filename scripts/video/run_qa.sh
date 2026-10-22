#!/bin/bash
# Zero-shot video QA with Video-LLaVA-7B (run inside the Video-LLaVA environment, see video_llava/README.md)
#
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/video/run_qa.sh <BENCHMARK> [METHOD] [PER_FRAME_KEEP]
#
#   BENCHMARK      : MSVD | MSRVTT | TGIF
#   METHOD         : stpruner (default) | spectral | none (full 2048 tokens)
#   PER_FRAME_KEEP : tokens kept per 256-token frame (default 17 -> 136 tokens over 8 frames)
#
# Then score with the GPT-3.5 judge (needs an OpenAI key):
#   python video_llava/eval_gpt_judge.py --pred <OUT>/merge.jsonl --gt-file <DATA>/test_a.json --out <OUT>/judge_gpt.json

BENCH=$1
METHOD=${2:-stpruner}
KEEP=${3:-17}

CKPT=${CKPT:-LanguageBind/Video-LLaVA-7B}
CACHE_DIR=${CACHE_DIR:-./cache_dir}
DATA_DIR=${DATA_DIR:-./playground/data/eval/GPT_Zero_Shot_QA}
OUTPUT_DIR=${OUTPUT_DIR:-./results}

case $BENCH in
    MSVD)   VIDEO_DIR=${DATA_DIR}/MSVD_Zero_Shot_QA/videos ;;
    MSRVTT) VIDEO_DIR=${DATA_DIR}/MSRVTT_Zero_Shot_QA/videos/all ;;
    TGIF)   VIDEO_DIR=${DATA_DIR}/TGIF_Zero_Shot_QA/mp4 ;;
    *) echo "Unknown benchmark: $BENCH (MSVD | MSRVTT | TGIF)"; exit 1 ;;
esac
GT_DIR=${DATA_DIR}/${BENCH}_Zero_Shot_QA

if [ "$METHOD" = "none" ]; then TAG="full"; else TAG="${METHOD}_$((KEEP * 8))"; fi
OUT=${OUTPUT_DIR}/Video-LLaVA-7B/${BENCH}/${TAG}

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python video_llava/infer_prune.py \
        --model_path ${CKPT} \
        --cache_dir ${CACHE_DIR} \
        --video_dir ${VIDEO_DIR} \
        --gt_file_question ${GT_DIR}/test_q.json \
        --gt_file_answers ${GT_DIR}/test_a.json \
        --output_dir ${OUT} \
        --output_name ${CHUNKS}_${IDX} \
        --num_chunks ${CHUNKS} \
        --chunk_idx ${IDX} \
        --method ${METHOD} \
        --per-frame-keep ${KEEP} &
done
wait

> ${OUT}/merge.jsonl
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${OUT}/${CHUNKS}_${IDX}.json >> ${OUT}/merge.jsonl
done
echo "Saved ${OUT}/merge.jsonl"
