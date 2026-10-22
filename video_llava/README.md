# ST-Pruner on Video-LLaVA

Video-LLaVA-7B represents a video with 8 frames × 256 patch tokens. ST-Pruner treats each frame as
an independent visual group: it scores the 256 patches with the spectral branch (R = 256/8 = 32) and
the inverted LanguageBind patch–text similarity, and keeps the top 17 tokens per frame
(136 tokens in total) before the multimodal projector.

`video_prune.py` wraps `model.encode_videos` and leaves the Video-LLaVA code unchanged. The
`videollava` package requires transformers 4.31, so this part runs in a separate environment and
re-declares the scoring functions of `stpruner/pruner.py`.

## Setup

```bash
git clone https://github.com/PKU-YuanGroup/Video-LLaVA.git
cd Video-LLaVA
conda create -n stpruner_video python=3.10 -y
conda activate stpruner_video
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install -e .
pip install decord opencv-python git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d
pip install openai        # GPT-3.5 judge
cd ..
```

We used transformers 4.31.0, decord 0.6.0 and pytorchvideo 0.1.5. Prepare the data as described in
[EVAL.md](../EVAL.md#video-benchmarks-video-llava).

## Inference

Run from the ST-Pruner repository root with the `stpruner_video` environment:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/video/run_qa.sh MSVD            # ST-Pruner, 136 tokens
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/video/run_qa.sh MSVD spectral   # spectral-only
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/video/run_qa.sh MSVD none       # full 2048 tokens
```

The merged predictions are written to `results/Video-LLaVA-7B/<BENCH>/<method>_136/merge.jsonl`.

Video-LLaVA loads its LanguageBind image/video towers with a hard-coded `cache_dir='./cache_dir'`, so the
towers are downloaded into `./cache_dir` of the directory you run from (the repository root here).

## GPT-3.5 judge

```bash
export OPENAI_API_KEY=...        # your API Keys
python video_llava/eval_gpt_judge.py \
    --pred    results/Video-LLaVA-7B/MSVD/stpruner_136/merge.jsonl \
    --gt-file playground/data/eval/GPT_Zero_Shot_QA/MSVD_Zero_Shot_QA/test_a.json \
    --out     results/Video-LLaVA-7B/MSVD/stpruner_136/judge_gpt.json \
    --model gpt-3.5-turbo --workers 8
```

The judge uses the Video-ChatGPT prompt and parsing from Video-LLaVA. Every judgment is appended to
`judge_gpt.jsonl`, so an interrupted run (e.g. on a daily rate limit) resumes where it stopped.

| File | Role |
|---|---|
| `video_prune.py` | per-frame spectral / semantic scoring and the `encode_videos` wrapper |
| `infer_prune.py` | video QA inference (`--method`, `--per-frame-keep`) |
| `eval_gpt_judge.py` | GPT-3.5 judge |
