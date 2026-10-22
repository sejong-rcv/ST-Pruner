# Don't Ask the LLM: Visual Token Pruning with Spectral Structure and Semantic Relevance

<!-- Teaser figure: put the image in assets/ and uncomment.
<p align="center"><img src="assets/teaser.png" width="90%"></p>
-->

<p align="center">
  <!-- Replace "#" with the paper link (CVF Open Access / arXiv) once available -->
  <a href="#"><img src="https://img.shields.io/badge/Paper-WACV%202027-blue"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green"></a>
</p>

**Intaek Shin, Geuntaek Lim, Sungjune Park, Yukyung Choi** &nbsp;·&nbsp; Sejong University

> **TL;DR.** ST-Pruner is a training-free, plug-and-play visual token pruner that selects tokens
> **before the LLM**, from the vision encoder alone. It multiplies an attention-free **spectral
> saliency** (low-rank SVD of the ViT patch features) with an **inverted patch–text similarity**
> from the paired CLIP text encoder. It keeps over 95% of full-token performance with at most
> 11.1% of the visual tokens, and runs LLaVA-NeXT 3.2× faster end-to-end.

## 📰 News

- ST-Pruner is accepted to **WACV 2027**.
- Code for LLaVA-1.5, LLaVA-NeXT and Video-LLaVA is released.

## 👁️ Overview

<!-- Method figure (Figure 3 of the paper): put the image in assets/ and uncomment.
<p align="center"><img src="assets/overview.png" width="95%"></p>
-->

[CLS]-attention selection becomes scattered and background-biased under aggressive compression.
Query-aware signals inside the LLM only become reliable in the mid-to-late decoder layers, after
most of the computation has already been spent. ST-Pruner gets both structure and query awareness
**before** the LLM runs. For each visual group (an image, a LLaVA-NeXT view or a video frame) with
patch features $Z \in \mathbb{R}^{N \times d}$:

| Component | Score |
|---|---|
| **Spectral saliency** (Sec. 3.2) | $Z \approx U_R \Sigma_R V_R^\top$ (randomized SVD, $R = N/8$), $\quad \bar s_i = \sum_{k \le R} \sqrt{\sigma_k}\, U_{ik}^2, \quad s = \mathcal{N}(\bar s)$ |
| **Semantic relevance** (Sec. 3.3) | $\rho_i = \phi(z_i)^\top \psi(T)$ (CLIP patch–text cosine), $\quad r = \mathcal{N}(-\rho)$ |
| **Dual-cue fusion** (Sec. 3.4) | $g_i = s_i \cdot r_i, \quad \mathcal{S} = \mathrm{TopK}(g, K)$ |

$\mathcal{N}$ is min–max normalization within the group. The kept tokens stay in their original order
and are passed to the LLM. The fusion rule has no tunable parameters, and the method needs no
training.

The method itself is in [`stpruner/pruner.py`](stpruner/pruner.py). It is called from
[`llava/model/llava_arch.py`](llava/model/llava_arch.py) (`encode_images`, marked `[ST-Pruner]`)
on the pre-projector ViT features.

## 🔥 Getting Started

### 🔨 Installation

```bash
git clone https://github.com/sejong-rcv/ST-Pruner.git
cd ST-Pruner

conda create -n stpruner python=3.10 -y
conda activate stpruner
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

All experiments in the paper were run on NVIDIA V100 GPUs with FP16 and SDPA attention (no
FlashAttention). Run every command from the repository root.

Video-LLaVA needs its own environment (transformers 4.31); see [`video_llava/README.md`](video_llava/README.md).

### 🎞 Models & Data

| Model | Checkpoint |
|---|---|
| LLaVA-1.5-7B | [liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b) |
| LLaVA-NeXT-7B (LLaVA-1.6) | [liuhaotian/llava-v1.6-vicuna-7b](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b) |
| Video-LLaVA-7B | [LanguageBind/Video-LLaVA-7B](https://huggingface.co/LanguageBind/Video-LLaVA-7B) |

The scripts load the Hugging Face checkpoints by default. The CLIP vision tower
(`openai/clip-vit-large-patch14-336`) is downloaded automatically, and its text tower is reused for
the semantic branch. To keep local copies instead:

```bash
huggingface-cli download liuhaotian/llava-v1.5-7b --local-dir checkpoints/llava-v1.5-7b
huggingface-cli download liuhaotian/llava-v1.6-vicuna-7b --local-dir checkpoints/llava-v1.6-vicuna-7b
huggingface-cli download openai/clip-vit-large-patch14-336                      # cached for the vision/text towers

CKPT=checkpoints/llava-v1.5-7b CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/pope.sh 64
```

Prepare the benchmarks under `./playground/data/eval` following [EVAL.md](EVAL.md) (the standard
LLaVA-1.5 layout).

### 📑 Evaluation

Each benchmark has a script. The first argument is the **number of retained visual tokens**, and
the second (optional) argument is the method:

```bash
# LLaVA-1.5-7B: keep 192 / 128 / 64 of the 576 visual tokens
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/pope.sh 64

# LLaVA-NeXT-7B: keep 640 / 320 / 160 of up to 2880 tokens (128 / 64 / 32 per 576-token view)
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_6/eval/textvqa.sh 320

# Spectral-only variant and the full-token baseline
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mme.sh 128 spectral
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mme.sh 576 none

# Multiple GPUs: the benchmark is split into one chunk per GPU, then merged and scored
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/v1_5/eval/vqav2.sh 128
```

| Script (`scripts/v1_5/eval/`, `scripts/v1_6/eval/`) | Benchmark | Metric |
|---|---|---|
| `vqav2.sh` | VQAv2 test-dev | EvalAI submission (`results/<model>/vqav2/upload/`) |
| `gqa.sh` | GQA testdev-balanced | accuracy |
| `vizwiz.sh` | VizWiz test-dev | accuracy (local GT if available) + EvalAI submission file |
| `sqa.sh` | ScienceQA test | accuracy |
| `textvqa.sh` | TextVQA val | accuracy |
| `pope.sh` | POPE | average F1 over random / popular / adversarial |
| `mme.sh` | MME | Perception score |
| `mmbench.sh` | MMBench-EN dev | CircularEval accuracy |

| Method | Score | Paper |
|---|---|---|
| `stpruner` (default) | $\mathcal{N}(\bar s) \cdot \mathcal{N}(-\rho)$ | ST-Pruner (Ours) |
| `spectral` | $\mathcal{N}(\bar s)$ | ST-Pruner (Spectral) |
| `none` | all tokens | Upper bound |

Answers are written to `results/<model>/<benchmark>/<method>_<tokens>.jsonl` and the scores to
`*_result.txt` next to them.

<details>
<summary><b>Running inference directly</b></summary>

```bash
python -m llava.eval.infer \
    --model-path liuhaotian/llava-v1.5-7b \
    --benchmark pope \
    --method stpruner --visual-tokens 64 \
    --answers-file results/llava-v1.5-7b/pope/stpruner_64.jsonl

python -m llava.eval.score --benchmark pope \
    --result-file results/llava-v1.5-7b/pope/stpruner_64.jsonl
```

| Argument | Default | Description |
|---|---|---|
| `--visual-tokens` | – | retained tokens (LLaVA-1.5: per image; LLaVA-NeXT: total for 5 views, split equally per view) |
| `--method` | `stpruner` | `stpruner` / `spectral` / `semantic` / `none` |
| `--rank` | `0` | spectral rank $R$; `0` uses $R = N/8$ (72 for 576 tokens) |
| `--sv-power` | `0.5` | exponent $p$ in $\sigma^p U^2$ (Table S4) |
| `--feat-space` | `vit` | `vit` (1024-d pre-projector) or `projector` (4096-d, Table S3) |
| `--semantic-sign` | `neg` | `neg` (−sim) or `pos` (+sim), for `--method semantic` (Table 5) |
| `--seed` | `0` | seed of the randomized SVD |

</details>

**Note on reproducibility.** The randomized SVD draws a random sketch, so results move slightly with
the seed (`--seed`, fixed to 0 by default). Expect small run-to-run differences from the numbers
below, mostly on MME, whose score sums many small subtasks.

## 📊 Results

**LLaVA-1.5-7B** (Table 1)

| Tokens | VQAv2 | GQA | VizWiz | SQA | TextVQA | POPE | MME-P | MMB | Avg. |
|---|---|---|---|---|---|---|---|---|---|
| 576 (upper bound) | 78.5 | 61.9 | 50.1 | 69.5 | 58.2 | 85.9 | 1506.5 | 64.7 | 100.0% |
| 192 | 77.2 | 60.3 | 52.3 | 69.4 | 57.6 | 87.3 | 1449.9 | 62.5 | 99.2% |
| 128 | 76.2 | 59.4 | 52.7 | 69.4 | 57.1 | 87.3 | 1432.3 | 62.2 | 98.6% |
| 64 | 73.3 | 56.8 | 52.1 | 69.8 | 55.5 | 86.5 | 1384.4 | 59.8 | 96.3% |

**LLaVA-NeXT-7B** (Table 2; token counts are maxima for five-view inputs)

| Tokens | VQAv2 | GQA | VizWiz | SQA | TextVQA | POPE | MME-P | Avg. |
|---|---|---|---|---|---|---|---|---|
| 2880 (upper bound) | 81.3 | 62.7 | 54.7 | 72.1 | 60.3 | 86.5 | 1501.1 | 100.0% |
| 640 | 79.1 | 62.0 | 55.8 | 71.8 | 60.0 | 87.0 | 1486.2 | 99.6% |
| 320 | 76.3 | 60.4 | 54.6 | 71.5 | 58.6 | 86.1 | 1448.2 | 97.5% |
| 160 | 73.0 | 58.1 | 54.2 | 71.7 | 55.9 | 85.0 | 1329.4 | 94.4% |

**Average retention vs. other pruning methods**

| Method | LLaVA-1.5 192 | 128 | 64 | LLaVA-NeXT 640 | 320 | 160 |
|---|---|---|---|---|---|---|
| FastV (ECCV'24) | 89.9 | 85.2 | 74.5 | 94.7 | 79.7 | – |
| SparseVLM (ICML'25) | 97.1 | 94.7 | 88.0 | 97.3 | 93.0 | – |
| VisionZip (CVPR'25) | 98.0 | 96.7 | 94.2 | 98.9 | 95.7 | 91.1 |
| DivPrune (CVPR'25) | 98.3 | 97.8 | 95.8 | 97.8 | 95.3 | 93.0 |
| VisPruner (ICCV'25) | 98.7 | 97.5 | 94.5 | 98.7 | 96.1 | 91.8 |
| ApET (CVPR'26) | 97.6 | 97.1 | 96.1 | 99.1 | 95.9 | 92.1 |
| **ST-Pruner (Ours)** | **99.2** | **98.6** | **96.3** | **99.6** | **97.5** | **94.4** |

**Video-LLaVA-7B** (Table 3, 136 of 2048 tokens, GPT-3.5 judged accuracy)

| Method | TGIF | MSVD | MSRVTT | Avg. |
|---|---|---|---|---|
| Upper bound (2048) | 47.1 | 69.8 | 56.7 | 100.0% |
| ST-Pruner (Spectral) | 43.6 | 66.9 | 52.5 | 93.7% |
| ST-Pruner (Ours) | 44.9 | 66.3 | 54.4 | 95.4% |

**Efficiency** (LLaVA-NeXT-7B, POPE, single V100, 320 tokens): prefill 679.2 → 188.0 ms (3.6×),
end-to-end 727.9 → 224.0 ms (3.2×), visual KV cache 1542 → 200 MB, at 86.1 POPE F1.
All scoring overhead is included. See the paper and the supplementary material for the
ablations and the complete efficiency sweep.

## 📁 Repository Structure

```
.
├── stpruner/
│   ├── pruner.py            # ST-Pruner: spectral saliency, semantic relevance, fusion
│   └── text_encoder.py      # frozen CLIP text encoder + visual projection
├── llava/                   # LLaVA codebase with the pre-LLM pruning hook
│   ├── model/llava_arch.py  # encode_images -> pruner.select() -> token mask
│   └── eval/
│       ├── infer.py         # benchmark inference (LLaVA-1.5 / LLaVA-NeXT)
│       ├── datasets.py      # benchmark loaders
│       ├── scorers.py       # local scorers (GQA, POPE, TextVQA, MME, SQA, VizWiz, MMBench)
│       └── score.py         # scoring CLI
├── scripts/
│   ├── eval.sh              # shared runner (multi-GPU chunking, merge, scoring)
│   ├── v1_5/eval/*.sh       # LLaVA-1.5 benchmarks
│   ├── v1_6/eval/*.sh       # LLaVA-NeXT benchmarks
│   ├── video/run_qa.sh      # Video-LLaVA video QA
│   ├── convert_vqav2_for_submission.py
│   ├── convert_vizwiz_for_submission.py
│   └── prepare_mme_answers.py
├── video_llava/             # Video-LLaVA integration + GPT-3.5 judge
├── EVAL.md                  # data preparation
└── requirements.txt
```

## 🔖 Citation

```bibtex
@inproceedings{shin2027stpruner,
  title     = {Don't Ask the LLM: Visual Token Pruning with Spectral Structure and Semantic Relevance},
  author    = {Shin, Intaek and Lim, Geuntaek and Park, Sungjune and Choi, Yukyung},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

## 👍 Acknowledgement

This code is built on [LLaVA](https://github.com/haotian-liu/LLaVA),
[VisPruner](https://github.com/Theia-4869/VisPruner) and
[Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA). We thank the authors for releasing
their code.

## 📄 License

This project is released under the [Apache 2.0 license](LICENSE).
