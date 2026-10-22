# Evaluation Data

The image benchmarks follow the [LLaVA-1.5 evaluation](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md)
layout under `./playground/data/eval` (override with `DATA_DIR=...`).

**First download LLaVA's [`eval.zip`](https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view?usp=sharing)**
and extract it to `./playground/data/eval`. It contains the question files (`llava_*.jsonl`) and the
folder structure. Then add the images and annotations of each benchmark as listed below.

```bash
mkdir -p playground/data
unzip eval.zip -d playground/data      # -> playground/data/eval/{vqav2,gqa,vizwiz,...}
```

## Image benchmarks

### VQAv2 (test-dev)
- Images: [COCO test2015](http://images.cocodataset.org/zips/test2015.zip) → `vqav2/test2015/`
- Questions: `vqav2/llava_vqav2_mscoco_test-dev2015.jsonl`, `vqav2/llava_vqav2_mscoco_test2015.jsonl` (in `eval.zip`)
- Scoring: `vqav2.sh` writes `results/<model>/vqav2/upload/<method>_<tokens>.json`. Submit it to the
  [EvalAI VQAv2 challenge](https://eval.ai/web/challenges/challenge-page/830/my-submission) (test-dev phase).

### GQA (testdev-balanced, 12,578 questions)
- Images: [GQA images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip) → `gqa/data/images/`
- Ground truth: `testdev_balanced_questions.json` from [GQA questions1.2](https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip) → `gqa/testdev_balanced_questions.json`
- Questions: `gqa/llava_gqa_testdev_balanced.jsonl` (in `eval.zip`)
- Metric: exact-match accuracy.

### VizWiz (test-dev, 8,000 questions)
- Images: [VizWiz test](https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip) → `vizwiz/test/`
- Questions: `vizwiz/llava_test.jsonl` (in `eval.zip`)
- Scoring: `vizwiz.sh` always writes the EvalAI submission file `results/<model>/vizwiz/upload/<method>_<tokens>.json`
  ([EvalAI VizWiz challenge](https://eval.ai/web/challenges/challenge-page/2185/my-submission)).
  If `vizwiz/vizwiz_answer.json` is present (a list of VizWiz-format annotations aligned with
  `question_id`, each with 10 `answers`), it also reports the local VQA accuracy.

### ScienceQA (test)
- From the [ScienceQA repo](https://github.com/lupantech/ScienceQA) (`data/scienceqa`): `images/test/`,
  `problems.json`, `pid_splits.json` → `scienceqa/`
- Questions: `scienceqa/llava_test_CQM-A.json` (in `eval.zip`)
- Metric: answer accuracy over all 4,241 test questions; text-only questions are answered without an
  image. The scorer also prints the accuracy on the 2,017 image questions (IMG accuracy).

### TextVQA (val, 5,000 questions)
- Images: [TextVQA train/val images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip) → `textvqa/train_images/`
- Ground truth: [`TextVQA_0.5.1_val.json`](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json) → `textvqa/`
- Questions: `textvqa/llava_textvqa_val_v051_ocr.jsonl` (in `eval.zip`)
- Metric: TextVQA soft accuracy (10 reference answers).

### POPE (8,910 questions)
- Images: [COCO val2014](http://images.cocodataset.org/zips/val2014.zip) → `pope/val2014/`
- Annotations: `coco_pope_{random,popular,adversarial}.json` from
  [POPE](https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco) → `pope/coco/`
- Questions: `pope/llava_pope_test.jsonl` (in `eval.zip`)
- Metric: F1 averaged over the three sampling strategies.

### MME
- Images: the official MME release (see [MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)) → `MME/MME_Benchmark_release_version/`
- Questions: `MME/llava_mme.jsonl` (in `eval.zip`)
- Ground truth: build `MME/mme_answers.json` once with
  ```bash
  python scripts/prepare_mme_answers.py --out ./playground/data/eval/MME/mme_answers.json
  ```
  (extracted from [lmms-lab/MME](https://huggingface.co/datasets/lmms-lab/MME), same 2,374 questions).
- Metric: Perception score (acc + acc+ summed over the 10 perception subtasks). The Cognition score is also printed.

### MMBench (dev)
- [`mmbench_dev_20230712.tsv`](https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv) → `mmbench/`
  (optional CN: [`mmbench_dev_cn_20231003.tsv`](https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_cn_20231003.tsv), benchmark `mmbench_cn`)
- Metric: CircularEval accuracy computed locally from the dev answers. A question counts as correct only
  if all of its circularly shifted copies are answered correctly.

### Final layout

```
playground/data/eval
├── vqav2      ├── test2015/  ├── llava_vqav2_mscoco_test-dev2015.jsonl  └── llava_vqav2_mscoco_test2015.jsonl
├── gqa        ├── data/images/  ├── llava_gqa_testdev_balanced.jsonl  └── testdev_balanced_questions.json
├── vizwiz     ├── test/  ├── llava_test.jsonl  └── (vizwiz_answer.json)
├── scienceqa  ├── images/test/  ├── llava_test_CQM-A.json  ├── problems.json  └── pid_splits.json
├── textvqa    ├── train_images/  ├── llava_textvqa_val_v051_ocr.jsonl  └── TextVQA_0.5.1_val.json
├── pope       ├── val2014/  ├── coco/coco_pope_*.json  └── llava_pope_test.jsonl
├── MME        ├── MME_Benchmark_release_version/  ├── llava_mme.jsonl  └── mme_answers.json
└── mmbench    └── mmbench_dev_20230712.tsv
```

## Video benchmarks (Video-LLaVA)

Download `MSVD_Zero_Shot_QA`, `MSRVTT_Zero_Shot_QA` and `TGIF_Zero_Shot_QA` from the
[Video-LLaVA data page](https://github.com/PKU-YuanGroup/Video-LLaVA/blob/main/TRAIN_AND_VALIDATE.md#data-for-validating)
(originally from Video-ChatGPT) and place them under `./playground/data/eval/GPT_Zero_Shot_QA/`:

```
playground/data/eval/GPT_Zero_Shot_QA
├── MSVD_Zero_Shot_QA    ├── videos/      ├── test_q.json  └── test_a.json
├── MSRVTT_Zero_Shot_QA  ├── videos/all/  ├── test_q.json  └── test_a.json
└── TGIF_Zero_Shot_QA    ├── mp4/         ├── test_q.json  └── test_a.json
```

Answers are open-ended and judged by GPT-3.5-turbo with the Video-ChatGPT protocol used by Video-LLaVA
(accuracy = yes / (yes + no)). See [`video_llava/README.md`](video_llava/README.md).
