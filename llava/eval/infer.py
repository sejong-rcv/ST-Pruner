"""
Benchmark inference with ST-Pruner for LLaVA-1.5 and LLaVA-NeXT (LLaVA-1.6).

Example (run from the repository root):
    python -m llava.eval.infer \
        --model-path liuhaotian/llava-v1.5-7b \
        --benchmark pope \
        --method stpruner --visual-tokens 64 \
        --answers-file results/llava-v1.5-7b/pope/stpruner_64.jsonl

Token budget (`--visual-tokens`):
    LLaVA-1.5  : tokens kept out of 576 (192 / 128 / 64).
    LLaVA-NeXT : total budget for a five-view input (640 / 320 / 160 out of 2880).
                 Each 576-token view keeps visual_tokens / 5 tokens (128 / 64 / 32),
                 so inputs with fewer views keep fewer tokens in total.

The answers are written as LLaVA-style jsonl ({"question_id", "prompt", "text", ...});
score them with `python -m llava.eval.score`.
"""

import os
import json
import argparse

import torch
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.model.pruner import NoPruner
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.eval.datasets import BENCHMARKS, load_image, split_chunk

from stpruner import STPruner, CLIPTextEncoder

NEXT_MAX_VIEWS = 5   # LLaVA-NeXT: up to 4 grid crops + 1 global view of 576 tokens each


def build_parser():
    parser = argparse.ArgumentParser(description="ST-Pruner inference (LLaVA-1.5 / LLaVA-NeXT)")
    # model / data
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--vision-tower", type=str, default=None,
                        help="override the CLIP vision tower path stored in the checkpoint config")
    parser.add_argument("--benchmark", type=str, required=True, choices=list(BENCHMARKS.keys()))
    parser.add_argument("--data-dir", type=str, default="./playground/data/eval")
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0, help="debug: only run the first N samples")
    # generation
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0, help="seed for the randomized SVD")
    # pruning
    parser.add_argument("--method", type=str, default="stpruner",
                        choices=["stpruner", "spectral", "semantic", "none"],
                        help="stpruner: spectral x semantic (ours) | spectral: spectral-only | "
                             "semantic: semantic-only | none: full-token baseline")
    parser.add_argument("--visual-tokens", type=int, default=None,
                        help="retained visual tokens (LLaVA-1.5: 192/128/64, LLaVA-NeXT: 640/320/160)")
    parser.add_argument("--rank", type=int, default=0, help="spectral rank R (0: R = N/8 = 72)")
    parser.add_argument("--sv-power", type=float, default=0.5, help="p in sigma^p * U^2 (paper: 0.5)")
    parser.add_argument("--feat-space", type=str, default="vit", choices=["vit", "projector"])
    parser.add_argument("--decomp", type=str, default="lowrank", choices=["lowrank", "full"])
    parser.add_argument("--semantic-sign", type=str, default="neg", choices=["neg", "pos"],
                        help="sign of the similarity for --method semantic (-sim / +sim)")
    return parser


def build_pruner(args, model):
    """Create the pruner and return (pruner, tokens kept per visual group)."""
    if args.method == "none":
        return NoPruner(keep_num=576), None

    if args.visual_tokens is None:
        raise ValueError("--visual-tokens is required unless --method none")
    is_next = getattr(model.config, "image_aspect_ratio", "pad") == "anyres"
    if is_next:
        if args.visual_tokens % NEXT_MAX_VIEWS != 0:
            raise ValueError(f"LLaVA-NeXT --visual-tokens must be divisible by {NEXT_MAX_VIEWS} (e.g. 640/320/160)")
        keep = args.visual_tokens // NEXT_MAX_VIEWS
        print(f"[ST-Pruner] LLaVA-NeXT: {keep} tokens per view (<= {args.visual_tokens} for {NEXT_MAX_VIEWS} views)")
    else:
        keep = args.visual_tokens
        print(f"[ST-Pruner] {keep} visual tokens per image")

    vision_tower = model.get_model().get_vision_tower()
    text_encoder = None if args.method == "spectral" else CLIPTextEncoder(vision_tower=vision_tower)
    pruner = STPruner(
        keep, text_encoder=text_encoder, method=args.method, rank=args.rank,
        sv_power=args.sv_power, feat_space=args.feat_space,
        mm_projector=model.get_model().mm_projector,
        semantic_sign=args.semantic_sign, decomp=args.decomp,
    )
    return pruner, keep


def main():
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, args.model_base, model_name,
        pruner=NoPruner(keep_num=576), vision_tower_path=args.vision_tower,
    )
    pruner, keep = build_pruner(args, model)
    model.set_pruner(pruner)

    rows = BENCHMARKS[args.benchmark](args.data_dir)
    rows = split_chunk(rows, args.num_chunks, args.chunk_idx)
    if args.max_samples > 0:
        rows = rows[:args.max_samples]

    os.makedirs(os.path.dirname(os.path.abspath(args.answers_file)), exist_ok=True)
    ans_file = open(args.answers_file, "w")

    for row in tqdm(rows, total=len(rows)):
        image = load_image(row)
        if image is not None:
            image_tensor = process_images([image], image_processor, model.config)[0]
            # LLaVA-1.5 (pad): (3, 336, 336) -> add batch dim. LLaVA-NeXT (anyres): (views, 3, 336, 336)
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(dtype=torch.float16, device="cuda")
            image_sizes = [image.size]
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + row["question"]
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + row["question"]
        else:
            image_tensor, image_sizes, qs = None, None, row["question"]

        # [ST-Pruner] The semantic branch scores patches against the full question text
        if hasattr(pruner, "set_question"):
            pruner.set_question(row["question"])

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if image is not None:
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        else:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        with torch.inference_mode():
            output_ids, _ = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=(args.temperature > 0),
                temperature=args.temperature,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if image is not None:
            image.close()

        ans_file.write(json.dumps({
            "question_id":   row["id"],
            "prompt":        row["question"],
            "text":          output,
            "category":      row["category"],
            "method":        args.method,
            "visual_tokens": args.visual_tokens if args.method != "none" else None,
            "keep_per_group": keep,
        }) + "\n")
        ans_file.flush()

    ans_file.close()
    print(f"Saved: {args.answers_file}")


if __name__ == "__main__":
    main()
