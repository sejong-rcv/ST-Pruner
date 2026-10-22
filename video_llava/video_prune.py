"""
ST-Pruner for Video-LLaVA-7B.

Video-LLaVA encodes 8 frames with the LanguageBind video tower:
    video tower output: [b, 8, 257, 1024]  (token 0 = [CLS], 256 patches)
Each frame is an independent visual group: its 256 patch tokens are scored and the
top `per_frame_keep` are kept (17 per frame -> 136 tokens in total), in their original
order, before the multimodal projector. LanguageBind is CLIP-style (text_model +
text_projection, visual_projection), so the semantic branch transfers directly.

The `videollava` package needs transformers 4.31, which is incompatible with this
repository's `llava` package, so the scoring functions are re-declared here. They
are identical to `stpruner/pruner.py`.

The pruner is injected by wrapping `model.encode_videos` (no change to Video-LLaVA).
"""

import torch
import torch.nn.functional as F


def _decompose(Z, rank, decomp):
    rank = min(rank, min(Z.shape))
    if decomp == "lowrank":
        U, S, _ = torch.svd_lowrank(Z, q=rank, niter=2)
        return U, S
    elif decomp == "full":
        U, S, _ = torch.linalg.svd(Z, full_matrices=False)
        return U[:, :rank], S[:rank]
    raise ValueError(f"Unknown decomposition: {decomp}")


def spectral_score(Z, rank, sv_power=0.5, decomp="lowrank"):
    """sum_{k<=R} sigma_k^p * U_ik^2 for one frame Z (N, d)."""
    U, S = _decompose(Z, rank, decomp)
    return (U.pow(2) * S.pow(sv_power).unsqueeze(0)).sum(dim=-1)


def minmax_norm(x):
    mn = x.min(dim=-1, keepdim=True).values
    mx = x.max(dim=-1, keepdim=True).values
    return (x - mn) / (mx - mn + 1e-8)


class LanguageBindTextEncoder:
    """Text tower + projections of the LanguageBind video model (frozen)."""

    def __init__(self, video_tower_name, cache_dir, device, dtype):
        from videollava.model.multimodal_encoder.languagebind import LanguageBindVideo
        from videollava.model.multimodal_encoder.languagebind.video.tokenization_video import (
            LanguageBindVideoTokenizer)
        full = LanguageBindVideo.from_pretrained(video_tower_name, cache_dir=cache_dir)
        self.text_model = full.text_model.to(device).to(dtype).eval()
        self.text_projection = full.text_projection.to(device).to(dtype).eval()
        self.visual_projection = full.visual_projection.to(device).to(dtype).eval()
        self.tokenizer = LanguageBindVideoTokenizer.from_pretrained(video_tower_name, cache_dir=cache_dir)
        self.device = device
        for p in list(self.text_model.parameters()) + list(self.text_projection.parameters()) \
                + list(self.visual_projection.parameters()):
            p.requires_grad_(False)

    @torch.no_grad()
    def encode_text(self, text):
        """Question -> (768,) text embedding (normalized when the similarity is computed)."""
        ti = self.tokenizer([text or ""], return_tensors="pt", padding=True, truncation=True, max_length=77)
        ti = {k: v.to(self.device) for k, v in ti.items()}
        pooled = self.text_model(**ti)[1]
        return self.text_projection(pooled)[0]

    @torch.no_grad()
    def project_patches(self, patch_feat):
        """(N, 1024) patch features -> (N, 768), L2-normalized."""
        v = self.visual_projection(patch_feat.to(self.visual_projection.weight.dtype))
        return F.normalize(v.float(), dim=-1)


@torch.no_grad()
def score_frame(patch_feat, method, rank, sv_power=0.5, decomp="lowrank", text_encoder=None, text_emb=None):
    """Token scores of one frame. patch_feat: (256, 1024) fp32 without [CLS].

    method: "stpruner" (spectral x inverted similarity) | "spectral" | "semantic"
    """
    if method in ("stpruner", "spectral"):
        s = minmax_norm(spectral_score(patch_feat, rank, sv_power, decomp).unsqueeze(0))[0]
        if method == "spectral":
            return s
    v = text_encoder.project_patches(patch_feat.to(text_encoder.device))
    t = F.normalize(text_emb.float().unsqueeze(0), dim=-1)
    rho = (v * t).sum(-1)
    r = minmax_norm((-rho).unsqueeze(0))[0]
    if method == "semantic":
        return r
    return r * s


def install_video_pruner(model, method, per_frame_keep, rank=0, sv_power=0.5, decomp="lowrank",
                         text_encoder=None):
    """Replace `model.encode_videos` with a version that prunes every frame before the projector.

    rank <= 0 uses R = N / 8 (32 for 256 patches). Call `model._st_set_question(q)` per sample.
    """
    get_video_tower = model.get_model().get_video_tower
    mm_projector = model.get_model().mm_projector
    state = {"question": ""}

    def set_question(text):
        state["question"] = text or ""
    model._st_set_question = set_question

    @torch.no_grad()
    def encode_videos_pruned(videos):                # videos: [b, c, t, h, w]
        vfeat = get_video_tower()(videos)            # [b, t, 257, 1024], token 0 = [CLS]
        B, T, N, C = vfeat.shape
        text_emb = text_encoder.encode_text(state["question"]) if method != "spectral" else None

        kept = []
        for b in range(B):
            frames = []
            for t in range(T):
                patch = vfeat[b, t, 1:]              # drop [CLS]: (256, 1024)
                if per_frame_keep >= patch.shape[0]:
                    frames.append(patch)
                    continue
                r = rank if rank > 0 else patch.shape[0] // 8
                sc = score_frame(patch.float(), method, r, sv_power, decomp, text_encoder, text_emb)
                idx = sc.topk(per_frame_keep).indices.sort().values   # keep original order
                frames.append(patch[idx])
            kept.append(torch.stack(frames, 0))      # (T, k, 1024)
        return mm_projector(torch.stack(kept, 0))    # (b, T, k, 4096)

    model.encode_videos = encode_videos_pruned
    return model
