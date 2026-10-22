"""
ST-Pruner: Spectral-Textual visual token pruner.

"Don't Ask the LLM: Visual Token Pruning with Spectral Structure and Semantic
Relevance" (WACV 2027).

For each visual group (an image, a LLaVA-NeXT view or a video frame) with patch
features Z in R^{N x d}, two token-wise scores are computed before the LLM:

  Spectral saliency (Sec. 3.2)
      Z ~= U_R Sigma_R V_R^T          (randomized rank-R SVD, R = N/8)
      s_bar_i = sum_{k<=R} sigma_k^{1/2} * U_ik^2
      s = minmax(s_bar)

  Semantic relevance (Sec. 3.3)
      rho_i = <phi(z_i), psi(T)>      (CLIP patch-text cosine similarity)
      r = minmax(-rho)                (inverted similarity)

  Dual-cue fusion (Sec. 3.4)
      g = s * r,   keep = TopK(g, K)

The kept indices are turned into a mask in `llava/model/llava_arch.py`, so the
surviving tokens keep their original order.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from llava.model.pruner import BasePruner


METHODS = ("stpruner", "spectral", "semantic")


# ----------------------------------------------------------------------------
# Spectral branch
# ----------------------------------------------------------------------------

def _decompose(Z, rank, decomp):
    """Z (N, d) -> leading left singular vectors U (N, R) and singular values S (R,)."""
    rank = min(rank, min(Z.shape))
    if decomp == "lowrank":
        # Randomized SVD: sketch size q = R, two power iterations, no oversampling
        U, S, _ = torch.svd_lowrank(Z, q=rank, niter=2)
        return U, S
    elif decomp == "full":
        # Exact SVD truncated to rank R (reference only, slower)
        U, S, _ = torch.linalg.svd(Z, full_matrices=False)
        return U[:, :rank], S[:rank]
    raise ValueError(f"Unknown decomposition: {decomp}")


def spectral_score(Z, rank, sv_power=0.5, decomp="lowrank"):
    """Raw spectral saliency of one group, Eq. (4): sum_k sigma_k^p * U_ik^2 (p = 0.5)."""
    U, S = _decompose(Z, rank, decomp)
    return (U.pow(2) * S.pow(sv_power).unsqueeze(0)).sum(dim=-1)


def _projector_forward(image_features, mm_projector):
    """ViT features (B, N, 1024) -> LVLM projector outputs (B, N, 4096) in fp32.

    Only used by the feature-space ablation (`feat_space="projector"`, Table S3).
    """
    linears = [m for m in mm_projector.modules() if isinstance(m, nn.Linear)]
    acts = [m for m in mm_projector.modules() if isinstance(m, (nn.GELU, nn.ReLU, nn.SiLU))]
    W1, W2 = linears[0].weight.float(), linears[1].weight.float()
    b1, b2 = linears[0].bias.float(), linears[1].bias.float()
    act = acts[0] if acts else nn.GELU()
    with torch.no_grad():
        return F.linear(act(F.linear(image_features.float(), W1, b1)), W2, b2)


# ----------------------------------------------------------------------------
# Semantic branch
# ----------------------------------------------------------------------------

def patch_text_similarity(image_features, text_emb, vis_proj_fn):
    """Cosine similarity between projected patches and the text query, Eq. (7). Returns (B, N)."""
    v = vis_proj_fn(image_features, image_features.device)    # (B, N, 768), L2-normalized
    t = F.normalize(text_emb.float(), dim=-1)                  # (B, 768)
    return (v * t.unsqueeze(1)).sum(dim=-1)


def minmax_norm(x):
    """Min-max normalization over the token axis, Eq. (5)."""
    mn = x.min(dim=-1, keepdim=True).values
    mx = x.max(dim=-1, keepdim=True).values
    return (x - mn) / (mx - mn + 1e-8)


# ----------------------------------------------------------------------------
# ST-Pruner
# ----------------------------------------------------------------------------

class STPruner(BasePruner):
    """Spectral-Textual pruner.

    Args:
        keep_num:      tokens kept per visual group (K).
        text_encoder:  `stpruner.CLIPTextEncoder`, needed by "stpruner" and "semantic".
        method:        "stpruner" (s * r, ours) | "spectral" (s only) | "semantic" (r only).
        rank:          spectral rank R. None or <= 0 uses R = N / 8 (72 for 576 tokens).
        sv_power:      singular-value exponent p in sigma^p * U^2 (0.5 in the paper, Table S4).
        feat_space:    "vit" (pre-projector ViT features, default) | "projector" (Table S3).
        mm_projector:  LVLM projector, only needed for feat_space="projector".
        semantic_sign: "neg" (-sim, default) | "pos" (+sim). Only affects method="semantic" (Table 5).
        decomp:        "lowrank" (randomized SVD, default) | "full" (exact SVD).
    """

    def __init__(self, keep_num, text_encoder=None, method="stpruner", rank=None,
                 sv_power=0.5, feat_space="vit", mm_projector=None,
                 semantic_sign="neg", decomp="lowrank"):
        super().__init__(keep_num)
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}. Choose from {METHODS}")
        if method != "spectral" and text_encoder is None:
            raise ValueError(f"method={method} requires a text encoder")
        if feat_space == "projector" and mm_projector is None:
            raise ValueError("feat_space=projector requires mm_projector")
        self.text_encoder = text_encoder
        self.method = method
        self.rank = rank
        self.sv_power = sv_power
        self.feat_space = feat_space
        self.mm_projector = mm_projector
        self.semantic_sign = semantic_sign
        self.decomp = decomp
        self._question = ""

    def set_question(self, text):
        """Set the textual query T of the current sample."""
        self._question = text or ""

    def spectral_branch(self, image_features):
        """Normalized spectral saliency s, Eq. (6). Returns (B, N)."""
        if self.feat_space == "vit":
            X = image_features.float()
        else:
            X = _projector_forward(image_features, self.mm_projector)
        N = X.shape[1]
        rank = self.rank if self.rank and self.rank > 0 else N // 8
        with torch.no_grad():
            s = torch.stack([spectral_score(X[b], rank, self.sv_power, self.decomp)
                             for b in range(X.shape[0])], dim=0)
        return minmax_norm(s)

    def semantic_branch(self, image_features):
        """Normalized inverted patch-text relevance r, Eq. (8). Returns (B, N)."""
        B = image_features.shape[0]
        with torch.no_grad():
            t = self.text_encoder.encode_text(self._question, image_features.device).expand(B, -1)
        rho = patch_text_similarity(image_features, t, self.text_encoder.project_patches)
        if self.method == "semantic" and self.semantic_sign == "pos":
            return minmax_norm(rho)
        return minmax_norm(-rho)

    def select(self, image_features, image_attentions=None):
        if self.method == "spectral":
            scores = self.spectral_branch(image_features)
        elif self.method == "semantic":
            scores = self.semantic_branch(image_features)
        else:
            r = self.semantic_branch(image_features)
            s = self.spectral_branch(image_features)
            scores = r * s                       # Eq. (9), parameter-free fusion
        return scores.topk(self.keep_num, dim=-1).indices   # Eq. (10)
