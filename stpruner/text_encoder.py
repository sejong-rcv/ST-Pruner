"""
Frozen CLIP text encoder and visual projection for the semantic branch.

LLaVA-1.5 / LLaVA-NeXT use the CLIP ViT-L/14-336 vision tower, so the paired CLIP
text tower and `visual_projection` map the question and the pre-projector patch
features into the shared 768-d embedding space (psi and phi in Sec. 3.3). These are
separate from the LVLM's multimodal projector and are only used for scoring.
"""

import torch
import torch.nn.functional as F


def _clip_name_from_vision_tower(vision_tower):
    for name in (getattr(vision_tower, "vision_tower_name", None),
                 getattr(getattr(vision_tower, "config", None), "_name_or_path", None)):
        if name:
            return name
    return "openai/clip-vit-large-patch14-336"


class CLIPTextEncoder:
    """Wraps a full `CLIPModel` loaded from the LVLM's vision tower checkpoint.

    Args:
        name_or_path: CLIP checkpoint (HF id or local path). If None, it is read
                      from `vision_tower`.
        vision_tower: LLaVA vision tower module.
    """

    def __init__(self, name_or_path=None, vision_tower=None):
        from transformers import CLIPModel, CLIPTokenizer

        name = name_or_path or _clip_name_from_vision_tower(vision_tower)
        print(f"[ST-Pruner] Loading CLIP text encoder: {name}")
        self.model = CLIPModel.from_pretrained(name).eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(name)

    def encode_text(self, text, device):
        """Query -> EOS text embedding q = psi(T), (1, 768), L2-normalized."""
        with torch.no_grad():
            inputs = self.tokenizer([text], padding=True, truncation=True,
                                    max_length=77, return_tensors="pt").to(device)
            self.model.to(device)
            t = self.model.get_text_features(**inputs)
            return F.normalize(t.float(), dim=-1)

    def project_patches(self, feat, device):
        """Pre-projector patch features (B, N, 1024) -> v_i = phi(z_i), (B, N, 768), L2-normalized."""
        with torch.no_grad():
            self.model.to(device)
            B, N, D = feat.shape
            v = self.model.visual_projection(feat.float().view(B * N, D)).view(B, N, -1)
            return F.normalize(v, dim=-1)
