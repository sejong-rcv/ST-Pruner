"""
Pre-LLM visual token pruning interface.

A pruner receives the vision-tower outputs of every visual group (an image, a
LLaVA-NeXT view, ...) and returns the indices of the tokens to keep.
`LlavaMetaForCausalLM.encode_images` turns those indices into a boolean mask, and
only the masked tokens are passed to the LLM, in their original order.

The ST-Pruner implementation lives in `stpruner/pruner.py`.
"""

import torch
from abc import ABC, abstractmethod


class BasePruner(ABC):
    """Base class of all pruners.

    Args:
        keep_num: number of visual tokens kept per visual group.
    """

    def __init__(self, keep_num: int):
        self.keep_num = keep_num

    @abstractmethod
    def select(
        self,
        image_features: torch.Tensor,    # (B, N, C) vision-tower patch features
        image_attentions: torch.Tensor,  # (B, H, N) [CLS]-to-patch attention (unused by ST-Pruner)
    ) -> torch.Tensor:
        """Return keep_indices, a (B, K) LongTensor."""
        pass


class NoPruner(BasePruner):
    """No pruning: all N tokens are passed to the LLM (full-token baseline)."""

    def select(self, image_features, image_attentions):
        B, N, C = image_features.shape
        return torch.arange(N, device=image_features.device).unsqueeze(0).expand(B, -1)
