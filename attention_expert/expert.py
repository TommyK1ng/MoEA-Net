"""Spatiotemporal-aware expert for paired retinal images."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .modules import AttentionPooling


class SpatiotemporalExpert(nn.Module):
    """Map paired patch features [B, M, 2, P, D] to [B, M, 3, D]."""

    def __init__(self, emb=768, heads=8, temperature=0.6):
        super().__init__()
        if emb <= 0 or heads <= 0 or emb % heads != 0:
            raise ValueError("Embedding dimension must be divisible by the number of heads.")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be finite and positive.")

        self.emb = emb
        self.heads = heads
        self.temperature = temperature
        self.spatial_attention = nn.MultiheadAttention(emb, heads, batch_first=True)
        self.temporal_attention = nn.MultiheadAttention(emb, heads, batch_first=True)
        self.diagnostic_pooling = AttentionPooling(emb)

    def _spatial_features(self, x):
        """Apply shared self-attention over patches for before/after images."""
        batch, modalities, _, patches, channels = x.shape
        paired = x.reshape(batch * modalities * 2, patches, channels)
        features, weights = self.spatial_attention(
            paired, paired, paired,
            need_weights=True,
            average_attn_weights=False,
        )
        return (
            features.reshape(batch, modalities, 2, patches, channels),
            weights.reshape(batch, modalities, 2, self.heads, patches, patches),
        )

    def _temporal_features(self, before, after):
        # The difference token queries the before/after history to model treatment change.
        difference = (after - before).reshape(-1, 1, self.emb)
        history = torch.stack((before, after), dim=-2).reshape(-1, 2, self.emb)
        changes, _ = self.temporal_attention(
            difference, history, history, need_weights=False
        )
        return changes.reshape_as(before)

    def forward(self, x):
        if (
            not isinstance(x, torch.Tensor)
            or x.ndim != 5
            or x.shape[2] != 2
            or x.shape[3] == 0
            or x.shape[-1] != self.emb
        ):
            raise ValueError("Expected paired features with shape [B, M, 2, P, D] and P > 0.")

        batch, modalities, _, patches, _ = x.shape
        spatial_features, attention = self._spatial_features(x)
        before, after = spatial_features.unbind(dim=2)
        before_global = self.diagnostic_pooling(before)
        after_global = self.diagnostic_pooling(after)

        changes = self._temporal_features(before, after)
        # Give larger weights to patches whose features changed more over time.
        similarity = F.cosine_similarity(before, after, dim=-1)
        patch_weights = F.softmax(-similarity / self.temperature, dim=-1)
        change_global = (changes * patch_weights.unsqueeze(-1)).sum(dim=-2)

        # Output order must match the three image placeholders per modality.
        features = torch.stack((before_global, after_global, change_global), dim=2)
        before_attention = attention[:, :, 0].reshape(
            batch * modalities, self.heads, patches, patches
        )
        after_attention = attention[:, :, 1].reshape(
            batch * modalities, self.heads, patches, patches
        )
        return features, before_attention, after_attention, patch_weights
