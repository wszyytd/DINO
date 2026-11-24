"""Slot-competitive Writer with the exact parameter layout of the original CrossBlock."""

import torch
from torch.nn import functional as F

from .local_attention import CrossBlock


class CompetitiveWriter(CrossBlock):
    def distribute(self, query, memory, valid):
        """Per-head assignment (over slots), weighted mean (over tokens), projected values."""
        attention = self.attention
        b, n, d = memory.shape
        heads = attention.num_heads
        width = d // heads
        query = self.norm(query)
        memory = memory.masked_fill(~valid[..., None], 0)
        qw, kw, vw = attention.in_proj_weight.chunk(3)
        qb, kb, vb = attention.in_proj_bias.chunk(3)
        q = F.linear(query, qw, qb).reshape(b, -1, heads, width).transpose(1, 2)
        k = F.linear(memory, kw, kb).reshape(b, n, heads, width).transpose(1, 2)
        v = F.linear(memory, vw, vb).reshape(b, n, heads, width).transpose(1, 2)
        logits = (q * width ** -0.5) @ k.transpose(-1, -2)
        # Keep the normalizations in at least float32 even for low-precision projections.
        stable = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
        assignment = stable.softmax(dim=-2).masked_fill(~valid[:, None, None, :], 0)
        weights = assignment / assignment.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return assignment, weights, v

    def forward(self, query, memory, valid, need_weights=False):
        _, weights, values = self.distribute(query, memory, valid)
        pooled = (weights.to(values.dtype) @ values).transpose(1, 2).reshape_as(query)
        x = query + self.attention.out_proj(pooled)
        return x + self.ffn(self.ffn_norm(x)), weights.mean(1) if need_weights else None
