"""Learned reference-conditioned crop matching, independent of the frozen backbone."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class SpatialAdapter(nn.Module):
    def __init__(self, input_dim, dim, layers):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(input_dim, dim) for _ in range(layers)])
        self.layer_weights = nn.Parameter(torch.zeros(layers))
        self.position = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, features, valid):
        if features.ndim != 4 or valid.shape != (features.shape[0], features.shape[2]):
            raise ValueError('features must be [B,L,N,C] and validity [B,N]')
        if features.shape[1] != len(self.projections):
            raise ValueError('cached feature layer count does not match model')
        side = math.isqrt(features.shape[2])
        if side * side != features.shape[2] or not valid.any(dim=1).all():
            raise ValueError('need a square patch grid and nonempty valid content')
        # Remove padding before learned transforms, including extremely large invalid values.
        clean = features.masked_fill(~valid[:, None, :, None], 0)
        weights = self.layer_weights.softmax(0)
        x = sum(w * linear(clean[:, i])
                for i, (w, linear) in enumerate(zip(weights, self.projections, strict=True)))
        axis = torch.linspace(-1, 1, side, device=x.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        pos = torch.stack((xx, yy), -1).reshape(1, side * side, 2)
        x = x + self.position(pos)
        x = x + self.mlp(self.norm(x))
        return x.masked_fill(~valid[..., None], 0)


class CrossBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, query, memory, valid, need_weights=False):
        value, weights = self.attention(self.norm(query), memory, memory,
                                        key_padding_mask=~valid, need_weights=need_weights)
        x = query + value
        return x + self.ffn(self.ffn_norm(x)), weights


class LocalMatcher(nn.Module):
    def __init__(self, input_dim=384, dim=128, layers=4, heads=4,
                 probes=4, reference_grid=8, blocks=2):
        super().__init__()
        if min(input_dim, dim, layers, heads, probes, reference_grid, blocks) < 1:
            raise ValueError('all model sizes must be positive')
        if dim % heads:
            raise ValueError('hidden dimension must be divisible by attention heads')
        self.config = dict(input_dim=input_dim, dim=dim, layers=layers, heads=heads,
                           probes=probes, reference_grid=reference_grid, blocks=blocks)
        self.reference_adapter = SpatialAdapter(input_dim, dim, layers)
        self.candidate_adapter = SpatialAdapter(input_dim, dim, layers)
        self.reference_grid = reference_grid
        self.probes = nn.Parameter(torch.randn(1, probes, dim) * 0.02)
        self.reference_pool = CrossBlock(dim, heads)
        self.correlation = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([CrossBlock(dim, heads) for _ in range(blocks)])
        self.candidate_pool = CrossBlock(dim, heads)
        self.classifier = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim),
                                         nn.GELU(), nn.Linear(dim, 1))

    def pool_reference(self, x, valid):
        b, n, d = x.shape
        side = math.isqrt(n)
        size = min(self.reference_grid, side)
        weights = valid.reshape(b, 1, side, side).to(x.dtype)
        content = x.transpose(1, 2).reshape(b, d, side, side) * weights
        coverage = F.adaptive_avg_pool2d(weights, (size, size))
        pooled = F.adaptive_avg_pool2d(content, (size, size)) / coverage.clamp_min(1e-6)
        return pooled.flatten(2).transpose(1, 2), coverage.flatten(1) > 0

    def forward(self, reference, reference_valid, candidate, candidate_valid):
        reference_valid, candidate_valid = reference_valid.bool(), candidate_valid.bool()
        r = self.reference_adapter(reference, reference_valid)
        q = self.candidate_adapter(candidate, candidate_valid)
        r, rvalid = self.pool_reference(r, reference_valid)
        descriptions, _ = self.reference_pool(self.probes.expand(r.shape[0], -1, -1), r, rvalid)
        similarity = F.normalize(q, dim=-1) @ F.normalize(r, dim=-1).transpose(-1, -2)
        similarity = similarity.masked_fill(~rvalid[:, None, :], -torch.inf)
        maximum = similarity.max(-1).values
        smooth = torch.logsumexp(similarity, -1) - rvalid.sum(-1).log()[:, None]
        global_sim = (F.normalize(q, dim=-1)
                      * F.normalize(descriptions.mean(1), dim=-1)[:, None]).sum(-1)
        q = q + self.correlation(torch.stack((maximum, smooth, global_sim), -1))
        for block in self.blocks:
            q, _ = block(q, r, rvalid)
        observed, attention = self.candidate_pool(descriptions, q, candidate_valid, True)
        combined = torch.cat((descriptions, observed, descriptions * observed,
                              (descriptions - observed).abs()), -1).mean(1)
        return {'logit': self.classifier(combined).squeeze(-1),
                'query_attention': attention}
