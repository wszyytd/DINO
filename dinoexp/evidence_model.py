"""Set memory, real-observation matcher, auxiliary decoder and direct action value."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .local_attention import CrossBlock, SpatialAdapter


class EvidenceModel(nn.Module):
    def __init__(self, input_dim=384, dim=128, layers=4, heads=4,
                 memory_tokens=8, blocks=2):
        super().__init__()
        if min(input_dim, dim, layers, heads, memory_tokens, blocks) < 1 or dim % heads:
            raise ValueError('invalid model dimensions')
        self.config = dict(input_dim=input_dim, dim=dim, layers=layers, heads=heads,
                           memory_tokens=memory_tokens, blocks=blocks)
        self.dim = dim
        self.observation_adapter = SpatialAdapter(input_dim, dim, layers)
        self.reference_adapter = SpatialAdapter(input_dim, dim, layers)
        self.observation_queries = nn.Parameter(torch.randn(1, 4, dim) * .02)
        self.reference_queries = nn.Parameter(torch.randn(1, 4, dim) * .02)
        self.observation_read = CrossBlock(dim, heads)
        self.reference_read = CrossBlock(dim, heads)
        self.pose_embed = nn.Sequential(nn.Linear(5, dim), nn.GELU(), nn.Linear(dim, dim))
        self.quality_embed = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.token_types = nn.Parameter(torch.randn(1, 1, 5, dim) * .02)
        self.memory_queries = nn.Parameter(torch.randn(1, memory_tokens, dim) * .02)
        self.empty_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.memory_blocks = nn.ModuleList([CrossBlock(dim, heads) for _ in range(blocks)])
        self.match_read = CrossBlock(dim, heads)
        self.classifier = nn.Sequential(nn.LayerNorm(dim*4), nn.Linear(dim*4, dim),
                                        nn.GELU(), nn.Linear(dim, 1))
        self.prediction_queries = nn.Parameter(torch.randn(1, 5, dim) * .02)
        self.prediction_read = CrossBlock(dim, heads)
        self.prediction_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, input_dim))

    def encode_observations(self, features, valid):
        tokens = self.observation_adapter(features, valid)
        local, _ = self.observation_read(
            self.observation_queries.expand(len(tokens), -1, -1), tokens, valid)
        global_token = tokens.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        return torch.cat((global_token[:, None], local), dim=1)

    def encode_reference(self, features, valid):
        tokens = self.reference_adapter(features, valid)
        return self.reference_read(
            self.reference_queries.expand(len(tokens), -1, -1), tokens, valid)[0]

    def aggregate(self, z, poses, quality, valid):
        if z.ndim != 4 or z.shape[2:] != (5, self.dim):
            raise ValueError('observations must be [B,H,5,D]')
        if valid.shape != z.shape[:2] or poses.shape != (*z.shape[:2], 5):
            raise ValueError('history mask/pose shape mismatch')
        if quality.shape != (*z.shape[:2], 3):
            raise ValueError('quality must be [B,H,3]')
        valid = valid.bool()
        z = z.masked_fill(~valid[..., None, None], 0)
        poses = poses.masked_fill(~valid[..., None], 0)
        quality = quality.masked_fill(~valid[..., None], 0)
        context = z + self.pose_embed(poses)[:, :, None] + self.token_types
        context = context + self.quality_embed(quality)[:, :, None]
        context = context.flatten(1, 2)
        mask = valid[..., None].expand(-1, -1, 5).flatten(1)
        context = torch.cat((context, self.empty_token.expand(len(z), -1, -1)), dim=1)
        mask = torch.cat((mask, ~valid.any(1, keepdim=True)), dim=1)
        memory = self.memory_queries.expand(len(z), -1, -1)
        for block in self.memory_blocks:
            memory, _ = block(memory, context, mask)
        return memory

    def match(self, q, memories):
        batch, candidates, slots, dim = memories.shape
        ref = q[:, None].expand(-1, candidates, -1, -1).reshape(batch*candidates, -1, dim)
        memory = memories.reshape(batch*candidates, slots, dim)
        observed, _ = self.match_read(ref, memory, torch.ones(
            memory.shape[:2], device=memory.device, dtype=torch.bool))
        combined = torch.cat((ref, observed, ref*observed, (ref-observed).abs()), -1).mean(1)
        return self.classifier(combined).reshape(batch, candidates)

    def predict(self, memory, poses):
        queries = self.prediction_queries + self.pose_embed(poses)[:, None]
        result, _ = self.prediction_read(queries, memory, torch.ones(
            memory.shape[:2], device=memory.device, dtype=torch.bool))
        return self.prediction_out(result)


class ActionValue(nn.Module):
    def __init__(self, dim=128, heads=4):
        super().__init__()
        self.config = dict(dim=dim, heads=heads)
        self.candidate = nn.Sequential(nn.Linear(dim+3, dim), nn.GELU(), nn.LayerNorm(dim))
        self.action = nn.Sequential(nn.Linear(dim*2+7, dim), nn.GELU(), nn.LayerNorm(dim))
        self.read = CrossBlock(dim, heads)
        self.output = nn.Sequential(nn.LayerNorm(dim*4), nn.Linear(dim*4, dim),
                                     nn.GELU(), nn.Linear(dim, 1))
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, q, memories, scores, counts, subjects, poses, candidate_mask=None):
        """Only present state and action geometry; no true target/future feature argument."""
        means = memories.mean(2)
        batch, candidates, dim = means.shape
        if candidate_mask is None:
            candidate_mask = torch.ones((batch, candidates), device=means.device, dtype=torch.bool)
        if not candidate_mask.any(1).all() or subjects.min() < 0 or subjects.max() >= candidates:
            raise ValueError('invalid action candidate or candidate mask')
        if not candidate_mask.gather(1, subjects).all():
            raise ValueError('action targets a padded candidate')
        scalars = torch.stack((scores, counts / 11., (counts > 0).float()), -1)
        context = self.candidate(torch.cat((means, scalars), -1))
        selected = means.gather(1, subjects[..., None].expand(-1, -1, dim))
        own_score = scores.gather(1, subjects)[..., None]
        own_count = counts.gather(1, subjects)[..., None] / 11.
        reference = q.mean(1)[:, None].expand(-1, subjects.shape[1], -1)
        queries = self.action(torch.cat((reference, selected, poses, own_score, own_count), -1))
        read, _ = self.read(queries, context, candidate_mask)
        combined = torch.cat((queries, read, queries*read, (queries-read).abs()), -1)
        return 4 * torch.tanh(self.output(combined).squeeze(-1))


def teacher_targets(features, valid, cls):
    side = math.isqrt(features.shape[2])
    if side*side != features.shape[2] or side < 2:
        raise ValueError('teacher features require a square grid of side >=2')
    mask = valid.reshape(-1, 1, side, side).float()
    last = features[:, -1].masked_fill(~valid[..., None], 0)
    spatial = last.transpose(1, 2).reshape(len(features), -1, side, side)
    coverage = F.adaptive_avg_pool2d(mask, 2)
    regions = F.adaptive_avg_pool2d(spatial, 2) / coverage.clamp_min(1e-6)
    regions = regions.flatten(2).transpose(1, 2)
    target = torch.cat((cls[:, None], regions), 1).detach()
    region_valid = coverage.flatten(1) > 0
    target_valid = torch.cat((region_valid.any(1, keepdim=True), region_valid), 1)
    return target, target_valid


def prediction_loss(prediction, target, valid, beta=.5):
    if not 0 <= beta <= 1:
        raise ValueError('beta must be in [0,1]')
    usable = valid[:, 0] & valid[:, 1:].any(1)
    if not usable.any():
        return prediction.sum() * 0.
    clean_target = target.detach().masked_fill(~valid[..., None], 0)
    clean_prediction = prediction.masked_fill(~valid[..., None], 0)
    losses = 1 - F.cosine_similarity(clean_prediction, clean_target, dim=-1)
    regional = (losses[:, 1:] * valid[:, 1:]).sum(1) / valid[:, 1:].sum(1).clamp_min(1)
    return (beta*losses[:, 0] + (1-beta)*regional)[usable].mean()
