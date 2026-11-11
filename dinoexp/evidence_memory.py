"""Oracle-routed, variable-candidate memory. IDs/masks stay outside this module."""
import torch
from torch import nn
from .local_attention import CrossBlock


class EvidenceMatcher(nn.Module):
    def __init__(self, dim=256, heads=4, slots=8, mode='memory'):
        super().__init__()
        if mode not in ('memory', 'current', 'mean') or min(dim, heads, slots) < 1 or dim % heads:
            raise ValueError('invalid memory configuration')
        self.config = dict(dim=dim, heads=heads, slots=slots, mode=mode)
        self.dim, self.slots, self.mode = dim, slots, mode
        self.initial = nn.Parameter(torch.randn(1, slots, dim) * .02)
        self.writer = CrossBlock(dim, heads)
        if mode == 'memory':
            self.updater = CrossBlock(dim, heads)
            self.gate = nn.Linear(dim * 2, 1)
            nn.init.constant_(self.gate.bias, -2.)
        self.reader = CrossBlock(dim, heads)
        self.match = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim),
                                   nn.GELU(), nn.Linear(dim, 1))
        self.null = nn.Parameter(torch.tensor(0.))

    def initial_state(self, batch, candidates, device, dtype):
        return dict(slots=self.initial.to(device=device, dtype=dtype).expand(
            batch * candidates, -1, -1).reshape(batch, candidates, self.slots, self.dim),
            seen=torch.zeros(batch, candidates, dtype=torch.bool, device=device),
            count=torch.zeros(batch, candidates, device=device, dtype=dtype))

    def update(self, observation, valid, state=None):
        if observation.ndim != 4 or valid.shape != observation.shape[:-1]:
            raise ValueError('expected observations [B,K,P,D], validity [B,K,P]')
        b, k, _, d = observation.shape
        if d != self.dim or k < 1:
            raise ValueError('need positive candidate capacity and matching dimension')
        valid = valid.bool()
        if state is None or self.mode == 'current':
            state = self.initial_state(b, k, observation.device, observation.dtype)
        elif state['slots'].shape[:2] != (b, k):
            old_k = state['slots'].shape[1]
            if state['slots'].shape[0] != b or old_k > k:
                raise ValueError('candidate removal/reordering requires explicit state management')
            extra = self.initial_state(b, k-old_k, observation.device, observation.dtype)
            state = {key: torch.cat((state[key], extra[key]), dim=1) for key in state}
        visible = valid.any(-1)
        if not visible.any():
            return state
        x = observation[visible].masked_fill(~valid[visible][..., None], 0)
        if not torch.isfinite(x).all():
            raise ValueError('nonfinite valid observation')
        written, _ = self.writer(self.initial.expand(len(x), -1, -1), x, valid[visible])
        previous = state['slots'][visible]
        if self.mode == 'memory':
            proposal, _ = self.updater(previous, written,
                                      torch.ones(written.shape[:2], dtype=torch.bool, device=x.device))
            gate = self.gate(torch.cat((previous, proposal), -1)).sigmoid()
            updated = previous + gate * (proposal - previous)
            updated = torch.where(state['seen'][visible][:, None, None], updated, written)
        elif self.mode == 'mean':
            count = state['count'][visible][:, None, None]
            updated = (previous * count + written) / (count + 1)
        else:
            updated = written
        slots = state['slots'].clone()
        slots[visible] = updated
        return dict(slots=slots, seen=state['seen'] | visible,
                    count=state['count'] + visible.to(observation.dtype))

    def score(self, reference, reference_valid, state):
        b, k, s, d = state['slots'].shape
        if reference.ndim != 3 or reference.shape[0] != b or reference.shape[-1] != d:
            raise ValueError('reference must be [B,R,D]')
        if reference_valid.shape != reference.shape[:2] or not reference_valid.any(-1).all():
            raise ValueError('reference needs nonempty valid tokens')
        reference = reference.masked_fill(~reference_valid[..., None], 0)
        r = reference[:, None].expand(-1, k, -1, -1).reshape(b*k, -1, d)
        read, _ = self.reader(r, state['slots'].reshape(b*k, s, d),
                              torch.ones(b*k, s, dtype=torch.bool, device=r.device))
        weights = reference_valid[:, None, :, None].expand(-1, k, -1, -1).reshape(b*k, -1, 1)
        a = (r * weights).sum(1) / weights.sum(1)
        z = (read * weights).sum(1) / weights.sum(1)
        scores = self.match(torch.cat((a, z, a*z, (a-z).abs()), -1)).reshape(b, k)
        scores = scores.masked_fill(~state['seen'], -1e4)
        return torch.cat((scores, self.null.expand(b, 1)), -1)
