"""Focal spatial/Null supervision and confirmed-region real adaptation."""
import torch


def dense_loss(probabilities, labels, gamma=2., null_weight=1.):
    if probabilities.ndim != 2 or len(labels) != len(probabilities) or gamma < 0 or null_weight <= 0:
        raise ValueError('invalid probability/label dimensions or loss weights')
    terms = []
    for p, item in zip(probabilities, labels, strict=True):
        target = torch.as_tensor(item['target'], device=p.device, dtype=p.dtype)
        if target.shape != p.shape or not torch.isclose(target.sum(), p.new_tensor(1.)):
            raise ValueError('target must be a normalized spatial/Null distribution')
        if item['mode'] == 'confirmed_regions':
            known = torch.as_tensor(item['known'], device=p.device, dtype=torch.bool)
            if known.shape != p[:-1].shape or not known.any() or target[-1] != 0:
                raise ValueError('real localization requires confirmed positive spatial regions')
            local = p[:-1][known]
            local = local / local.sum().clamp_min(1e-12)
            t = target[:-1][known]
            if not torch.isclose(t.sum(), p.new_tensor(1.)):
                raise ValueError('positive target cannot lie in unknown region')
            terms.append(-(t * (1-local).pow(gamma) * local.clamp_min(1e-12).log()).sum())
        elif item['mode'] == 'simulation':
            weight = torch.ones_like(p); weight[-1] = null_weight
            terms.append(-(target * weight * (1-p).pow(gamma) * p.clamp_min(1e-12).log()).sum())
        else:
            raise ValueError('unsupported supervision mode')
    return torch.stack(terms).mean()
