"""Fixed slot EMA control for the two-candidate replay experiment."""

import torch

from .competitive_writer import CompetitiveWriter
from .evidence_memory import EvidenceMatcher


class SlotEMAMatcher(EvidenceMatcher):
    def __init__(self, ema_decay, **architecture):
        if not 0 <= ema_decay <= 1:
            raise ValueError("EMA old-state weight must be in [0, 1]")
        super().__init__(**architecture, mode="current")
        self.ema_decay = ema_decay
        self.config.update(mode="ema", ema_decay=ema_decay)

    def update(self, observation, valid, state=None):
        # The original current-only model evaluates exactly the same Writer once.
        fresh = super().update(observation, valid)
        if state is None:
            return fresh
        if state["slots"].shape != fresh["slots"].shape:
            raise ValueError("slot EMA replay requires a stable candidate layout")
        visible = fresh["seen"]
        mixed = self.ema_decay * state["slots"] + (1 - self.ema_decay) * fresh["slots"]
        written = torch.where(state["seen"][..., None, None], mixed, fresh["slots"])
        return dict(slots=torch.where(visible[..., None, None], written, state["slots"]),
                    seen=state["seen"] | visible, count=state["count"] + fresh["count"])


def build_replay_matcher(config):
    if config["mode"] == "ema":
        return SlotEMAMatcher(**{k: v for k, v in config.items() if k != "mode"})
    if "writer_attention" in config:
        if config["writer_attention"] != "competitive" or config["mode"] != "memory":
            raise ValueError("unsupported replay Writer configuration")
        model = EvidenceMatcher(**{k: v for k, v in config.items() if k != "writer_attention"})
        model.writer = CompetitiveWriter(model.dim, model.writer.attention.num_heads)
        model.config.update(writer_attention="competitive")
        return model
    return EvidenceMatcher(**config)


def initialize_slot_ema(config, weights, decay):
    if config["mode"] != "memory":
        raise ValueError("expected a learned-memory source checkpoint")
    model = build_replay_matcher(dict(config, mode="ema", ema_decay=decay))
    shared = {k: v for k, v in weights.items() if not k.startswith(("updater.", "gate."))}
    model.load_state_dict(shared, strict=True)
    return model
