"""
train/ema.py — exponential moving average of the model weights
(TRAINING_SPEC §5, §1: kept as its own module rather than inlined in the
loop, so it can be tested on its own).

**Decay is 0.999, not 0.9999** — STAGE_SPEC §5 supersedes DESIGN_SPEC
here: "runs are short and 0.9999 has a ~10,000-step warmup that would
never converge in time". CLAUDE.md's "never cut EMA" still holds; it's
the decay constant that changed, not whether EMA exists.

Train with the live weights, **sample and evaluate with the EMA weights**
(STAGE_SPEC §5). Both get saved.
"""

import copy

import torch
import torch.nn as nn


class EMA:
    """Shadow copy of the model, updated after each optimizer step.

    Includes buffers as well as parameters: `PositionalFourier.freqs` is a
    registered buffer, and a shadow model missing it would fail to load or
    silently use a different positional encoding. Buffers are copied
    outright rather than averaged -- they aren't learned, so averaging
    them has no meaning.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, p in zip(self.shadow.parameters(), model.parameters()):
            # ema = decay*ema + (1-decay)*p, in-place
            ema_p.lerp_(p.detach(), 1.0 - self.decay)
        for ema_b, b in zip(self.shadow.buffers(), model.buffers()):
            ema_b.copy_(b)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)

    @property
    def module(self) -> nn.Module:
        """The averaged model — what sampling and evaluation should use."""
        return self.shadow
