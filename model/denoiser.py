"""
model/denoiser.py — STAGE_SPEC §11 build-order step 4 (part 2 of 2).

The single object training and sampling both call — mirrors the role
TRAINING_SPEC §3's `EDMPrecond` played pre-pivot, simplified because
STAGE_SPEC §5 drops EDM preconditioning entirely: no `c_in`/`c_out`/
`c_skip`, the network predicts `y_0` directly.

**Output width follows GaussianCube's `learn_sigma`** (their
`configs/shapenet_uncond.yml`: `in_channels: 14, out_channels: 28`). With
`cfg.learn_sigma` the head emits `2 * n_target_ch`: the first half is the
predicted `y_0`, the second half is the `LEARNED_RANGE` variance
interpolant consumed by `diffusion/losses.py`. `split_output` is the one
place that split is written down, so a caller can never get the halves
backwards.
"""

import torch
import torch.nn as nn

from model.conditioning import build_input, extract_pos
from model.ptv3 import PTv3
from stage_config import StageConfig


class Denoiser(nn.Module):
    """`forward(cond, target_noisy, t, offset) -> (total_N, out_ch)`.

    `out_ch` is `cfg.n_model_out_ch`: `n_target_ch` normally, `2 *
    n_target_ch` under `learn_sigma`. The prediction is `y_0`, NOT
    `epsilon` (STAGE_SPEC §5).
    """

    def __init__(self, cfg: StageConfig):
        super().__init__()
        self.cfg = cfg
        in_ch = cfg.n_cond_ch + cfg.n_target_ch
        self.backbone = PTv3(
            in_ch=in_ch,
            out_ch=cfg.n_model_out_ch,
            d_model=cfg.d_model,
            depth=cfg.n_blocks,
            heads=cfg.n_heads,
            patch_size=cfg.patch_size,
            mlp_ratio=cfg.mlp_ratio,
        )
        if cfg.learn_sigma:
            # iDDPM's variance head starts at the FIXED_SMALL bound rather
            # than mid-range: `frac = (v + 1) / 2`, so v = -1 selects
            # `posterior_log_variance_clipped` exactly. Zero-init would start
            # every step at the geometric midpoint between the small and large
            # bounds, which is a much noisier reverse process for the first few
            # thousand steps and shows up as visibly grainy early samples.
            # Weights zero + bias -1 makes the initial variance exactly the
            # classical DDPM one, and the VB term moves it from there.
            with torch.no_grad():
                self.backbone.out_proj.weight[cfg.n_target_ch:].zero_()
                self.backbone.out_proj.bias[cfg.n_target_ch:].fill_(-1.0)

    def forward(self, cond: torch.Tensor, target_noisy: torch.Tensor,
                t: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        pos = extract_pos(cond, self.cfg)
        x_in = build_input(cond, target_noisy)
        return self.backbone(x_in, pos, offset, t)

    def split_output(self, out: torch.Tensor):
        """(total_N, out_ch) -> (pred_y0, var_values). `var_values` is None
        when `learn_sigma` is off."""
        c = self.cfg.n_target_ch
        if not self.cfg.learn_sigma:
            return out, None
        return out[:, :c], out[:, c:]

    def predict_y0(self, cond, target_noisy, t, offset) -> torch.Tensor:
        """Just the mean — what a sampler that ignores the learned variance
        (DPM-Solver) needs. `diffusers`' `DPMSolverMultistepScheduler` does
        NOT strip the extra channels under `prediction_type="sample"` (it
        only does so on the `epsilon` branch, and with a hardcoded `[:, :3]`
        for images), so stripping has to happen here."""
        return self.split_output(self(cond, target_noisy, t, offset))[0]

    def gate_means(self) -> list:
        """LOGGING.md panel 2, see `PTv3.gate_means`."""
        return self.backbone.gate_means()
