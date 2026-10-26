"""
diffusion/schedule.py — STAGE_SPEC §11 build-order step 5.

GaussianCube-style diffusion (STAGE_SPEC §5), not EDM: cosine schedule
(`beta_schedule="squaredcos_cap_v2"`, diffusers' name for Nichol &
Dhariwal), `T=1000`, `prediction_type="sample"` (the network predicts
`y_0` directly, not `epsilon`), `rescale_betas_zero_snr=True`
(NORMALIZATION.md §3: required, not optional — without it `y_T` still
carries a faint trace of `y_0` at train time that isn't there at sample
time, and with 16,384 redundant per-object copies even a faint trace is
enough to recover the mean colour, NORMALIZATION.md §2).

`diffusers` *schedulers* only — no hand-rolled DPM-Solver (STAGE_SPEC
§5): these are algorithms, not pretrained weights, the same "from
scratch" distinction TRAINING_SPEC §0 draws for the PTv3 reference code.
"""

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler

# Shared across both schedulers. `clip_sample` is DDPM-specific (DPM-
# Solver has no such argument -- it never clips intermediate samples to
# a fixed range), so it's added only in make_train_scheduler below.
SCHEDULER_KWARGS = dict(
    beta_schedule="squaredcos_cap_v2",   # cosine
    prediction_type="sample",             # predict y_0, not epsilon (STAGE_SPEC §5)
    rescale_betas_zero_snr=True,          # NORMALIZATION.md §3
)


def make_train_scheduler(T: int = 1000, learn_sigma: bool = False) -> DDPMScheduler:
    """`learn_sigma` sets `variance_type="learned_range"`, which changes
    exactly one thing about this scheduler: `.step()` then expects a
    `2C`-wide `model_output`, splits it, and scales its noise term by the
    learned log-variance. Training's `add_noise` is unaffected (the forward
    process has no learned parts) — so the flag only matters to
    `diffusion.sampling.sample_ddpm`. See `diffusion/losses.py` for the
    matching training-side term."""
    kwargs = dict(SCHEDULER_KWARGS)
    if learn_sigma:
        kwargs["variance_type"] = "learned_range"
    return DDPMScheduler(
        num_train_timesteps=T,
        clip_sample=False,   # target is standardized (mean 0, std ~1), not [-1,1] pixels
        **kwargs,
    )


def make_sample_scheduler(T: int = 1000) -> DPMSolverMultistepScheduler:
    """~50 steps at sample time (STAGE_SPEC §5) via
    `scheduler.set_timesteps(50)` at the call site (diffusion/sampling.py)
    -- the scheduler is trained/constructed at T=1000 either way.

    `solver_order=3`, not diffusers' default of 2, to match GaussianCube's
    own sampler: `inference.py:218` passes `order=3 if not cond_gen else 2`,
    and this model is not classifier-free-guided, so their unconditional
    branch is the one that applies. diffusers' own guidance agrees (order 2
    for guided sampling, 3 for unconditional). `algorithm_type` is
    `dpmsolver++` in both, which is diffusers' default and their explicit
    argument.
    """
    return DPMSolverMultistepScheduler(num_train_timesteps=T, solver_order=3,
                                       **SCHEDULER_KWARGS)


def add_noise(scheduler, y0: torch.Tensor, noise: torch.Tensor,
              t: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
    """`y0`/`noise`: (total_N, C) flat target, per NORMALIZATION.md §5's
    `y_t = sqrt(alpha_bar_t)*y0 + sqrt(1-alpha_bar_t)*noise`. `t`: (B,)
    one timestep per OBJECT (diffusion noise level is per-object, not
    per-point); `batch_idx`: (total_N,) broadcasts it to `y0`'s flat
    per-point layout before `scheduler.add_noise` (which accepts a
    per-row timestep matching `original_samples`'s leading dim)."""
    t_per_point = t[batch_idx]
    return scheduler.add_noise(y0, noise, t_per_point)


def l_simple(pred_y0: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
    """STAGE_SPEC §5: `L_simple = || y0_hat(y_t, t, cond) - y_0 ||^2`.

    **Superseded as the training objective by `diffusion/losses.py`'s
    GaussianCube hybrid loss** (MSE + learned-variance VB). Kept because it
    is still the right thing for a plain reconstruction number: the test
    evaluation and the overfit gate both report it, and the gate figures
    recorded in PLAN.md are in these units, so it stays comparable.
    """
    return F.mse_loss(pred_y0, y0)


def dynamic_clip(target: torch.Tensor, p: float = 0.995) -> torch.Tensor:
    """GaussianCube `dataset/dataset.py::dynamic_clip`: per-channel symmetric
    clip at the `p`-quantile of `|value|`, computed **within one object**.

    OFF by default here (`StageConfig.clip_input_p = 0.0`) on a measurement,
    not by default-copying — see that field's comment. Kept so it can be
    switched on without re-deriving it if training turns unstable.
    """
    if not p:
        return target
    s = torch.quantile(target.abs(), p, dim=0)     # (C,) per-channel threshold
    return torch.clamp(target, -s, s)
