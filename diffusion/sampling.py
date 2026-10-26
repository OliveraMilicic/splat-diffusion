"""
diffusion/sampling.py — needed early by STAGE_SPEC §11 step 6's HARD GATE:
checks 8 ("sampler reproduces the overfitted object") and 9 (mode
collapse) are both sampling checks.

Two samplers, both `diffusers` schedulers (STAGE_SPEC §5: "Do not
hand-roll DPM-Solver"):

  `sample`      DPMSolverMultistepScheduler, ~50 steps. The default, and
                what GaussianCube's own `inference.py` uses. **Ignores the
                learned variance** — DPM-Solver is a deterministic ODE
                solver, there is no noise term for a variance to scale.
  `sample_ddpm` DDPMScheduler, full T steps, ancestral. The learned
                variance (`cfg.learn_sigma`) is only *used* here. Slower by
                the ratio T/num_steps, so it is the check that the variance
                head learned something, not the everyday sampler.

One shared denoising trajectory per call: every object in `cond`/`offset`
advances through the same sequence of `t` values together (standard
batched sampling) — a different `t` usage than `diffusion.schedule.
add_noise`'s per-OBJECT `t` at train time.
"""

import torch

from diffusion.schedule import make_sample_scheduler, make_train_scheduler


def _init_noise(model, cond, seed, device):
    n_target_ch = model.cfg.n_target_ch
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    return torch.randn(cond.shape[0], n_target_ch, generator=gen, device=device), gen


@torch.no_grad()
def sample(model, cond: torch.Tensor, offset: torch.Tensor, T: int = 1000,
           num_steps: int = 50, seed: int = None, device=None) -> torch.Tensor:
    """Returns the sampled target (standardized units, same as training)."""
    device = device or cond.device
    scheduler = make_sample_scheduler(T=T)
    scheduler.set_timesteps(num_steps, device=device)

    target, _ = _init_noise(model, cond, seed, device)
    B = offset.shape[0]
    was_training = model.training
    model.eval()
    for t in scheduler.timesteps:
        t_batch = torch.full((B,), int(t), dtype=torch.long, device=device)
        # `predict_y0` strips the learned-variance channels. DPMSolver's
        # `convert_model_output` only strips them on its `epsilon` branch
        # (and with a hardcoded `[:, :3]`), so under `prediction_type=
        # "sample"` an unstripped 2C tensor would be treated as the
        # prediction itself and silently corrupt every step.
        pred_y0 = model.predict_y0(cond, target, t_batch, offset)
        target = scheduler.step(pred_y0, t, target).prev_sample
    model.train(was_training)
    return target


@torch.no_grad()
def sample_ddpm(model, cond: torch.Tensor, offset: torch.Tensor, T: int = 1000,
                num_steps: int = None, seed: int = None, device=None) -> torch.Tensor:
    """Ancestral DDPM sampling, which is the only sampler that consumes the
    learned variance.

    `DDPMScheduler.step` splits a `2C`-wide `model_output` itself when
    `variance_type` is `learned_range` (`scheduling_ddpm.py`: `torch.split(
    model_output, sample.shape[1], dim=1)`), and its interpolation
    `frac = (v+1)/2; log_var = frac*log(beta_t) + (1-frac)*log(posterior_var)`
    is the same formula `diffusion/losses.py` trains against — so the raw
    model output is passed straight through here, unlike `sample` above.
    """
    device = device or cond.device
    scheduler = make_train_scheduler(T=T, learn_sigma=model.cfg.learn_sigma)
    scheduler.set_timesteps(num_steps or T, device=device)

    target, gen = _init_noise(model, cond, seed, device)
    B = offset.shape[0]
    was_training = model.training
    model.eval()
    for t in scheduler.timesteps:
        t_batch = torch.full((B,), int(t), dtype=torch.long, device=device)
        out = model(cond, target, t_batch, offset)
        if not model.cfg.learn_sigma:
            out = model.split_output(out)[0]
        target = scheduler.step(out, t, target, generator=gen).prev_sample
    model.train(was_training)
    return target
