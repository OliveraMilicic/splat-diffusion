"""
check_schedule_stage.py — STAGE_SPEC §11 build-order step 5 check: N1
(NORMALIZATION.md §6) on real standardized target data.

At t = T-1 (the last valid training index), y_t must have mean ~ 0,
std ~ 1, and show no measurable correlation with y_0 -- confirming
`rescale_betas_zero_snr=True` actually gives zero terminal SNR rather
than leaving a faint, exploitable trace of y_0 (NORMALIZATION.md §2-3).

Run: python check_schedule_stage.py
"""

import container_env  # noqa: F401 -- MUST precede torch, see its docstring

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import numpy as np
import torch

from batching import lengths_to_offset, offset_to_batch_idx
from diffusion.schedule import add_noise, make_train_scheduler
from stage_config import StageConfig
from train.dataset import ChairPointDataset

torch.manual_seed(0)
cfg = StageConfig(stage=1, num_workers=0)
scheduler = make_train_scheduler(T=cfg.T)

print(f"scheduler: {scheduler.config.beta_schedule}, T={scheduler.config.num_train_timesteps}, "
      f"prediction_type={scheduler.config.prediction_type}, "
      f"rescale_betas_zero_snr={scheduler.config.rescale_betas_zero_snr}")
alphas_cumprod = scheduler.alphas_cumprod
print(f"alpha_bar[0] = {alphas_cumprod[0].item():.6f}  (near 1: almost no noise at t=0)")
print(f"alpha_bar[T-1] = {alphas_cumprod[-1].item():.2e}  (must be ~0 for zero terminal SNR)\n")
assert alphas_cumprod[-1].item() < 1e-4, "rescale_betas_zero_snr doesn't look like it's taking effect"

# ----------------------------------------------------------------- N1: real batch, t = T-1
ds = ChairPointDataset(cfg, split="train", augment=False)
samples = [ds[i] for i in range(4)]
y0 = torch.cat([s["target"] for s in samples], dim=0)  # (total_N, 3) standardized RGB
lengths = [s["n"] for s in samples]
offset = lengths_to_offset(lengths)
batch_idx = offset_to_batch_idx(offset)

noise = torch.randn_like(y0)
t_max = torch.full((len(samples),), cfg.T - 1, dtype=torch.long)
y_T = add_noise(scheduler, y0, noise, t_max, batch_idx)

mean_yT, std_yT = y_T.mean().item(), y_T.std().item()
print(f"[N1] y_T (t={cfg.T - 1}) over {y0.shape[0]} points x {y0.shape[1]} channels:")
print(f"    mean = {mean_yT:.4f}  (expect ~0)")
print(f"    std  = {std_yT:.4f}  (expect ~1)")

corr = np.corrcoef(y_T.flatten().numpy(), y0.flatten().numpy())[0, 1]
print(f"    Pearson corr(y_T, y0) = {corr:.4f}  (expect ~0 -- nonzero means terminal SNR isn't actually zero)")

assert abs(mean_yT) < 0.05, f"y_T mean {mean_yT:.4f} too far from 0"
assert abs(std_yT - 1.0) < 0.05, f"y_T std {std_yT:.4f} too far from 1"
assert abs(corr) < 0.02, f"y_T correlates with y_0 (corr={corr:.4f}) -- terminal SNR isn't zero"
print("    PASS\n")

# sanity in the other direction: t=0 should be almost unchanged from y0
t_min = torch.zeros(len(samples), dtype=torch.long)
y_0_noised = add_noise(scheduler, y0, noise, t_min, batch_idx)
corr_t0 = np.corrcoef(y_0_noised.flatten().numpy(), y0.flatten().numpy())[0, 1]
print(f"[sanity] t=0: corr(y_t, y0) = {corr_t0:.4f}  (expect ~1 -- almost no noise yet)")
assert corr_t0 > 0.99
print("    PASS\n")

print("ALL SCHEDULE-STAGE CHECKS PASSED.")
