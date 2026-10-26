"""
check_losses.py — self-checks for `diffusion/losses.py` (GaussianCube's
hybrid MSE + learned-variance VB loss) and the learn_sigma plumbing
through `model/denoiser.py` and `diffusion/sampling.py`.

Every check here is either "agrees with an independent recomputation" or
"has the property the reference says it must have". None of them is
"it ran without crashing".

Run: python check_losses.py
"""

import container_env  # noqa: F401 -- MUST precede torch

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import numpy as np
import torch

from batching import lengths_to_offset, offset_to_batch_idx
from diffusion.losses import DiffusionLoss, normal_kl, per_object_mean
from diffusion.schedule import add_noise, make_train_scheduler
from model.denoiser import Denoiser
from stage_config import StageConfig

torch.manual_seed(0)
np.random.seed(0)

cfg = StageConfig(stage=3, n_gaussians=1024, batch_size=3, num_workers=0)
print(f"stage {cfg.stage} ({cfg.target_names}): "
      f"{cfg.n_target_ch} target ch, {cfg.n_cond_ch} cond ch, "
      f"learn_sigma={cfg.learn_sigma} -> model out_ch={cfg.n_model_out_ch}")

sched = make_train_scheduler(T=cfg.T, learn_sigma=cfg.learn_sigma)
loss_fn = DiffusionLoss(sched, learn_sigma=cfg.learn_sigma,
                         min_snr_gamma=cfg.min_snr_gamma)

lengths = [700, 1024, 900]
offset = lengths_to_offset(lengths)
batch_idx = offset_to_batch_idx(offset)
total_N = sum(lengths)
C = cfg.n_target_ch

y0 = torch.randn(total_N, C)
noise = torch.randn(total_N, C)


# ----------------------------------------------------------------- L1
print("\n[check L1] schedule constants match a from-scratch recomputation")
abar = sched.alphas_cumprod.numpy().astype(np.float64)
betas = sched.betas.numpy().astype(np.float64)
abar_prev = np.append(1.0, abar[:-1])
ref_post_var = betas * (1 - abar_prev) / (1 - abar)
ref_coef1 = betas * np.sqrt(abar_prev) / (1 - abar)
ref_coef2 = (1 - abar_prev) * np.sqrt(1 - betas) / (1 - abar)
loss_fn.to(torch.device("cpu"))
err1 = np.abs(loss_fn._c["posterior_mean_coef1"] - ref_coef1).max()
err2 = np.abs(loss_fn._c["posterior_mean_coef2"] - ref_coef2).max()
# posterior_variance[0] == 0 exactly is what forces the log-clip; assert it
# rather than assume, since a schedule change would silently make the clip
# a no-op or an -inf.
print(f"    posterior_variance[0] = {ref_post_var[0]:.3e} (must be exactly 0 -> log-clip needed)")
print(f"    alphas_cumprod[-1]    = {abar[-1]:.3e} (zero terminal SNR, NORMALIZATION.md §3)")
print(f"    coef1 max err {err1:.2e}   coef2 max err {err2:.2e}")
assert ref_post_var[0] == 0.0
assert abar[-1] == 0.0
assert err1 < 1e-12 and err2 < 1e-12
assert np.isfinite(loss_fn._c["posterior_log_variance_clipped"]).all()
print("    PASS")


# ----------------------------------------------------------------- L2
print("\n[check L2] q_posterior recovers y0 exactly at t=0")
# At t=0, q(y_{-1}|y_t,y_0) collapses onto y_0: coef1[0]=1, coef2[0]=0.
# If an off-by-one crept into the t indexing this is where it shows.
t0 = torch.zeros(len(lengths), dtype=torch.long)
t0_point = torch.repeat_interleave(t0, torch.as_tensor(lengths))
y_t0 = add_noise(sched, y0, noise, t0, batch_idx)
mean0, _ = loss_fn.q_posterior_mean_log_variance(y0, y_t0, t0_point)
print(f"    max|q_posterior_mean(t=0) - y0| = {(mean0 - y0).abs().max():.2e}")
assert (mean0 - y0).abs().max() < 1e-5
print("    PASS")


# ----------------------------------------------------------------- L3
print("\n[check L3] VB term is the KL for t>0 and the decoder NLL at t=0")
t_mixed = torch.tensor([0, 500, 999])
t_point = torch.repeat_interleave(t_mixed, torch.as_tensor(lengths))
y_t = add_noise(sched, y0, noise, t_mixed, batch_idx)
pred_y0 = y0 + 0.1 * torch.randn_like(y0)
var_values = torch.zeros(total_N, C)          # midway between the two bounds
vb = loss_fn.vb_term(pred_y0, var_values, y0, y_t, t_point, t_mixed, lengths)

# Independent recomputation of the t=500 object's KL, straight from the formula.
s, e = lengths[0], lengths[0] + lengths[1]
tm, tlv = loss_fn.q_posterior_mean_log_variance(y0[s:e], y_t[s:e], t_point[s:e])
pm, plv = loss_fn.p_mean_log_variance(pred_y0[s:e], var_values[s:e], y_t[s:e], t_point[s:e])
ref_kl = normal_kl(tm, tlv, pm, plv).mean() / np.log(2.0)
print(f"    object 1 (t=500): vb={vb[1]:.6f}  independent KL recomputation={ref_kl:.6f}")
assert torch.allclose(vb[1], ref_kl, atol=1e-5)
assert torch.isfinite(vb).all(), "VB non-finite — check the t=0 log-variance clip"
print(f"    vb per object (t={t_mixed.tolist()}): {[round(float(v), 4) for v in vb]}")
print("    PASS")


# ----------------------------------------------------------------- L4
print("\n[check L4] the VB gradient reaches the variance head and NOT the mean")
# GaussianCube's `frozen_out = cat([model_output.detach(), model_var_values])`.
# Without the detach, the VB term competes with the MSE term for the mean.
pm_leaf = (y0 + 0.1 * torch.randn_like(y0)).requires_grad_(True)
pv_leaf = torch.zeros(total_N, C, requires_grad=True)
loss_fn.vb_term(pm_leaf, pv_leaf, y0, y_t, t_point, t_mixed, lengths).mean().backward()
g_mean = pm_leaf.grad.abs().max().item() if pm_leaf.grad is not None else 0.0
g_var = pv_leaf.grad.abs().max().item()
print(f"    max|dVB/d(pred_y0)| = {g_mean:.2e}  (must be exactly 0)")
print(f"    max|dVB/d(var)|     = {g_var:.2e}  (must be > 0)")
assert g_mean == 0.0, "the VB term is training the mean — the .detach() is missing"
assert g_var > 0.0
print("    PASS")


# ----------------------------------------------------------------- L5
print("\n[check L5] learn_sigma=False reduces to L_simple (per-object mean)")
plain = DiffusionLoss(sched, learn_sigma=False)
terms = plain(pred_y0, y0, y_t, t_mixed, lengths)
ref_mse = per_object_mean((y0 - pred_y0) ** 2, lengths).mean()
print(f"    loss={terms['loss']:.6f}   direct per-object MSE={ref_mse:.6f}")
assert torch.allclose(terms["loss"], ref_mse, atol=1e-7)
assert "vb" not in terms
print("    PASS")


# ----------------------------------------------------------------- L6
print("\n[check L6] min-SNR weighting: off by default, correct when on")
w_off = loss_fn.mse_weight(torch.tensor([0, 500, 999]))
snr_on = DiffusionLoss(sched, learn_sigma=True, min_snr_gamma=5.0)
w_on = snr_on.mse_weight(torch.tensor([0, 500, 999]))
snr = abar / (1 - abar)
print(f"    gamma=0 (GaussianCube default): {w_off.tolist()}")
print(f"    gamma=5: {[round(float(x), 4) for x in w_on]}   "
      f"true snr at those t: {[round(float(snr[i]), 4) for i in (0, 500, 999)]}")
assert torch.allclose(w_off, torch.ones(3))
assert abs(float(w_on[0]) - 5.0) < 1e-4, "low-t weight should saturate at gamma"
assert abs(float(w_on[1]) - float(snr[500])) < 1e-3
# zero terminal SNR makes snr[999] exactly 0; iDDPM's guard forces weight 1
# so the last timestep still trains.
assert abs(float(w_on[2]) - 1.0) < 1e-6, "snr==0 guard missing — t=T-1 would never train"
print("    PASS")


# ----------------------------------------------------------------- L7
print("\n[check L7] the real Denoiser: output width, variance-head init, split")
model = Denoiser(cfg)
cond = torch.randn(total_N, cfg.n_cond_ch)
out = model(cond, y_t, t_mixed, offset)
p, v = model.split_output(out)
print(f"    out {tuple(out.shape)} -> mean {tuple(p.shape)} + var {tuple(v.shape)}")
assert out.shape == (total_N, 2 * C)
assert p.shape == v.shape == (total_N, C)
# At init the variance head must emit exactly -1, which selects the
# FIXED_SMALL bound (frac = (v+1)/2 = 0) -> the classical DDPM variance.
print(f"    variance head at init: min={v.min():.6f} max={v.max():.6f} (must be exactly -1)")
assert torch.allclose(v, torch.full_like(v, -1.0), atol=1e-6)
_, lv_init = loss_fn.p_mean_log_variance(p, v, y_t, t_point)
_, lv_fixed = loss_fn.q_posterior_mean_log_variance(y0, y_t, t_point)
print(f"    -> initial log-variance equals posterior_log_variance_clipped: "
      f"max diff {(lv_init - lv_fixed).abs().max():.2e}")
assert (lv_init - lv_fixed).abs().max() < 1e-5

terms = loss_fn(out, y0, y_t, t_mixed, lengths)
print(f"    loss={terms['loss']:.4f}  mse={terms['mse']:.4f}  vb={terms['vb']:.4f}")
assert all(torch.isfinite(v_) for v_ in terms.values())
terms["loss"].backward()
n_no_grad = sum(1 for q in model.parameters() if q.requires_grad and q.grad is None)
assert n_no_grad == 0, f"{n_no_grad} parameters got no gradient from the hybrid loss"
print(f"    backward: all {sum(1 for _ in model.parameters())} parameter tensors got gradient")
print("    PASS")


# ----------------------------------------------------------------- L8
print("\n[check L8] both samplers run and the DPM path strips the variance channels")
from diffusion.sampling import sample, sample_ddpm  # noqa: E402

small_lengths = [256]
small_offset = lengths_to_offset(small_lengths)
small_cond = torch.randn(256, cfg.n_cond_ch)
s_dpm = sample(model, small_cond, small_offset, T=cfg.T, num_steps=4, seed=0)
s_ddpm = sample_ddpm(model, small_cond, small_offset, T=cfg.T, num_steps=4, seed=0)
print(f"    DPM-Solver  -> {tuple(s_dpm.shape)}  finite={bool(torch.isfinite(s_dpm).all())}")
print(f"    DDPM        -> {tuple(s_ddpm.shape)}  finite={bool(torch.isfinite(s_ddpm).all())}")
assert s_dpm.shape == (256, C) and torch.isfinite(s_dpm).all()
assert s_ddpm.shape == (256, C) and torch.isfinite(s_ddpm).all()
# The strip is the thing to prove: an unstripped 2C tensor into a C-wide
# sampler state would either throw or broadcast-corrupt, so shape alone is
# the check that `predict_y0` ran.
assert model.predict_y0(small_cond, s_dpm, torch.zeros(1, dtype=torch.long),
                         small_offset).shape == (256, C)
print("    PASS")

print("\nALL LOSS CHECKS PASSED.")
