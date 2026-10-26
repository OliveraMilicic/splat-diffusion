"""
diffusion/losses.py — GaussianCube's training loss, replacing the plain
`L_simple` that `diffusion/schedule.py` provided.

**What GaussianCube actually optimises** (read off
`reference/gaussiancube-official/`, not from the paper text — consulted per
TRAINING_SPEC §0, never imported):

    configs/shapenet_uncond.yml   learn_sigma: true, rescale_learned_sigmas: true,
                                  predict_type: xstart, noise_schedule: cosine
    model/gaussian_diffusion.py   training_losses() ->
                                      terms['mse'] = mean_flat((x_start - out)^2)
                                      terms['vb']  = _vb_terms_bpd(frozen mean)
                                                     * num_timesteps / 1000
                                      terms['loss'] = mse * w(snr) + vb
    train.py forward_backward()   loss = terms['loss'].mean() * diffusion_loss_weight
                                        (+ render L1 + render LPIPS, see below)

So the diffusion loss is the **Improved-DDPM hybrid**: the simple x_0 MSE
we already had, PLUS a variational-bound term that trains a *learned*
per-channel variance. `learn_sigma: true` doubles the network's output
channels (their config: `in_channels: 14`, `out_channels: 28`).

Three properties of the VB term that are easy to get wrong, all of which
`training_losses` is explicit about and this file reproduces:

  1. **The mean is detached inside the VB term.** `frozen_out =
     cat([model_output.detach(), model_var_values])`. The VB gradient
     trains the variance head only; without the detach it fights the MSE
     term for control of the mean and destabilises training.
  2. **At `t == 0` the VB term is the decoder NLL, not the KL** — the KL
     against `q(y_{-1}|...)` doesn't exist. `torch.where(t == 0, ...)`.
  3. **Units are bits**: both halves are divided by `log(2)`. This is
     what makes the `T/1000` rescale (`RESCALED_MSE`) the right relative
     weight against the MSE term, so it is not cosmetic.

**Min-SNR-gamma is implemented but OFF by default, deliberately.** The
path exists in `training_losses` (`min(snr, 5)`), but `min_snr=False` in
`gaussian_diffusion.py:151` and no shipped config turns it on — so
enabling it here would be *diverging* from GaussianCube while claiming to
match it. `cfg.min_snr_gamma > 0` opts in.

**What is NOT ported, and why** (flagged rather than silently dropped, per
CLAUDE.md): GaussianCube's total loss also has an image-space term,
`render_l1_weight * L1(render(y0_hat), gt) + render_lpips_weight *
LPIPS(...)`, enabled by `--use_vgg --load_camera 1` and only switched on
after 50k steps (`has_pretrain_weight`). That is STAGE_SPEC §5's
`L_image` stretch goal, and §5 gates it behind three prerequisites — one
of which ("camera/3DGS frame alignment verified" against ShapeSplat's
released 2D renders) is unverified and is exactly the kind of silent
geometric misalignment that produces wrong gradients rather than an
error. It is not in this build. `eval/metrics.py` renders with gsplat for
*evaluation*, which needs no such alignment (it compares our own renders
of predicted vs. ground-truth Gaussians under identical cameras).

Reduction: **per object, then over objects** — GaussianCube's `mean_flat`
returns one value per sample before `.mean()`. Averaging over the flat
concatenated point array instead would weight objects by their point
count, which varies (STAGE_SPEC §3: 11,100 to 32,768 here).
"""

import numpy as np
import torch


# ----------------------------------------------------------------- iDDPM helpers
# Ported from `reference/gaussiancube-official/model/losses.py` (itself the
# Ho et al. / OpenAI guided-diffusion originals).
def normal_kl(mean1, logvar1, mean2, logvar2):
    """KL( N(mean1, exp(logvar1)) || N(mean2, exp(logvar2)) ), elementwise."""
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


# GaussianCube's `discretized_gaussian_log_likelihood` assumes `x` is uint8
# rescaled to [-1, 1], hence a bin half-width of 1/255. Our targets are
# standardized continuous channels, not 8-bit — but this is kept literal
# because it is what GaussianCube runs on ITS continuous standardized
# volumes too, and because the term is confined to `t == 0` and only ever
# reaches the variance head (the mean is detached). Numerically it stays
# well-conditioned here: at t=0 the posterior log-variance is ~-11.5, so
# the bin half-width is ~1.2 standard deviations, not a degenerate spike.
DISCRETIZATION_HALF_BIN = 1.0 / 255.0


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + DISCRETIZATION_HALF_BIN)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - DISCRETIZATION_HALF_BIN)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    return torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )


def per_object_mean(x: torch.Tensor, lengths) -> torch.Tensor:
    """(total_N, C) -> (B,), mean over each object's own points and channels.

    GaussianCube's `mean_flat` in our flat/variable-N layout. A `lengths`
    slice loop rather than `index_add_(batch_idx)`: with B=4 it is four
    contiguous reductions, where the scatter form pays atomic contention
    on B addresses for total_N values (the same class of cost that was
    measured at 89% of a training step for the adaLN broadcast, see
    `model/ptv3.py::broadcast_per_object`).
    """
    out, start = [], 0
    for n in lengths:
        out.append(x[start:start + n].mean())
        start += n
    return torch.stack(out)


# ----------------------------------------------------------------- the loss
class DiffusionLoss:
    """GaussianCube's `training_losses`, for our flat (total_N, C) layout.

    All schedule constants are derived from the **live scheduler object**
    (`scheduler.alphas_cumprod`), never re-derived from a formula. The
    training scheduler carries `rescale_betas_zero_snr=True`
    (NORMALIZATION.md §3, our documented addition over GaussianCube), which
    changes `alphas_cumprod`; recomputing a "cosine schedule" here from
    `cos((t+0.008)/1.008 * pi/2)^2` would silently give the loss a
    different schedule than the noise the target was corrupted with.
    """

    def __init__(self, scheduler, learn_sigma: bool = True,
                 min_snr_gamma: float = 0.0, device=None):
        self.learn_sigma = learn_sigma
        self.min_snr_gamma = float(min_snr_gamma)

        abar = scheduler.alphas_cumprod.detach().cpu().numpy().astype(np.float64)
        betas = scheduler.betas.detach().cpu().numpy().astype(np.float64)
        alphas = 1.0 - betas
        self.T = len(betas)

        abar_prev = np.append(1.0, abar[:-1])

        # q(y_{t-1} | y_t, y_0) — Ho et al. eq. (6)/(7).
        posterior_variance = betas * (1.0 - abar_prev) / (1.0 - abar)
        # posterior_variance[0] is exactly 0 (abar_prev[0] == 1), so log() is
        # -inf there. iDDPM clips by substituting the t=1 value — needed
        # because the t=0 branch of the VB term still evaluates a log-variance.
        posterior_log_variance_clipped = np.log(
            np.append(posterior_variance[1], posterior_variance[1:])
        )

        self._c = dict(
            sqrt_alphas_cumprod=np.sqrt(abar),
            sqrt_one_minus_alphas_cumprod=np.sqrt(1.0 - abar),
            posterior_mean_coef1=betas * np.sqrt(abar_prev) / (1.0 - abar),
            posterior_mean_coef2=(1.0 - abar_prev) * np.sqrt(alphas) / (1.0 - abar),
            posterior_log_variance_clipped=posterior_log_variance_clipped,
            log_betas=np.log(betas),
        )
        self._cached = {}
        if device is not None:
            self.to(device)

    def to(self, device):
        self._cached = {
            k: torch.as_tensor(v, dtype=torch.float32, device=device)
            for k, v in self._c.items()
        }
        return self

    def _const(self, name: str, t_idx: torch.Tensor) -> torch.Tensor:
        """Schedule constant gathered at `t_idx`, shaped (len(t_idx), 1) to
        broadcast against (total_N, C). Always float32 regardless of an
        enclosing bf16 autocast — these are schedule constants, and
        `exp(logvar1 - logvar2)` in `normal_kl` is not something to run at
        bf16's ~3 decimal digits."""
        if name not in self._cached or self._cached[name].device != t_idx.device:
            self.to(t_idx.device)
        return self._cached[name][t_idx][:, None]

    def q_posterior_mean_log_variance(self, y0, y_t, t_point):
        mean = (self._const("posterior_mean_coef1", t_point) * y0
                + self._const("posterior_mean_coef2", t_point) * y_t)
        return mean, self._const("posterior_log_variance_clipped", t_point)

    def p_mean_log_variance(self, pred_y0, var_values, y_t, t_point):
        """Model's p(y_{t-1} | y_t): mean from the predicted y_0 through the
        same posterior formula, log-variance interpolated between the two
        classical bounds by the learned `var_values`.

        `LEARNED_RANGE` (GaussianCube's `model_var_type`): `v` in [-1, 1]
        maps to log-variance between `posterior_log_variance_clipped`
        (FIXED_SMALL, the lower bound) and `log(beta_t)` (FIXED_LARGE, the
        upper). Nothing constrains `v` to [-1, 1] — iDDPM leaves it
        unconstrained on purpose so the bound can be exceeded slightly —
        so the interpolation is an extrapolation at the edges. That is the
        reference behaviour; do not add a tanh.
        """
        mean, min_log = self.q_posterior_mean_log_variance(pred_y0, y_t, t_point)
        if var_values is None:
            return mean, min_log
        max_log = self._const("log_betas", t_point)
        frac = (var_values.float() + 1.0) / 2.0
        return mean, frac * max_log + (1.0 - frac) * min_log

    def vb_term(self, pred_y0, var_values, y0, y_t, t_point, t_obj, lengths):
        """One VB term per object, in bits. KL at t>0, decoder NLL at t==0."""
        true_mean, true_log_var = self.q_posterior_mean_log_variance(y0, y_t, t_point)
        # The detach is the point (see module docstring, property 1): this
        # term must move the variance head only.
        model_mean, model_log_var = self.p_mean_log_variance(
            pred_y0.detach().float(), var_values, y_t, t_point)

        kl = normal_kl(true_mean, true_log_var, model_mean, model_log_var)
        kl = per_object_mean(kl, lengths) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            y0, means=model_mean, log_scales=0.5 * model_log_var)
        decoder_nll = per_object_mean(decoder_nll, lengths) / np.log(2.0)

        return torch.where(t_obj == 0, decoder_nll, kl)

    def mse_weight(self, t_obj: torch.Tensor) -> torch.Tensor:
        """Min-SNR-gamma, `min(snr, gamma)` — off unless `min_snr_gamma > 0`
        (see module docstring: GaussianCube ships it disabled)."""
        if self.min_snr_gamma <= 0:
            return torch.ones_like(t_obj, dtype=torch.float32)
        alpha = self._const("sqrt_alphas_cumprod", t_obj)[:, 0]
        sigma = self._const("sqrt_one_minus_alphas_cumprod", t_obj)[:, 0]
        snr = (alpha / sigma) ** 2
        w = torch.clamp(snr, max=self.min_snr_gamma)
        # snr == 0 is reachable here and not in GaussianCube: zero terminal
        # SNR (NORMALIZATION.md §3) makes alphas_cumprod[T-1] exactly 0, so
        # the final step would get weight 0 and never train. iDDPM's own
        # guard (`mse_loss_weight[snr == 0] = 1.0`) covers it.
        return torch.where(snr == 0, torch.ones_like(w), w)

    def __call__(self, pred, y0, y_t, t_obj, lengths) -> dict:
        """`pred`: (total_N, C) or (total_N, 2C) when `learn_sigma`.
        `t_obj`: (B,) one timestep per object. `lengths`: python list of
        per-object point counts. Returns a dict of scalar tensors; `loss`
        is what to call `.backward()` on."""
        n_target = y0.shape[1]
        if self.learn_sigma:
            assert pred.shape[1] == 2 * n_target, (
                f"learn_sigma=True needs {2 * n_target} output channels, got "
                f"{pred.shape[1]} — model/denoiser.py sizes out_ch from cfg.learn_sigma")
            pred_y0, var_values = pred[:, :n_target], pred[:, n_target:]
        else:
            assert pred.shape[1] == n_target
            pred_y0, var_values = pred, None

        t_point = torch.repeat_interleave(
            t_obj, torch.as_tensor(lengths, device=t_obj.device))

        mse = per_object_mean((y0 - pred_y0.float()) ** 2, lengths)
        terms = dict(mse=mse.mean(), loss=None)

        weighted = mse * self.mse_weight(t_obj)
        if var_values is not None:
            # RESCALED_MSE: `terms['vb'] *= num_timesteps / 1000` — "without a
            # factor of 1/1000 the VB term hurts the MSE term"
            # (gaussian_diffusion.py:843). Exactly 1.0 at our T=1000, kept
            # explicit so a T change scales it the way the reference does.
            vb = self.vb_term(pred_y0, var_values, y0, y_t, t_point, t_obj, lengths)
            vb = vb * (self.T / 1000.0)
            terms["vb"] = vb.mean()
            terms["loss"] = (weighted + vb).mean()
        else:
            terms["loss"] = weighted.mean()
        return terms
