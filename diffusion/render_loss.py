"""
diffusion/render_loss.py — the second half of GaussianCube's total loss:
an image-space term on the RENDERED prediction.

`diffusion/losses.py` ports their diffusion loss. This is the rest of what
`train.py::forward_backward` actually optimises:

    loss = diffusion_loss * diffusion_loss_weight
         + render_l1_weight   * L1  (render(x0_hat), gt_image)
         + render_lpips_weight * LPIPS_vgg(render(x0_hat)*2-1, gt_image*2-1)

Their shipped ShapeNet command (README) sets `--load_camera 1
--render_l1_weight 10 --render_lpips_weight 10`, i.e. **one camera per
object per step** and weight 10 on each — those are the defaults here.

**The 10 is deliberate and the paper says why.** Eq. (5) is
`L = L_simple + lambda * L_image`, "where lambda is a balancing weight",
and §4.2 states "loss weight lambda set to 10". §3.3 gives the motivation:
"We **additionally** impose image-level supervision to improve the
rendering quality of generated GaussianCube, which has been demonstrated to
effectively enhance the perceptual details in previous works." So
`L_image` is *meant* to outweigh `L_simple` by an order of magnitude — a
measured gradient budget where the image term dominates the MSE term is
this design working, not a symptom. What would be wrong is training
destabilising, not the ratio itself.

**Paper vs. code, a real discrepancy, resolved in favour of the code.**
Paper eq. (4) is

    L_image = E[ sum_l || Psi_l(I_pred) - Psi_l(I_gt) ||_2^2 ]   (perceptual)
            + E[ || I_pred - I_gt ||_2 ]                          (pixel)

i.e. an **L2** norm on pixels. Their `train.py:285` uses
`th.nn.L1Loss()`. The code is what produced the published numbers, so L1
is the default here; `pixel_norm="l2"` switches to the paper's form. The
perceptual half needs no such choice: LPIPS *is* a squared-L2 distance on
multi-resolution VGG features with learned channel weights, which is
exactly eq. (4)'s first term.
The term is gated off until the model can already produce a roughly-right
`x0`: `forward_backward` does `has_pretrain_weight or self.step >= 50_000`.

**One deliberate difference, and it is the reason this was buildable at
all.** GaussianCube's `gt_image` is a *real dataset render* loaded from
disk with its camera. Ours is a render of the **ground-truth Gaussians**
under the same camera as the prediction. STAGE_SPEC §5 gates `L_image`
behind three prerequisites, and the third — "camera/3DGS frame alignment
verified" against ShapeSplat's released 2D renders — is the one it warns
"is the likeliest thing to silently break; a misalignment produces
geometrically wrong gradients, not an error." Rendering both sides
ourselves, with one rig, removes that failure mode entirely: any frame
convention we pick is applied identically to prediction and target, so the
gradient is right even if the convention is arbitrary.

What it costs: the target is the fitted 3DGS rather than the photograph it
was fitted to, so the representation's own fitting error is invisible to
this loss. That error is small (GaussianCube's Table 6 reports PSNR 34.94
for their fitting) and it is *not* something our model could fix anyway —
positions and rotations are frozen conditioning, so the fit is given.

**Gradients only need to reach three channel groups.** At stage 3 the
target is log-scale, opacity and RGB; positions and quaternions are frozen
conditioning and are rendered from `cond` unchanged. So this loss teaches
exactly the attributes a point renderer cannot even draw — which is the
argument for having it.
"""

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch

from eval.render3d import (gsplat_available, orbit_cameras, rasterize_differentiable,
                            tokens_to_gaussians)
from stage_config import recombine_channels


class RenderLoss:
    """`L_image` — call it with the model's predicted `y0` and get back
    `dict(render_l1, render_lpips, render_total, render_weight)`.

    `render_total` is already weight- and ramp-scaled, so the caller adds it
    to the diffusion loss and nothing else.
    """

    def __init__(self, cfg, norm_stats, device, image_size: int = 128,
                 n_views: int = 1, l1_weight: float = 10.0,
                 lpips_weight: float = 10.0, start_step: int = 10_000,
                 ramp_steps: int = 2_000, max_points: int = 0,
                 backend: str = "auto", seed: int = 0,
                 pixel_norm: str = "l1"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.image_size = image_size
        self.n_views = n_views
        self.l1_weight = l1_weight
        self.lpips_weight = lpips_weight
        assert pixel_norm in ("l1", "l2"), pixel_norm
        self.pixel_norm = pixel_norm      # "l1" = their code, "l2" = paper eq. (4)
        self.start_step = start_step
        self.ramp_steps = max(1, ramp_steps)
        self.max_points = max_points
        self.backend = ("gsplat" if gsplat_available() else "torch") \
            if backend == "auto" else backend
        self._lpips = None
        self._rng = np.random.default_rng(seed)

        if self.backend == "torch":
            # The reference splatter is O(N x H x W) with no tiling — it exists
            # so `check_render_loss.py` can verify gradient flow without a GPU,
            # not to train with. Say so once, loudly, rather than letting a run
            # appear to hang.
            print(f"  WARNING: L_image is using the pure-PyTorch reference "
                  f"rasterizer (no CUDA gsplat here). It is a VERIFICATION path: "
                  f"cost is O(N x H x W) with no tiling, so at a real N this will "
                  f"be unusably slow. Fine for a smoke run; not a training config.")

        # Standardization constants as tensors, so unstandardizing the
        # prediction stays inside the graph. `preprocessing.unstandardize` is
        # numpy and would detach.
        self.mean = torch.as_tensor(norm_stats["mean"], dtype=torch.float32,
                                     device=self.device)
        self.std = torch.as_tensor(norm_stats["std"], dtype=torch.float32,
                                    device=self.device)

    # ------------------------------------------------------------------
    @property
    def lpips_model(self):
        if self._lpips is None:
            import lpips
            self._lpips = lpips.LPIPS(net="vgg", verbose=False).to(self.device).eval()
            for p in self._lpips.parameters():
                p.requires_grad_(False)   # the VGG is a fixed metric, not a trained part
        return self._lpips

    def weight_at(self, step: int) -> float:
        """0 before `start_step`, then linear to 1 over `ramp_steps`.

        STAGE_SPEC §5: "weight ramped from 0 over the first few thousand
        steps — rendering an untrained prediction produces enormous
        gradients." GaussianCube's own gate is a hard switch at step 50,000,
        which is ~4% of a one-week 16-GPU run; copying 50,000 literally into
        our 60,000-step run would leave the term almost no time to act, so
        the gate is scaled to our run length and the hard switch is softened
        into a ramp. Flagged as a deviation rather than presented as their
        setting.
        """
        if step < self.start_step:
            return 0.0
        return float(min(1.0, (step - self.start_step + 1) / self.ramp_steps))

    # ------------------------------------------------------------------
    def _to_tokens(self, cond: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """(cond, target) standardized -> RAW (N, 14) tokens, differentiable."""
        tok_std = recombine_channels(cond, target, self.cfg)   # torch branch: new_empty
        return tok_std * self.std + self.mean

    def _render(self, tok, viewmats, Ks):
        means, quats, scales, opacities, colors = tokens_to_gaussians(
            tok, device=self.device)
        return rasterize_differentiable(
            means, quats, scales, opacities, colors, viewmats, Ks,
            self.image_size, backend=self.backend)

    def _cameras(self):
        """One random azimuth per call, matching GaussianCube's
        `np.random.randint(0, 150, size=self.load_camera)` — a fresh camera
        each step, not a fixed one, so the term does not overfit a viewpoint.
        Its own RNG, so it cannot perturb the training noise stream."""
        elev = float(self._rng.uniform(-10.0, 30.0))
        azim_offset = float(self._rng.uniform(0.0, 360.0))
        viewmats, Ks = orbit_cameras(n_views=self.n_views, elevations=(elev,),
                                     image_size=self.image_size)
        vm = torch.as_tensor(viewmats, dtype=torch.float32, device=self.device)
        ks = torch.as_tensor(Ks, dtype=torch.float32, device=self.device)
        return self._rotate_rig(vm, azim_offset), ks

    @staticmethod
    def _rotate_rig(viewmats: torch.Tensor, azim_deg: float) -> torch.Tensor:
        """Spin the whole rig about the up axis. Cheaper and less error-prone
        than rebuilding it: a world->cam matrix composed with a world rotation
        is `view @ R_world`, and the up axis is z (STAGE_SPEC §2)."""
        a = np.radians(azim_deg)
        c, s = float(np.cos(a)), float(np.sin(a))
        R = torch.tensor([[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                         dtype=viewmats.dtype, device=viewmats.device)
        return viewmats @ R

    def _subsample(self, n: int):
        """The reference rasterizer is O(N x H x W) with no tiling, so the
        check path caps N. `max_points=0` (the default, and what the gsplat
        backend uses) means no cap."""
        if not self.max_points or n <= self.max_points:
            return None
        idx = self._rng.choice(n, self.max_points, replace=False)
        return torch.as_tensor(np.sort(idx), dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    def __call__(self, pred_y0: torch.Tensor, cond: torch.Tensor,
                 target: torch.Tensor, lengths, step: int) -> dict:
        """`pred_y0` is the MEAN half of the model output (the caller splits
        off the learned variance). Returns scalars; `render_total` is the one
        to add to the diffusion loss."""
        w = self.weight_at(step)
        zero = pred_y0.new_zeros(())
        terms = dict(render_l1=zero, render_lpips=zero, render_total=zero,
                     render_weight=w)
        if w == 0.0:
            return terms

        l1_total, lpips_total, start = [], [], 0
        for n in lengths:
            # A FRESH camera per object, not one shared across the batch.
            # GaussianCube draws `np.random.randint(0, 150, size=load_camera)`
            # per dataset item and `build_single_viewpoint_cam(cam_info, b)`
            # picks object b's own camera, so a batch of 4 sees 4 viewpoints per
            # step rather than 1. Sharing one made the term see 4x fewer
            # viewpoints per unit of compute for no reason.
            viewmats, Ks = self._cameras()
            sl = slice(start, start + n)
            start += n
            sub = self._subsample(n)

            cond_i, pred_i, tgt_i = cond[sl], pred_y0[sl], target[sl]
            if sub is not None:
                cond_i, pred_i, tgt_i = cond_i[sub], pred_i[sub], tgt_i[sub]

            img_pred = self._render(self._to_tokens(cond_i, pred_i), viewmats, Ks)
            # The target render carries no gradient: it is data, and letting a
            # graph hang off it would keep the whole GT render alive through
            # the backward for nothing.
            with torch.no_grad():
                img_gt = self._render(self._to_tokens(cond_i, tgt_i), viewmats, Ks)

            diff = img_pred - img_gt
            # "l1": nn.L1Loss(), i.e. mean |.|, matching train.py:285.
            # "l2": paper eq. (4)'s || . ||_2 -- the NORM, not the squared
            # norm, so sqrt of the sum of squares per image.
            l1_total.append(diff.abs().mean() if self.pixel_norm == "l1"
                            else diff.flatten(1).pow(2).sum(1).sqrt().mean())
            if self.lpips_weight:
                p = img_pred.permute(0, 3, 1, 2)
                g = img_gt.permute(0, 3, 1, 2)
                # [-1, 1], exactly GaussianCube's `self.vgg(pred*2-1, gt*2-1)`.
                lpips_total.append(self.lpips_model(p * 2 - 1, g * 2 - 1).mean())

        l1 = torch.stack(l1_total).mean()
        lp = torch.stack(lpips_total).mean() if lpips_total else zero
        terms["render_l1"] = l1
        terms["render_lpips"] = lp
        terms["render_total"] = w * (self.l1_weight * l1 + self.lpips_weight * lp)
        return terms
