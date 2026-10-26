"""
eval/metrics.py — STAGE_SPEC §9's image metrics, on gsplat renders.

Five numbers, in two families that answer different questions and must not
be read as if they answered the same one:

  **Paired, per-object** — PSNR, SSIM, LPIPS. Prediction and ground truth
  are the SAME object under the SAME cameras, compared image by image.
  This is the family that fits this task: STAGE_SPEC §1 is
  geometry-conditioned appearance synthesis evaluated against ground
  truth, so a per-object reconstruction error is exactly the right thing.

  **Distributional** — FID, KID. Compares the SET of rendered predictions
  against the SET of rendered ground truths. Reported because it is what
  GaussianCube publishes, but see the comparability note below: it is not
  measuring the same experiment they measure.

**How comparable are these to GaussianCube's published numbers?** Stated
plainly rather than left for the reader to assume:

  - PSNR/SSIM/LPIPS: measured the same way (renders of predicted Gaussians
    vs. renders of ground-truth Gaussians, identical cameras). SSIM here
    uses GaussianCube's own constants and window
    (`utils/loss_util.py`: 11x11 Gaussian, sigma 1.5, C1=0.01^2,
    C2=0.03^2), so it is their SSIM, not skimage's. LPIPS is VGG, the net
    their training loss uses. Renderer, resolution and view count still
    differ, so treat these as same-family, not same-number.
  - FID/KID: **not directly comparable.** GaussianCube's ShapeNet FID is
    for *unconditional generation* — generated objects against the real
    dataset. Ours compares generated *appearance on held-out real
    geometry* against those same objects' real appearance, so the two sets
    share their geometry exactly and our FID is bounded far below theirs
    by construction. It is a useful number for ranking our own runs and
    baselines against each other; it is not a number to put in the same
    cell of a table as theirs. Sample size differs too — 170 test objects
    x `n_views` images, against the 10k-50k images FID was calibrated for,
    which biases FID upward. KID is the one to trust at this sample size
    (it is unbiased); both are reported with `n_images` beside them.

A `mean_fill` baseline is scored through the identical pipeline, because
none of these numbers means anything without one: "PSNR 22" is only
interpretable next to what predicting the dataset mean appearance scores.
The deterministic same-network-no-diffusion regression baseline that
STAGE_SPEC §9 calls the most important comparison needs a second training
run and is NOT included here.
"""

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import torch.nn.functional as F

# ----------------------------------------------------------------- SSIM
# GaussianCube's `utils/loss_util.py` (itself the 3DGS/Inria implementation):
# 11x11 Gaussian window, sigma 1.5, C1 = 0.01^2, C2 = 0.03^2, per-channel
# via grouped conv, no downsampling. Reimplemented rather than imported —
# `reference/` is consultation only (TRAINING_SPEC §0) — so that our SSIM is
# the same estimator theirs is, not skimage's (which differs in window and
# in whether it crops the border).
SSIM_C1 = 0.01 ** 2
SSIM_C2 = 0.03 ** 2


def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype):
    coords = torch.arange(window_size, dtype=torch.float64, device=device)
    g = torch.exp(-((coords - window_size // 2) ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())[:, None]
    w2d = (g @ g.T)[None, None]
    return w2d.expand(channels, 1, window_size, window_size).contiguous().to(dtype)


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11,
         sigma: float = 1.5) -> torch.Tensor:
    """(B, C, H, W) in [0, 1] -> (B,) mean SSIM per image."""
    c = img1.shape[-3]
    w = _gaussian_window(window_size, sigma, c, img1.device, img1.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(img1, w, padding=pad, groups=c)
    mu2 = F.conv2d(img2, w, padding=pad, groups=c)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, w, padding=pad, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, w, padding=pad, groups=c) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, w, padding=pad, groups=c) - mu1_mu2

    ssim_map = (((2 * mu1_mu2 + SSIM_C1) * (2 * sigma12 + SSIM_C2))
                / ((mu1_sq + mu2_sq + SSIM_C1) * (sigma1_sq + sigma2_sq + SSIM_C2)))
    return ssim_map.flatten(1).mean(1)


def psnr(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) in [0, 1] -> (B,) PSNR in dB, data range 1.0.

    Per-image, then averaged by the caller — NOT PSNR of the pooled MSE.
    The two differ (log of a mean is not the mean of logs) and the
    per-image form is what every 3DGS paper reports.
    """
    mse = ((img1 - img2) ** 2).flatten(1).mean(1)
    # A perfect match would be +inf and would poison the mean. Clamp at the
    # 8-bit noise floor rather than dropping the image, so the cap is visible
    # in the number (~48 dB) instead of silently changing the sample set.
    return 10.0 * torch.log10(1.0 / mse.clamp(min=1e-5))


# ----------------------------------------------------------------- the suite
class MetricSuite:
    """Accumulates all five metrics over a stream of (prediction, ground
    truth) image sets, one call per object.

    Lazy model construction: LPIPS-VGG and InceptionV3 together are ~800MB
    of weights, and a caller that only wants PSNR/SSIM (the mid-training
    render, say) should not pay for them. Weights are pre-fetched into
    `~/.cache/torch/hub/checkpoints`, which the GPU container sees over
    /home, so no evaluation depends on network access at run time.
    """

    def __init__(self, device, lpips_net: str = "vgg", fid_feature: int = 2048,
                 kid_subset_size: int = 100, backend: str = "unknown"):
        self.device = torch.device(device)
        self.lpips_net = lpips_net
        self.fid_feature = fid_feature
        self.kid_subset_size = kid_subset_size
        self.backend = backend      # which renderer produced the images
        self._lpips = self._fid = self._kid = None
        self._psnr, self._ssim, self._lpips_vals = [], [], []
        self._n_images = 0
        self._per_object = []

    # -- lazy nets --------------------------------------------------
    @property
    def lpips_model(self):
        if self._lpips is None:
            import lpips
            self._lpips = lpips.LPIPS(net=self.lpips_net, verbose=False).to(self.device).eval()
            for p in self._lpips.parameters():
                p.requires_grad_(False)
        return self._lpips

    def _ensure_distribution_metrics(self):
        if self._fid is None:
            from torchmetrics.image.fid import FrechetInceptionDistance
            from torchmetrics.image.kid import KernelInceptionDistance
            # normalize=True: accept float images in [0,1] instead of uint8.
            self._fid = FrechetInceptionDistance(
                feature=self.fid_feature, normalize=True).to(self.device)
            self._kid = KernelInceptionDistance(
                feature=self.fid_feature, subset_size=self.kid_subset_size,
                normalize=True).to(self.device)

    # -- accumulation -----------------------------------------------
    @staticmethod
    def _to_nchw(imgs, device):
        """(V, H, W, 3) float [0,1] numpy or tensor -> (V, 3, H, W) tensor."""
        t = torch.as_tensor(np.asarray(imgs), dtype=torch.float32, device=device)
        return t.permute(0, 3, 1, 2).contiguous().clamp(0.0, 1.0)

    @torch.no_grad()
    def update(self, pred_imgs, gt_imgs, name: str = None,
               distribution: bool = True):
        """One object: its `V` predicted views against its `V` ground-truth
        views, same cameras, same order."""
        p = self._to_nchw(pred_imgs, self.device)
        g = self._to_nchw(gt_imgs, self.device)
        assert p.shape == g.shape, f"view sets differ in shape: {p.shape} vs {g.shape}"

        ps = psnr(p, g)
        ss = ssim(p, g)
        # LPIPS wants [-1, 1] — the same rescale GaussianCube applies in its
        # own training loss (`self.vgg(pred_img*2 - 1., gt_img*2 - 1.)`).
        lp = self.lpips_model(p * 2 - 1, g * 2 - 1).flatten()

        self._psnr.append(ps.cpu().numpy())
        self._ssim.append(ss.cpu().numpy())
        self._lpips_vals.append(lp.cpu().numpy())
        self._n_images += p.shape[0]
        self._per_object.append(dict(
            name=name, psnr=float(ps.mean()), ssim=float(ss.mean()),
            lpips=float(lp.mean()), n_views=int(p.shape[0])))

        if distribution:
            self._ensure_distribution_metrics()
            self._fid.update(g, real=True)
            self._fid.update(p, real=False)
            self._kid.update(g, real=True)
            self._kid.update(p, real=False)

    # -- results ----------------------------------------------------
    def compute(self) -> dict:
        if not self._psnr:
            raise RuntimeError("compute() before any update()")
        # Mean over OBJECTS of the per-object mean over views, not a pooled
        # mean over images: objects have equal weight regardless of view
        # count, and the per-object spread is what says whether a good mean
        # hides a few failures.
        per_obj_psnr = np.array([o["psnr"] for o in self._per_object])
        per_obj_ssim = np.array([o["ssim"] for o in self._per_object])
        per_obj_lpips = np.array([o["lpips"] for o in self._per_object])

        out = dict(
            psnr=float(per_obj_psnr.mean()), psnr_std=float(per_obj_psnr.std()),
            psnr_min=float(per_obj_psnr.min()), psnr_max=float(per_obj_psnr.max()),
            ssim=float(per_obj_ssim.mean()), ssim_std=float(per_obj_ssim.std()),
            lpips=float(per_obj_lpips.mean()), lpips_std=float(per_obj_lpips.std()),
            n_objects=len(self._per_object), n_images=self._n_images,
            render_backend=self.backend, lpips_net=self.lpips_net,
        )
        if self._fid is not None:
            out["fid"] = float(self._fid.compute())
            # KID needs subset_size <= n_samples; torchmetrics raises rather
            # than adjusting, and a raise here would throw away a completed
            # evaluation over the whole test set.
            if self._n_images >= self.kid_subset_size:
                kid_mean, kid_std = self._kid.compute()
                out["kid"] = float(kid_mean)
                out["kid_std"] = float(kid_std)
            else:
                out["kid"] = None
                out["kid_note"] = (f"skipped: {self._n_images} images < "
                                    f"subset_size {self.kid_subset_size}")
            out["fid_note"] = (
                f"{self._n_images} images per set. FID is biased upward at this "
                f"sample size and is NOT comparable to GaussianCube's "
                f"unconditional-generation FID — different experiment, see "
                f"eval/metrics.py's docstring. KID is the unbiased one here.")
        return out

    def per_object_table(self) -> list:
        return sorted(self._per_object, key=lambda o: o["psnr"])

    @property
    def last_object(self) -> dict:
        """The scores from the most recent `update()`. For callers that want
        to annotate an object as they go (the per-row PSNR in
        `experiments/reconstruction_grid.py`) rather than only at the end —
        `per_object_table()` is sorted, so its last entry is the best-scoring
        object, not the most recent one."""
        return self._per_object[-1]


# ----------------------------------------------------------------- the comparison
# GaussianCube's published numbers, read off `reference/2403.19655v4.pdf`
# (Table 3 and Table 6) so the comparison the project wants is against real
# figures rather than remembered ones. Every entry carries what it actually
# measured, because two of the three are NOT the same quantity we compute and
# a table that omits that is worse than no table.
GAUSSIANCUBE_REFERENCE = {
    "shapenet_chair_generation": dict(
        source="Table 3, ShapeNet Chair",
        fid=15.99, kid=9.95e-3,          # paper reports KID in per-mille: 9.95‰
        measures="UNCONDITIONAL generation — sampled objects vs. the real dataset",
        n_images=50_000,                  # FID-50K / KID-50K
        n_train_objects=6775,             # Table 7
        comparable=False,
        why=("different experiment. They generate whole objects from nothing; we "
             "generate appearance ON held-out real geometry, so our two image sets "
             "share their geometry exactly and our FID is bounded far below theirs "
             "by construction. Also 50,000 images vs our few thousand (FID is biased "
             "upward at small n) and 6,775 training objects vs our 1,103."),
    ),
    "shapenet_car_fitting": dict(
        source="Table 6 row D, ShapeNet Car",
        psnr=34.94, ssim=0.9863, lpips=0.0346,
        measures="REPRESENTATION FITTING — how well their 3DGS fits the real renders",
        comparable=False,
        why=("an upper bound on their pipeline, not a generation result: it is the "
             "reconstruction error of the representation itself, before any diffusion "
             "model is involved. Our PSNR/SSIM/LPIPS measure generated appearance "
             "against ground truth. Closest published analogue, different quantity — "
             "and it is Car, not Chair."),
    ),
}


def format_gaussiancube_comparison() -> str:
    """The published numbers with their caveats attached, printed next to
    ours. Deliberately not merged into one table: putting 15.99 and our FID
    in the same column would assert a comparability that does not hold, and
    that is exactly the misreading a presentation slide invites."""
    lines = ["  GaussianCube, published (reference/2403.19655v4.pdf):"]
    g = GAUSSIANCUBE_REFERENCE["shapenet_chair_generation"]
    lines.append(f"    {g['source']:<28} FID {g['fid']:.2f}   KID {g['kid']:.5f}   "
                 f"({g['n_images']:,} images, {g['n_train_objects']:,} train objects)")
    f = GAUSSIANCUBE_REFERENCE["shapenet_car_fitting"]
    lines.append(f"    {f['source']:<28} PSNR {f['psnr']:.2f}  SSIM {f['ssim']:.4f}  "
                 f"LPIPS {f['lpips']:.4f}")
    lines.append("  NEITHER row is a like-for-like comparison with ours:")
    for key in ("shapenet_chair_generation", "shapenet_car_fitting"):
        e = GAUSSIANCUBE_REFERENCE[key]
        lines.append(f"    - {e['source']} measures {e['measures']}")
        lines.append(f"      {e['why']}")
    return "\n".join(lines)


def format_results(tag: str, r: dict) -> str:
    lines = [f"  {tag:<12} PSNR {r['psnr']:6.2f} +- {r['psnr_std']:.2f} dB   "
             f"SSIM {r['ssim']:.4f} +- {r['ssim_std']:.4f}   "
             f"LPIPS {r['lpips']:.4f} +- {r['lpips_std']:.4f}"]
    if "fid" in r:
        kid = "n/a" if r.get("kid") is None else f"{r['kid']:.5f}"
        lines.append(f"  {'':<12} FID  {r['fid']:6.2f}          KID  {kid}"
                     f"          ({r['n_objects']} objects, {r['n_images']} images, "
                     f"renderer={r['render_backend']})")
    return "\n".join(lines)
