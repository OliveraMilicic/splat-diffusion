"""
check_metrics.py — checks for `eval/metrics.py`.

A metric that is silently wrong produces a plausible number and no error,
so each check here pins a value that is known independently of the
implementation: PSNR against its closed form on a known-MSE pair, SSIM
against its known value of 1.0 on identical images and against `torchmetrics`'
independent implementation, LPIPS against 0 on identical images and its
ordering property, FID against 0 for identical sets.

Run: python check_metrics.py
"""

import container_env  # noqa: F401

import numpy as np
import torch

from eval.metrics import MetricSuite, format_results, psnr, ssim

torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

B, C, H, W = 4, 3, 64, 64
a = torch.rand(B, C, H, W, device=device)


# ----------------------------------------------------------------- M1
print("\n[check M1] PSNR matches its closed form at a known MSE")
for target_mse in (0.01, 0.0025, 0.16):
    # A constant offset gives EXACTLY the requested MSE. Deliberately not
    # clamped to [0,1] — clamping would change the MSE and break the
    # closed-form comparison, which is the whole point of this check.
    b = a + np.sqrt(target_mse)
    got = psnr(a, b).mean().item()
    want = 10 * np.log10(1.0 / target_mse)
    print(f"    mse={target_mse:<8} psnr={got:7.4f} dB   closed form={want:7.4f} dB")
    assert abs(got - want) < 1e-3
# The +inf guard: identical images must not return inf and poison a mean.
same = psnr(a, a.clone())
print(f"    identical images -> {same.mean().item():.2f} dB (capped, not inf)")
assert torch.isfinite(same).all()
print("    PASS")


# ----------------------------------------------------------------- M2
print("\n[check M2] SSIM = 1 on identical images, and agrees with torchmetrics")
s_same = ssim(a, a.clone())
print(f"    identical images -> SSIM {s_same.mean().item():.8f} (must be 1.0)")
assert torch.allclose(s_same, torch.ones_like(s_same), atol=1e-5)

from torchmetrics.functional.image import structural_similarity_index_measure as tm_ssim  # noqa: E402
noisy = (a + 0.1 * torch.randn_like(a)).clamp(0, 1)
ours = ssim(a, noisy).mean().item()
# torchmetrics' default matches the same Wang et al. estimator: 11x11
# Gaussian window, sigma 1.5, K1=0.01, K2=0.03. Same knobs, independent code.
theirs = tm_ssim(noisy, a, data_range=1.0, gaussian_kernel=True,
                 kernel_size=11, sigma=1.5, k1=0.01, k2=0.03).item()
print(f"    noisy pair: ours={ours:.6f}  torchmetrics={theirs:.6f}  "
      f"diff={abs(ours - theirs):.2e}")
# Not bit-identical: torchmetrics crops the border by the window radius,
# ours pads (GaussianCube's implementation pads too). Same estimator, a
# different edge convention, so agreement to ~1e-2 is the right bar.
assert abs(ours - theirs) < 2e-2, "SSIM disagrees with an independent implementation"
# Monotonicity is the property that actually matters for ranking runs.
worse = (a + 0.3 * torch.randn_like(a)).clamp(0, 1)
print(f"    SSIM(noise 0.1)={ours:.4f} > SSIM(noise 0.3)={ssim(a, worse).mean().item():.4f}")
assert ours > ssim(a, worse).mean().item()
print("    PASS")


# ----------------------------------------------------------------- M3
print("\n[check M3] LPIPS: 0 on identical images, ordered by corruption")
suite = MetricSuite(device, kid_subset_size=8, backend="synthetic")
lp = suite.lpips_model
def lp_of(x, y):
    return lp(x * 2 - 1, y * 2 - 1).flatten().mean().item()
l_same = lp_of(a, a.clone())
l_small = lp_of(a, noisy)
l_big = lp_of(a, worse)
print(f"    identical={l_same:.6f}  noise0.1={l_small:.4f}  noise0.3={l_big:.4f}")
assert l_same < 1e-4, "LPIPS of an image with itself is not ~0"
assert l_same < l_small < l_big
print("    PASS")


# ----------------------------------------------------------------- M4
print("\n[check M4] FID = 0 for identical sets, > 0 for different ones")
def run_suite(pred, gt, ksub=8):
    s = MetricSuite(device, kid_subset_size=ksub, backend="synthetic")
    for i in range(pred.shape[0]):
        s.update(pred[i], gt[i], name=f"obj{i}")
    return s.compute()

n_obj, n_view = 6, 8
gt = torch.rand(n_obj, n_view, 32, 32, 3, device=device)
r_same = run_suite(gt, gt)
print(f"    identical sets: FID={r_same['fid']:.5f}  KID={r_same['kid']:.6f} "
      f"PSNR={r_same['psnr']:.2f}  SSIM={r_same['ssim']:.4f}")
assert r_same["fid"] < 1e-2, "FID of a set against itself is not ~0"
assert abs(r_same["ssim"] - 1.0) < 1e-4

pred = (gt + 0.35 * torch.randn_like(gt)).clamp(0, 1)
r_diff = run_suite(pred, gt)
print(f"    corrupted set : FID={r_diff['fid']:.4f}  KID={r_diff['kid']:.6f} "
      f"PSNR={r_diff['psnr']:.2f}  SSIM={r_diff['ssim']:.4f}")
assert r_diff["fid"] > r_same["fid"]
assert r_diff["kid"] > r_same["kid"]
assert r_diff["psnr"] < r_same["psnr"]
assert r_diff["lpips"] > r_same["lpips"]
print("    PASS")


# ----------------------------------------------------------------- M5
print("\n[check M5] KID is skipped, not crashed, when there are too few images")
s = MetricSuite(device, kid_subset_size=10_000, backend="synthetic")
s.update(gt[0], gt[0], name="one")
r = s.compute()
print(f"    n_images={r['n_images']} < subset_size -> kid={r['kid']}  ({r['kid_note']})")
assert r["kid"] is None and "skipped" in r["kid_note"]
print("    PASS")

print("\n" + format_results("synthetic", r_diff))
print("\nALL METRIC CHECKS PASSED.")
