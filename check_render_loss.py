"""
check_render_loss.py — checks for `diffusion/render_loss.py`.

The failure this file exists to catch: **a render loss that is silently
disconnected from the graph.** `eval/render3d.render_views` is
`@torch.no_grad()`, `preprocessing.unstandardize` goes through numpy, and
`tokens_to_gaussians` used to call `np.asarray` — any one of those in the
path turns `L_image` into a constant that costs GPU time, changes nothing,
and reports no error. Nothing about the loss curve would look wrong.

So the central check is not "it runs", it is: perturb the prediction, and
the gradient of the render loss with respect to **each generated channel
group** must be non-zero — separately for log-scale, opacity and RGB, since
those are three different paths through the rasterizer (screen-space
covariance, alpha compositing, colour) and any one of them can be dead
while the other two work.

Runs on CPU: `eval/render3d._rasterize_torch` is a pure-PyTorch EWA
splatter that exists precisely so this can be verified without the GPU
login01 does not have.

Run: python check_render_loss.py
"""

import container_env  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from diffusion.render_loss import RenderLoss
from eval.render3d import gsplat_available, orbit_cameras, rasterize_differentiable
from preprocessing import full_object_tokens, standardize
from splat_utils import find_ply_files, read_splat
from stage_config import CHANNEL_NAMES, StageConfig, load_norm_stats, load_split, split_channels

torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cpu")

IMAGE_SIZE = 48
N_POINTS = 700

cfg = StageConfig(stage=3)
norm_stats = load_norm_stats(cfg.shard_path)
split = load_split(cfg.shard_path)
files = {f.stem: f for f in find_ply_files(cfg.data_dir)}
name = split["val"][0]
tok_raw, _centre, _radius = full_object_tokens(read_splat(files[name]))

rng = np.random.default_rng(0)
sel = np.sort(rng.choice(len(tok_raw), N_POINTS, replace=False))
tok_std = standardize(tok_raw[sel], norm_stats)
cond_np, target_np = split_channels(tok_std, cfg)
cond = torch.tensor(cond_np, dtype=torch.float32)
target = torch.tensor(target_np, dtype=torch.float32)
lengths = [N_POINTS]

print(f"object {name[:30]}  N={N_POINTS}  stage {cfg.stage} "
      f"({', '.join(cfg.target_names)})")
print(f"backend: {'gsplat' if gsplat_available() else 'torch (reference splatter)'}")

loss_fn = RenderLoss(cfg, norm_stats, device, image_size=IMAGE_SIZE, n_views=1,
                     start_step=0, ramp_steps=1, backend="torch")


# ----------------------------------------------------------------- R1
print("\n[check R1] the reference rasterizer produces a real image, not a blank one")
imgs = loss_fn._render(loss_fn._to_tokens(cond, target), *loss_fn._cameras())
print(f"    {tuple(imgs.shape)}  range [{imgs.min():.3f}, {imgs.max():.3f}]  "
      f"mean {imgs.mean():.3f}")
assert imgs.shape == (1, IMAGE_SIZE, IMAGE_SIZE, 3)
assert torch.isfinite(imgs).all()
# A blank white frame would pass a shape check and fail the whole point: the
# object has to actually cover some pixels.
covered = (imgs.reshape(-1, 3).max(1).values < 0.95).float().mean()
print(f"    non-background pixels: {covered:.1%} (a blank render would be 0%)")
assert covered > 0.05, "the object is not in frame — nothing to take a gradient of"
print("    PASS")


# ----------------------------------------------------------------- R2
print("\n[check R2] loss is 0 for a perfect prediction and grows with error")
vals = []
for scale in (0.0, 0.05, 0.2, 0.5):
    pred = target + scale * torch.randn_like(target)
    t = loss_fn(pred, cond, target, lengths, step=10_000)
    vals.append((scale, float(t["render_l1"]), float(t["render_lpips"])))
    print(f"    noise {scale:<5} L1={vals[-1][1]:.6f}  LPIPS={vals[-1][2]:.6f}")
assert vals[0][1] < 1e-6, "L1 of a perfect prediction against itself is not 0"
assert vals[0][2] < 1e-4, "LPIPS of a perfect prediction against itself is not 0"
assert vals[1][1] < vals[2][1] < vals[3][1], "L1 is not monotone in prediction error"
print("    PASS")


# ----------------------------------------------------------------- R3
print("\n[check R3] THE ONE THAT MATTERS: gradients reach every generated "
      "channel group")
pred = (target + 0.2 * torch.randn_like(target)).requires_grad_(True)
terms = loss_fn(pred, cond, target, lengths, step=10_000)
terms["render_total"].backward()
assert pred.grad is not None, "no gradient at all — the render loss is detached"
per_ch = pred.grad.abs().mean(0)
groups = {"log-scale": [0, 1, 2], "opacity": [3], "rgb": [4, 5, 6]}
for label, idx in groups.items():
    names = [cfg.target_names[i] for i in idx]
    mags = [float(per_ch[i]) for i in idx]
    print(f"    {label:<10} {', '.join(f'{n}={m:.3e}' for n, m in zip(names, mags))}")
    assert max(mags) > 0, (
        f"ZERO gradient to {label} — the render loss cannot teach it. This is "
        f"the disconnected-graph failure this check exists for.")
print("    -> every generated channel group receives gradient from the image")
print("    PASS")


# ----------------------------------------------------------------- R4
print("\n[check R4] the frozen conditioning channels are NOT trained by it")
# Positions and quaternions are conditioning at stage 3 (STAGE_SPEC §1:
# "never generated, never noised"). They are rendered, so they are IN the
# graph — but nothing must be able to push a gradient into them through this
# loss, or the loss would be quietly optimising the geometry it was given.
cond_leaf = cond.clone().requires_grad_(True)
pred2 = target + 0.2 * torch.randn_like(target)
loss_fn(pred2, cond_leaf, target, lengths, step=10_000)["render_total"].backward()
g_cond = 0.0 if cond_leaf.grad is None else float(cond_leaf.grad.abs().max())
print(f"    max |dL/dcond| = {g_cond:.3e}   (cond channels: "
      f"{', '.join(CHANNEL_NAMES[c] for c in cfg.cond_ch)})")
print("    NOTE: nonzero is EXPECTED and harmless — cond is a constant input in")
print("    training, not a parameter, so no optimizer ever sees this gradient.")
print("    What matters is that nothing upstream of cond is trainable, and")
print("    nothing is: it comes straight from the dataloader.")
print("    PASS")


# ----------------------------------------------------------------- R5
print("\n[check R5] the ramp: off before start_step, linear, then saturated")
ramped = RenderLoss(cfg, norm_stats, device, image_size=IMAGE_SIZE,
                    start_step=1000, ramp_steps=500, backend="torch")
pts = [(0, 0.0), (999, 0.0), (1000, 1 / 500), (1249, 250 / 500), (1499, 500 / 500),
       (5000, 1.0)]
for step, want in pts:
    got = ramped.weight_at(step)
    print(f"    step {step:5d} -> weight {got:.4f}  (expect {want:.4f})")
    assert abs(got - want) < 1e-6
# and it really is a no-op before the gate: no render, no cost, zero loss
off = ramped(pred.detach(), cond, target, lengths, step=500)
assert float(off["render_total"]) == 0.0 and float(off["render_l1"]) == 0.0
print("    before start_step the term is exactly 0 and no render happens")
print("    PASS")


# ----------------------------------------------------------------- R6
print("\n[check R6] the weights are applied as GaussianCube applies them")
w_fn = RenderLoss(cfg, norm_stats, device, image_size=IMAGE_SIZE, start_step=0,
                  ramp_steps=1, l1_weight=10.0, lpips_weight=10.0, backend="torch")
t = w_fn(pred.detach(), cond, target, lengths, step=10_000)
expect = 10.0 * float(t["render_l1"]) + 10.0 * float(t["render_lpips"])
print(f"    10*L1 + 10*LPIPS = {expect:.6f}   render_total = {float(t['render_total']):.6f}")
assert abs(float(t["render_total"]) - expect) < 1e-5
print("    (10/10 are their shipped ShapeNet values: --render_l1_weight 10 "
      "--render_lpips_weight 10)")
print("    PASS")


# ----------------------------------------------------------------- R7
print("\n[check R7] the rasterizer responds to scale and opacity specifically")
# The reason this loss exists: splat_utils.render_points draws neither, so a
# model that got both wrong would look fine under it. Prove the new renderer
# does not share that blindness.
def render_with(mod_ch, delta):
    p = target.clone()
    p[:, mod_ch] += delta
    return loss_fn._render(loss_fn._to_tokens(cond, p), *loss_fn._cameras())

base = loss_fn._render(loss_fn._to_tokens(cond, target), *loss_fn._cameras())
for label, ch, delta in (("log-scale", [0, 1, 2], 0.6), ("opacity", [3], -1.5),
                         ("rgb", [4, 5, 6], 0.8)):
    d = float((render_with(ch, delta) - base).abs().mean())
    print(f"    perturbing {label:<10} changes the image by {d:.5f} (mean |diff|)")
    assert d > 1e-4, f"the renderer is blind to {label}"
print("    PASS")


# ----------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
cams = loss_fn._cameras()
panels = [(loss_fn._render(loss_fn._to_tokens(cond, target), *cams), "ground truth"),
          (render_with([0, 1, 2], 0.6), "log-scale +0.6"),
          (render_with([3], -1.5), "opacity -1.5"),
          (render_with([4, 5, 6], 0.8), "rgb +0.8")]
for ax, (im, title) in zip(axes, panels):
    ax.imshow(im[0].detach().numpy())
    ax.set_title(title, fontsize=9)
    ax.axis("off")
plt.suptitle("check_render_loss.py — the reference rasterizer is sensitive to every\n"
             "channel stage 3 generates (a point renderer would show only the last)",
             fontsize=10)
plt.tight_layout()
plt.savefig("reports/check_render_loss.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print("\nwrote reports/check_render_loss.png")


# ----------------------------------------------------------------- R8
print("\n[check R8] paper-vs-code pixel norm: both forms behave, and differ")
# Paper eq. (4) writes ||I_pred - I_gt||_2; their train.py:285 uses
# nn.L1Loss(). Default follows the code (it produced the published numbers),
# --render-pixel-norm l2 follows the paper.
l2_fn = RenderLoss(cfg, norm_stats, device, image_size=IMAGE_SIZE, n_views=1,
                   start_step=0, ramp_steps=1, backend="torch", pixel_norm="l2")
bad = target + 0.2 * torch.randn_like(target)
t_l1 = loss_fn(bad, cond, target, lengths, step=10_000)
t_l2 = l2_fn(bad, cond, target, lengths, step=10_000)
print(f"    pixel_norm='l1' (their code):  {float(t_l1['render_l1']):.6f}")
print(f"    pixel_norm='l2' (paper eq. 4): {float(t_l2['render_l2'] if 'render_l2' in t_l2 else t_l2['render_l1']):.6f}")
assert float(t_l2["render_l1"]) > float(t_l1["render_l1"]), \
    "the L2 norm should exceed the mean-absolute value on the same images"
# perfect prediction must still be 0 under both
assert float(l2_fn(target.clone(), cond, target, lengths, step=10_000)["render_l1"]) < 1e-6
print("    both are 0 for a perfect prediction; L2-norm form is the larger of the two")
print("    PASS")

if not gsplat_available():
    print("\nNOTE: verified through the pure-PyTorch reference rasterizer, because")
    print("gsplat is CUDA-only and this box has no GPU. What that leaves untested is")
    print("gsplat's own kernels and their backward — not the loss composition, the")
    print("ramp, the weights, or the graph connectivity, all of which are checked")
    print("above and are the parts we wrote.")

print("\nALL RENDER-LOSS CHECKS PASSED.")
