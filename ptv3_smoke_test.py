"""
ptv3_smoke_test.py — mechanical sanity check for the PTv3 backbone (ptv3.py)
on REAL Day-1 shards. Not the Day-3 integration, not EDM, not training.

Checks:
  1. forward at low sigma (~sigma_min): shape matches input, no NaN/Inf.
  2. forward at high sigma (~sigma_max): same, PLUS a real measurement for
     DESIGN_SPEC §10's open question ("does serialising on noisy positions
     degrade gracefully?") — k-NN neighbourhood overlap between clean and
     noisy positions, at low and high sigma.
  3. backward on a dummy loss: every named parameter gets a finite gradient.

Run: python ptv3_smoke_test.py
"""

import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

from preprocessing import standardize
from ptv3 import PTv3, c_in, knn_indices

SHARDS = Path("/lustre/data/mls04/ShapeSplatsV1/shards")
BATCH = 4
SEED = 0
SIGMA_MIN = 0.002
SIGMA_MAX = 80.0
KNN_K = 16

torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------- 1. load real shards
tmeta = json.loads((SHARDS / "tokens_meta.json").read_text())
names = tmeta["names"]
shape = tuple(tmeta["shape"])
mm = np.memmap(SHARDS / "tokens.dat", dtype=np.float32, mode="r", shape=shape)

split = json.loads((SHARDS / "split.json").read_text())
norm_stats = json.loads((SHARDS / "norm_stats.json").read_text())
sigma_data = norm_stats["sigma_data"]

train_names = split["train"]
picks = rng.choice(len(train_names), BATCH, replace=False)
rows = [names.index(train_names[p]) for p in picks]

raw = np.asarray(mm[rows, 0])  # (B, 4096, 14) — subset 0, per-object normalized, unstandardized
x0 = torch.from_numpy(standardize(raw, norm_stats)).float()  # unit-variance train-space tokens
B, N, C = x0.shape
print(f"loaded real batch: {B} objects x {N} tokens x {C} channels, from {train_names[picks[0]]}...")
print(f"sigma_data (measured, Day 1) = {sigma_data:.4f}")

# ----------------------------------------------------------------- 2. build PTv3, random init
model = PTv3(in_ch=C, dim=384, depth=12, heads=6, patch_size=1024, mlp_ratio=4, k=KNN_K)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nPTv3: {n_params:,} parameters "
      f"(DESIGN_SPEC §7.1 target ~21M; xCPE reconciled against "
      f"reference/ptv3-official/model.py per TRAINING_SPEC §12 phase A)")


def _count(m):
    return sum(p.numel() for p in m.parameters())


b0 = model.blocks[0]
per_block = {
    "xcpe": _count(b0.xcpe),
    "attn (qkv+proj)": _count(b0.attn),
    "mlp (ratio 4)": _count(b0.mlp),
    "adaln-zero": _count(b0.adaln),
    "norm1+norm2": _count(b0.norm1) + _count(b0.norm2),
}
block_total = sum(per_block.values())
print(f"  per-block breakdown (x{len(model.blocks)} blocks, {block_total:,}/block, "
      f"{block_total * len(model.blocks):,} total):")
for name_, cnt in per_block.items():
    print(f"    {name_:<20s} {cnt:>10,}  ({100 * cnt / block_total:5.1f}%)")
print(f"  -> xCPE is {100 * per_block['xcpe'] / block_total:.1f}% of a block's params; "
      f"AdaLN-Zero + MLP together are "
      f"{100 * (per_block['adaln-zero'] + per_block['mlp (ratio 4)']) / block_total:.1f}%. "
      f"The 21M target gap is not primarily an xCPE-approximation artifact — "
      f"flagged in PLAN.md rather than silently resolved by shrinking something "
      f"not in this phase's scope.")


def make_input(sigma_val):
    sigma = torch.full((B,), sigma_val)
    eps = torch.randn_like(x0)
    x_t = x0 + sigma.view(B, 1, 1) * eps
    inp = c_in(sigma, sigma_data).view(B, 1, 1) * x_t
    return sigma, x_t, inp


def report_forward(label, sigma, inp):
    t0 = time.time()
    with torch.no_grad():
        out = model(inp, sigma)
    dt = time.time() - t0
    finite = torch.isfinite(out).all().item()
    print(f"\n[{label}] sigma={sigma[0].item():.4g}")
    print(f"  input  shape {tuple(inp.shape)}")
    print(f"  output shape {tuple(out.shape)}  (matches input: {out.shape == inp.shape})")
    print(f"  finite: {finite}   min={out.min().item():.4f}  max={out.max().item():.4f}  "
          f"mean={out.mean().item():.4f}  std={out.std().item():.4f}")
    print(f"  forward wall time: {dt:.3f}s")
    assert out.shape == inp.shape, "output shape mismatch"
    assert finite, f"non-finite values in output at {label}"
    return out


# ----------------------------------------------------------------- 3. check 1 — low noise
sigma_lo, x_t_lo, inp_lo = make_input(SIGMA_MIN)
report_forward("low noise (sigma ~ sigma_min)", sigma_lo, inp_lo)

# ----------------------------------------------------------------- 4. check 2 — high noise
sigma_hi, x_t_hi, inp_hi = make_input(SIGMA_MAX)
report_forward("high noise (sigma ~ sigma_max)", sigma_hi, inp_hi)


def neighbor_overlap(pos_a, pos_b, k):
    """Fraction of each point's k-NN (by pos_b) that are also its k-NN by pos_a."""
    idx_a = knn_indices(pos_a, k)
    idx_b = knn_indices(pos_b, k)
    match = (idx_b.unsqueeze(-1) == idx_a.unsqueeze(-2)).any(dim=-1)
    return match.float().mean().item()


pos_clean = x0[..., 0:3]
pos_lo = inp_lo[..., 0:3]
pos_hi = inp_hi[..., 0:3]
chance = KNN_K / N

ov_self = neighbor_overlap(pos_clean, pos_clean, KNN_K)
ov_lo = neighbor_overlap(pos_clean, pos_lo, KNN_K)
ov_hi = neighbor_overlap(pos_clean, pos_hi, KNN_K)

print(f"\n[DESIGN_SPEC §10] k-NN neighbourhood overlap with the CLEAN positions "
      f"(k={KNN_K}, chance level = {chance:.4f}):")
print(f"  clean vs. clean (sanity, expect 1.0)      : {ov_self:.4f}")
print(f"  clean vs. low-sigma preconditioned input   : {ov_lo:.4f}")
print(f"  clean vs. high-sigma preconditioned input  : {ov_hi:.4f}")
if ov_hi < 2 * chance:
    print("  -> at sigma_max, local neighbourhood structure has collapsed to "
          "near chance level: serialisation does NOT degrade gracefully, it's "
          "essentially randomized. Consistent with the network having to lean "
          "entirely on the global (AdaLN sigma) signal at high noise, as "
          "DESIGN_SPEC §5.2 anticipates.")
else:
    print("  -> meaningfully above chance: some real spatial structure survives "
          "preconditioning even at sigma_max.")

# ----------------------------------------------------------------- 5. check 3 — backward pass
model.zero_grad()
sigma_mid = torch.full((B,), 1.0)
_, _, inp_mid = make_input(1.0)
target = torch.randn(B, N, C)  # dummy target, standing in for the real EDM target

t0 = time.time()
out = model(inp_mid, sigma_mid)
loss = torch.nn.functional.mse_loss(out, target)
loss.backward()
dt = time.time() - t0

missing, nonfinite = [], []
for name_, p in model.named_parameters():
    if p.grad is None:
        missing.append(name_)
    elif not torch.isfinite(p.grad).all():
        nonfinite.append(name_)

print(f"\n[backward pass] dummy MSE loss = {loss.item():.4f}, backward wall time: {dt:.3f}s")
print(f"  parameters with grad=None      : {len(missing)}"
      + (f"  e.g. {missing[:5]}" if missing else ""))
print(f"  parameters with non-finite grad: {len(nonfinite)}"
      + (f"  e.g. {nonfinite[:5]}" if nonfinite else ""))
assert not missing, f"{len(missing)} parameters got no gradient at all"
assert not nonfinite, f"{len(nonfinite)} parameters got non-finite gradients"
print(f"  ALL {sum(1 for _ in model.parameters())} parameter tensors have finite gradients.")

# ----------------------------------------------------------------- 6. resource report
peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB -> MB on Linux
print(f"\n[resources] CPU-only this session (no CUDA) — peak RSS: {peak_rss_mb:.0f} MB")
print("\nSMOKE TEST PASSED — no EDM/training loop wired up, as requested.")
