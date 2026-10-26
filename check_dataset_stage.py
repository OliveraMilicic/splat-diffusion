"""
check_dataset_stage.py — STAGE_SPEC §11 build-order step 2 checks, for
`train/dataset.py` (distinct from `check_dataset.py`, which checks the
old FPS@4096 `data/` package).

Run: python check_dataset_stage.py [--stage 1] [--n3-objects 1103]
"""

import argparse
import time

import container_env  # noqa: F401 -- MUST precede torch, see its docstring

# CLAUDE.md / TRAINING_SPEC §4: required before any DataLoader with
# num_workers>0 is constructed (/dev/shm is 64MB on this node).
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import numpy as np
import torch

from stage_config import StageConfig
from train.dataset import ChairPointDataset, build_dataloader

ap = argparse.ArgumentParser()
ap.add_argument("--stage", type=int, default=1)
ap.add_argument("--n-gaussians", type=int, default=16384)
ap.add_argument("--n-batches", type=int, default=100)
ap.add_argument("--num-workers", type=int, default=4)
ap.add_argument("--n3-objects", type=int, default=1103, help="0 = all train objects")
args = ap.parse_args()

cfg = StageConfig(stage=args.stage, n_gaussians=args.n_gaussians,
                   batch_size=8, num_workers=args.num_workers)
print(f"stage {cfg.stage}: target_ch={cfg.target_ch}  cond_ch={cfg.cond_ch}  "
      f"n_gaussians={cfg.n_gaussians}\n")

# ----------------------------------------------------------------- 1. augmentation varies
print("[1] does the same train object vary across draws?")
train_ds = ChairPointDataset(cfg, split="train", augment=True)
name0 = train_ds.names[0]
draws = [train_ds[0] for _ in range(4)]
ns = [d["n"] for d in draws]
# compare on the shared min-n prefix isn't meaningful post-shuffle-subsample;
# instead compare summary stats that should differ under mirror/rotate/resample.
means = [d["cond"][:, :3].mean(0).numpy() for d in draws]  # mean xyz cond channels
pairwise = [float(np.abs(means[i] - means[0]).max()) for i in range(1, 4)]
print(f"    object: {name0}  n per draw: {ns}")
print(f"    max |mean(xyz)[i] - mean(xyz)[0]| for i=1,2,3: {[f'{d:.4f}' for d in pairwise]}")
assert any(d > 1e-3 for d in pairwise), "same object gives (near-)identical draws — augmentation isn't varying"
print("    PASS\n")

# ----------------------------------------------------------------- 2. single-sample shape/dtype
print("[2] single-sample shape/dtype sanity")
sample = train_ds[0]
n_cond, n_target = cfg.n_cond_ch, cfg.n_target_ch
print(f"    cond {tuple(sample['cond'].shape)}  target {tuple(sample['target'].shape)}  n={sample['n']}")
assert sample["cond"].shape == (sample["n"], n_cond)
assert sample["target"].shape == (sample["n"], n_target)
assert torch.isfinite(sample["cond"]).all() and torch.isfinite(sample["target"]).all()
print("    PASS\n")

# ----------------------------------------------------------------- 3. variable-N batching
print("[3] variable-N batching (STAGE_SPEC §3/§4, no padding)")
loader = build_dataloader(cfg, split="train")
batch = next(iter(loader))
lengths = batch["lengths"].numpy()
print(f"    batch lengths: min={lengths.min()} mean={lengths.mean():.0f} max={lengths.max()}")
print(f"    offset: {batch['offset'].tolist()}")
print(f"    cond {tuple(batch['cond'].shape)}  target {tuple(batch['target'].shape)}  "
      f"batch_idx {tuple(batch['batch_idx'].shape)}")
assert batch["cond"].shape[0] == lengths.sum() == batch["offset"][-1].item()
assert batch["batch_idx"].shape[0] == lengths.sum()
assert (batch["batch_idx"].bincount() == batch["lengths"]).all()
n_batches_at_full_N = 0
for i, b in enumerate(loader):
    if i >= 20:
        break
    if b["lengths"].min().item() == cfg.n_gaussians:
        n_batches_at_full_N += 1
print(f"    of 20 batches, {n_batches_at_full_N} have every object clipped to the full "
      f"{cfg.n_gaussians} — the rest genuinely mix shorter objects")
print("    PASS\n")

# ----------------------------------------------------------------- 4. DataLoader, num_workers>0, no crash
print(f"[4] iterating {args.n_batches} batches, num_workers={args.num_workers} "
      f"(the actual /dev/shm-prone path)...")
t0 = time.time()
n_seen = 0
all_finite = True
while n_seen < args.n_batches:
    for b in loader:
        if not (torch.isfinite(b["cond"]).all() and torch.isfinite(b["target"]).all()):
            all_finite = False
        n_seen += 1
        if n_seen >= args.n_batches:
            break
dt = time.time() - t0
print(f"    consumed {n_seen} batches ({dt:.1f}s, {dt / n_seen * 1000:.0f} ms/batch), "
      f"no crash, all finite: {all_finite}")
assert all_finite
print("    PASS\n")

# ----------------------------------------------------------------- 5. val split loads
print("[5] val split loads")
val_ds = ChairPointDataset(cfg, split="val", augment=False)
val_sample = val_ds[0]
print(f"    val objects: {len(val_ds)}, sample cond {tuple(val_sample['cond'].shape)}")
assert len(val_ds) > 0
print("    PASS\n")

# ----------------------------------------------------------------- N3. within vs between std
print(f"[N3] within_object_std vs between_object_std (NORMALIZATION.md §2)")
n3_ds = ChairPointDataset(cfg, split="train", augment=False)
n_objects = len(n3_ds) if args.n3_objects == 0 else min(args.n3_objects, len(n3_ds))
print(f"    measuring over {n_objects} train objects, standardized RGB, N={cfg.n_gaussians}...")

# Per-channel, not pooled across r/g/b: NORMALIZATION.md's "one colour
# repeated N times" argument is about a single channel's value being
# near-constant across an object's Gaussians. Pooling r,g,b together
# instead measures "does this object's R differ from its G" — real, large,
# and unrelated to the redundancy phenomenon; it dominates the pooled
# number and inverts the conclusion (verified: pooled gives within=0.86 >
# between=0.38, i.e. the OPPOSITE of what NORMALIZATION.md's narrative
# describes — a real "measured number is off from what the spec assumed"
# per CLAUDE.md, traced to this pooling choice, not a data problem).
t0 = time.time()
within_per_channel, mean_per_channel = [], []
for i in range(n_objects):
    rgb = n3_ds[i]["target"].numpy()  # (n, 3) standardized RGB, stage 1
    within_per_channel.append(rgb.std(axis=0))   # (3,) std over the n points, per channel
    mean_per_channel.append(rgb.mean(axis=0))    # (3,) mean over the n points, per channel
    if (i + 1) % 200 == 0:
        print(f"      {i + 1}/{n_objects}  ({time.time() - t0:.0f}s)")

within_per_channel = np.stack(within_per_channel)  # (n_objects, 3)
mean_per_channel = np.stack(mean_per_channel)       # (n_objects, 3)

within_object_std = within_per_channel.mean(axis=0)   # (3,): mean over objects of within-object std
between_object_std = mean_per_channel.std(axis=0)      # (3,): std over objects of within-object mean

print(f"\n    per-channel (r, g, b):")
print(f"    within_object_std  = {within_object_std.tolist()}")
print(f"    between_object_std = {between_object_std.tolist()}")
print(f"    ratio between/within = {(between_object_std / within_object_std).tolist()}")
print(f"    mean over channels: within={within_object_std.mean():.4f}  "
      f"between={between_object_std.mean():.4f}  "
      f"ratio={between_object_std.mean() / within_object_std.mean():.2f}x")
if (between_object_std > within_object_std).all():
    print("    -> between >> within on every channel: redundancy problem is REAL "
          "(NORMALIZATION.md §2/§3/§4 apply)")
elif (within_object_std > between_object_std).all():
    print("    -> within > between on EVERY channel — the OPPOSITE of NORMALIZATION.md "
          "§2's stated expectation ('between >> within (expect this)'). Flagging per "
          "CLAUDE.md rather than silently treating this as 'mild': objects are NOT "
          "one-or-two-colour with 16,384 near-duplicate values — likely because these "
          "3DGS reconstructions bake real per-Gaussian shading/AO/texture into colour, "
          "not flat per-part colour. NORMALIZATION.md §2's collapse-to-averaging concern "
          "does not apply as stated; do NOT add §4's input scaling pre-emptively — its "
          "own rule (act on a flat loss-vs-t curve, not this number) still governs.")
else:
    print("    -> within ~ between: objects are internally varied, problem is mild")

print("\nALL DATASET-STAGE CHECKS PASSED.")
