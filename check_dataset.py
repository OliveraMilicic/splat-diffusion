"""
check_dataset.py — TRAINING_SPEC §12 phase B check: iterate the DataLoader
for a few hundred batches with num_workers>0 (the actual /dev/shm-prone
path), confirm no crash, and confirm augmentation actually varies between
epochs for the same object.

Run: python check_dataset.py
"""

import time

# TRAINING_SPEC §4 / CLAUDE.md: required before any DataLoader with
# num_workers>0 is constructed, or it crashes confusingly mid-run
# (/dev/shm is 64MB on this node).
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import numpy as np
import torch

from config import TrainConfig
from data.dataset import ChairTokenDataset, build_dataloader
from data.shards import load_shard_store

N_BATCHES = 300
NUM_WORKERS = 4

cfg = TrainConfig(batch_size=16, num_workers=NUM_WORKERS)

# ----------------------------------------------------------------- 1. same-object variation
print("[1] does the same object vary across draws (TRAINING_SPEC §4)?")
store = load_shard_store(cfg.shard_path)
dataset = ChairTokenDataset(store, cfg, split="train")
name0 = dataset.names[0]
print(f"    object: {name0}")

draws = [dataset[0].numpy() for _ in range(4)]
pairwise_diffs = [np.abs(draws[i] - draws[0]).max() for i in range(1, 4)]
print(f"    max |draw[i] - draw[0]| for i=1,2,3: {[f'{d:.4f}' for d in pairwise_diffs]}")
assert all(d > 1e-3 for d in pairwise_diffs), \
    "same object returned (near-)identical tensors across draws — augmentation isn't varying"
print("    PASS — same index gives a materially different tensor every call\n")

# ----------------------------------------------------------------- 2. shapes / dtype
print("[2] single-sample shape/dtype sanity")
sample = dataset[0]
print(f"    shape={tuple(sample.shape)} dtype={sample.dtype} "
      f"finite={torch.isfinite(sample).all().item()}")
assert sample.shape == (cfg.n_gaussians, cfg.channels)
assert torch.isfinite(sample).all()
print("    PASS\n")

# ----------------------------------------------------------------- 3. DataLoader, num_workers>0, no crash
print(f"[3] iterating {N_BATCHES} batches, num_workers={NUM_WORKERS} "
      f"(the actual /dev/shm-prone path)...")
loader = build_dataloader(cfg, split="train")
print(f"    train objects: {len(dataset)}, batches/epoch: {len(loader)}, "
      f"batch_size: {cfg.batch_size}")

t0 = time.time()
n_seen = 0
n_epochs = 0
all_finite = True
while n_seen < N_BATCHES:
    n_epochs += 1
    for i, batch in enumerate(loader):
        if not torch.isfinite(batch).all():
            all_finite = False
            print(f"    non-finite values in batch {i}")
        n_seen += 1
        if n_seen >= N_BATCHES:
            break
dt = time.time() - t0

print(f"    consumed {n_seen} batches across {n_epochs} epoch(s) ({dt:.1f}s, "
      f"{dt / n_seen * 1000:.1f} ms/batch) — worker pool respawns each epoch, "
      f"no crash, all finite: {all_finite}")
assert all_finite, "non-finite values encountered in the DataLoader path"
print("    PASS\n")

# ----------------------------------------------------------------- 4. val split sanity (no augmentation gate needed, just loads)
print("[4] val split loads too")
val_dataset = ChairTokenDataset(store, cfg, split="val")
val_sample = val_dataset[0]
print(f"    val objects: {len(val_dataset)}, sample shape {tuple(val_sample.shape)}")
assert len(val_dataset) > 0
print("    PASS\n")

print("ALL DATASET CHECKS PASSED.")
