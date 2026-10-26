"""How far can information travel through our PTv3 in one forward pass?

To output a CONSISTENT object-wide colour, the network has to agree across
the whole chair. Our backbone is 12 blocks of patch attention over
1024-point Morton patches, with no pooling, no downsampling, no global
token and no cross-patch attention -- DESIGN_SPEC §9.2 lists "PTv3's
hierarchical U-Net" as a deliberate cut. Information can only spread
because each block re-serializes with a different axis permutation, and all
four permutations are still SPATIALLY LOCAL.

Measured directly: d(output at point i) / d(input at every point j), by
autograd. If the gradient mass is confined near point i, the network cannot
represent a globally-agreed colour, and the object-mean hue is an average of
many independent local guesses -- which regresses to the dataset mean, i.e.
grey.
"""
import sys
from pathlib import Path as _Path; sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import container_env  # noqa
import numpy as np
import torch

from eval.load_run import load_run
from train.dataset import ChairPointDataset
from stage_config import load_norm_stats

N_PTS = int(sys.argv[1]) if len(sys.argv) > 1 else 32768
dev = torch.device("cpu")
torch.set_num_threads(32)

model, cfg, meta = load_run("stage3_full", dev)
cfg.n_gaussians = N_PTS
ds = ChairPointDataset(cfg, split="test", augment=False)
it = ds[0]
n = it["n"]
cond = it["cond"].to(dev)
y_t = torch.randn(n, cfg.n_target_ch, device=dev)
y_t.requires_grad_(True)
off = torch.tensor([n], dtype=torch.long, device=dev)
t = torch.full((1,), 500, dtype=torch.long, device=dev)
print(f"object {it['name'][-10:]}  N={n}  patch_size={min(cfg.patch_size, n)}  "
      f"-> {n // min(cfg.patch_size, n)} patches")

out = model(cond, y_t, t, off)
pred = model.split_output(out)[0]

# xyz of every point, un-standardized enough to measure distance in the
# normalized object frame (channels 0:3 of cond are x,y,z at stage 3).
ns = load_norm_stats(cfg.shard_path)
xyz = cond[:, :3].detach().numpy() * np.asarray(ns["std"])[:3] + np.asarray(ns["mean"])[:3]

# One probe point near the object centre, and the RGB part of its output.
i = int(np.argmin(((xyz - xyz.mean(0)) ** 2).sum(1)))
rgb_idx = [cfg.target_ch.index(c) for c in (11, 12, 13)]
pred[i, rgb_idx].sum().backward()

g = y_t.grad.abs().sum(1).numpy()          # (N,) influence of each input point
d = np.sqrt(((xyz - xyz[i]) ** 2).sum(1))  # distance from the probe point
extent = float(d.max())
order = np.argsort(d)
cum = np.cumsum(g[order]) / max(g.sum(), 1e-12)

print(f"\nprobe point {i}, object extent (max distance) {extent:.3f}")
print(f"gradient mass within a radius, as a fraction of the object's extent:")
for frac in (0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00):
    r = frac * extent
    k = np.searchsorted(d[order], r)
    print(f"    r <= {frac:4.0%} of extent ({r:.3f}) : "
          f"{cum[min(k, len(cum)-1)]:6.1%} of gradient mass, "
          f"{k / len(d):5.1%} of points")

# The number that matters: how much influence does the far half of the
# object have on this point's colour?
far = d > 0.5 * extent
print(f"\n  points beyond half the extent: {far.mean():5.1%} of the object, "
      f"{g[far].sum() / g.sum():5.1%} of the gradient mass")
print(f"  ratio of mean per-point influence, near half vs far half: "
      f"{g[~far].mean() / max(g[far].mean(), 1e-20):.1f}x")
