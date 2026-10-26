"""analysis/snr_probe.py — at what noise level does the model lose the hue?

Feeds y_t built from the object's TRUE colour at a range of timesteps and
reads the one-shot y0 prediction. The point is to separate "the model is
broken" from "the information is gone": if the hue comes back exactly at
low t and vanishes at high t, the colour pathway works and the failure is
the task.

    python analysis/snr_probe.py                       # default sweep
    python analysis/snr_probe.py --t 900 950 980 999    # the tail
    python analysis/snr_probe.py --objects 7 0 --n-points 4096

See reports/COLOUR_COLLAPSE.md for what the output means.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import container_env  # noqa: F401 -- MUST precede torch

import numpy as np
import torch
from matplotlib.colors import rgb_to_hsv

from diffusion.schedule import add_noise, make_train_scheduler
from eval.load_run import load_run
from stage_config import load_norm_stats
from train.dataset import ChairPointDataset

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="stage3_full")
ap.add_argument("--objects", type=int, nargs="+", default=[7, 0])
ap.add_argument("--t", type=int, nargs="+",
                default=[50, 100, 150, 200, 300, 500, 700, 900, 950, 980, 999])
ap.add_argument("--n-points", type=int, default=4096)
ap.add_argument("--threads", type=int, default=48)
args = ap.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if dev.type == "cpu":
    torch.set_num_threads(args.threads)
model, cfg, meta = load_run(args.run, dev)
cfg.n_gaussians = args.n_points
ns = load_norm_stats(cfg.shard_path)
m, s = np.asarray(ns["mean"])[11:14], np.asarray(ns["std"])[11:14]
ci = [cfg.target_ch.index(c) for c in (11, 12, 13)]
sch = make_train_scheduler(T=cfg.T, learn_sigma=cfg.learn_sigma)
ds = ChairPointDataset(cfg, split="test", augment=False)

# Per-channel std of the HUE component in standardized units, from
# analysis/colour_variance.py's decomposition over the train split.
HUE_SIGMA = 0.257


def stat(a):
    mu = a.mean(0)
    return float(a.max(1).mean() - a.min(1).mean()), float(rgb_to_hsv(mu[None])[0][0] * 360), mu


def gap(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


for idx in args.objects:
    it = ds[idx]
    cond, tgt = it["cond"].to(dev), it["target"].to(dev)
    off = torch.tensor([it["n"]], dtype=torch.long, device=dev)
    bi = torch.zeros(it["n"], dtype=torch.long, device=dev)
    gch, ghue, gmu = stat(np.clip(tgt.cpu().numpy()[:, ci] * s + m, 0, 1))
    print(f"\n{'=' * 74}\n[{idx}] {it['name'][-12:]}  N={it['n']}  "
          f"TRUE chroma {gch:.3f}  hue {ghue:.0f}deg  rgb {gmu.round(3)}\n{'=' * 74}")
    print(f"  {'t':>5} {'SNR/point':>10} {'x N':>9} {'chroma':>9} {'hue':>8} "
          f"{'hue err':>8}  recovered")
    g = torch.Generator().manual_seed(0)
    noise = torch.randn(tgt.shape, generator=g).to(dev)
    for t_val in args.t:
        t = torch.tensor([t_val], dtype=torch.long, device=dev)
        y_t = add_noise(sch, tgt, noise, t, bi)
        with torch.no_grad():
            y0 = model.split_output(model(cond, y_t, t, off))[0]
        ch, hue, _mu = stat(np.clip(y0.cpu().numpy()[:, ci] * s + m, 0, 1))
        ab = float(sch.alphas_cumprod[t_val])
        # Hue is a COLLECTIVE property of all N Gaussians, so the quantity
        # that decides recoverability is the per-point SNR times N, not the
        # per-point SNR. That is why the cliff lands near t=961 at N=4096
        # and near t=986 at N=32768, not at t=153 where a single point's
        # hue SNR crosses 1.
        snr = (ab / max(1 - ab, 1e-30)) * HUE_SIGMA ** 2
        print(f"  {t_val:5d} {snr:10.2e} {snr * it['n']:9.3f} {ch:9.4f} "
              f"{hue:8.1f} {gap(hue, ghue):8.1f}  "
              f"{'YES' if gap(hue, ghue) < 30 and ch > 0.4 * gch else 'no'}")
