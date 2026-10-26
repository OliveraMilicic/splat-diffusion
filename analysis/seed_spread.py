"""Is the colour collapse per-OBJECT or per-NOISE-DRAW?

The per-object scan came out bimodal, not uniformly desaturated: some
colourful test chairs were reproduced at roughly the right chroma, others
fell all the way to ~0.006. Two very different explanations:

  MODE SELECTION   the model represents colour fine, but the deterministic
                   DPM-Solver trajectory lands in whichever basin the single
                   initial noise draw picks. Then the SAME object at
                   different seeds would be sometimes coloured, sometimes
                   grey -- and the fix is the sampler.
  NO INFORMATION   the model genuinely cannot tell what colour this geometry
                   should be, and outputs the conditional mean every time.
                   Then every seed gives the same grey, and the fix is not
                   the sampler.

Also reports HUE error, which chroma alone cannot: a prediction with the
right saturation and the wrong hue is not a success.
"""
import sys
from pathlib import Path as _Path; sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import container_env  # noqa
import numpy as np
import torch
from matplotlib.colors import rgb_to_hsv

from diffusion.sampling import sample as dpm_sample, sample_ddpm
from eval.load_run import load_run
from train.dataset import ChairPointDataset
from stage_config import load_norm_stats

N_PTS, STEPS, SEEDS = 4096, 50, 6
dev = torch.device("cpu")
torch.set_num_threads(48)
model, cfg, meta = load_run("stage3_full", dev)
cfg.n_gaussians = N_PTS
ns = load_norm_stats(cfg.shard_path)
m, s = np.asarray(ns["mean"])[11:14], np.asarray(ns["std"])[11:14]
ci = [cfg.target_ch.index(c) for c in (11, 12, 13)]
ds = ChairPointDataset(cfg, split="test", augment=False)


def raw(t_):
    return np.clip(t_.detach().cpu().numpy()[:, ci] * s + m, 0, 1)


def summarise(a, w):
    """opacity-weighted mean colour, chroma and hue of an (N,3) rgb array"""
    w = w / max(w.sum(), 1e-9)
    mu = (w[:, None] * a).sum(0)
    hsv = rgb_to_hsv(mu[None])[0]
    return float(a.max(1).mean() - a.min(1).mean()), float(hsv[0] * 360), mu


def hue_gap(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


# 0025db2266 collapsed (gt chroma 0.160 -> 0.006); 62b3e723e2 did not
# (0.102 -> 0.108); 1096fae070 was the worst (0.349 -> 0.005).
for idx in (0, 2, 7):
    it = ds[idx]
    cond = it["cond"].to(dev)
    off = torch.tensor([it["n"]], dtype=torch.long, device=dev)
    gt = raw(it["target"])
    w = 1.0 / (1.0 + np.exp(-(it["target"].numpy()[:, cfg.target_ch.index(10)]
                              * ns["std"][10] + ns["mean"][10])))
    g_ch, g_hue, g_mu = summarise(gt, w)
    print(f"\n{'=' * 68}\n[{idx}] {it['name'][-12:]}   GT chroma {g_ch:.3f}  "
          f"hue {g_hue:6.1f}deg  mean rgb {g_mu.round(3)}\n{'=' * 68}")
    for tag, fn, n_seeds in (("dpm", dpm_sample, SEEDS), ("ddpm", sample_ddpm, 3)):
        chs, hues = [], []
        for seed in range(n_seeds):
            with torch.no_grad():
                p = fn(model, cond, off, T=cfg.T, num_steps=STEPS, seed=seed,
                       device=dev)
            pr = raw(p)
            ch, hue, mu = summarise(pr, w)
            chs.append(ch); hues.append(hue)
            print(f"  {tag:4s} seed {seed}: chroma {ch:.4f}  hue {hue:6.1f}deg  "
                  f"(hue err {hue_gap(hue, g_hue):5.1f}deg)  rgb {mu.round(3)}")
        print(f"  {tag:4s} ACROSS SEEDS: chroma {np.mean(chs):.4f} +/- "
              f"{np.std(chs):.4f}   (gt {g_ch:.4f})")
