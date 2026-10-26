"""Does the model READ the hue out of y_t, or GENERATE it from geometry?

Take a magenta chair. Build y_t at t=950 from a DELIBERATELY WRONG colour
(blue / green / grey) while keeping its real geometry as conditioning, and
ask for the one-shot y0.

  returns the WRONG colour  -> it reads the hue off y_t. Then at t=999,
                               where y_t carries none, it can only fall back
                               to E[colour|geometry], and a deterministic
                               sampler feeds that answer back to itself.
  returns the TRUE magenta  -> it has a geometry->colour prior and something
                               else is breaking sampling.
"""
import sys
from pathlib import Path as _Path; sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import container_env  # noqa
import numpy as np, torch
from matplotlib.colors import rgb_to_hsv
from diffusion.schedule import add_noise, make_train_scheduler
from eval.load_run import load_run
from train.dataset import ChairPointDataset
from stage_config import load_norm_stats

dev = torch.device("cpu"); torch.set_num_threads(48)
model, cfg, meta = load_run("stage3_full", dev)
cfg.n_gaussians = 4096
ns = load_norm_stats(cfg.shard_path)
m, s = np.asarray(ns["mean"])[11:14], np.asarray(ns["std"])[11:14]
ci = [cfg.target_ch.index(c) for c in (11, 12, 13)]
sch = make_train_scheduler(T=cfg.T, learn_sigma=cfg.learn_sigma)
ds = ChairPointDataset(cfg, split="test", augment=False)

def stat(a):
    mu = a.mean(0)
    return float(a.max(1).mean()-a.min(1).mean()), float(rgb_to_hsv(mu[None])[0][0]*360), mu

it = ds[7]
cond, tgt = it["cond"].to(dev), it["target"].to(dev)
off = torch.tensor([it["n"]], dtype=torch.long, device=dev)
bi = torch.zeros(it["n"], dtype=torch.long, device=dev)
true_rgb = np.clip(tgt.numpy()[:, ci]*s+m, 0, 1)
print(f"object {it['name'][-12:]}  TRUE  chroma {stat(true_rgb)[0]:.3f}  "
      f"hue {stat(true_rgb)[1]:.0f}deg  rgb {stat(true_rgb)[2].round(3)}\n")

# Replace the object's colour while KEEPING its per-point shading pattern:
# shift every Gaussian's rgb by the same offset, so only the global hue moves.
variants = {"TRUE magenta": None,
            "forced BLUE":  np.array([0.10, 0.10, 0.45]),
            "forced GREEN": np.array([0.10, 0.40, 0.12]),
            "forced GREY":  np.array([0.35, 0.35, 0.35])}
g = torch.Generator().manual_seed(0)
noise = torch.randn(tgt.shape, generator=g)
for t_val in (950, 900):
    print(f"--- t = {t_val} ---------------------------------------------")
    for name, target_mu in variants.items():
        y0 = tgt.clone()
        if target_mu is not None:
            new = np.clip(true_rgb - true_rgb.mean(0) + target_mu, 0, 1)
            y0[:, ci] = torch.from_numpy(((new - m)/s).astype(np.float32))
        t = torch.tensor([t_val], dtype=torch.long, device=dev)
        y_t = add_noise(sch, y0, noise, t, bi)
        with torch.no_grad():
            p = model.split_output(model(cond, y_t, t, off))[0]
        pr = np.clip(p.numpy()[:, ci]*s+m, 0, 1)
        ch, hue, mu = stat(pr)
        fed = stat(np.clip(y0.numpy()[:, ci]*s+m, 0, 1))
        print(f"  fed {name:14s} (hue {fed[1]:5.0f}deg) -> model says "
              f"chroma {ch:.4f}  hue {hue:5.0f}deg  rgb {mu.round(3)}")
    print()
