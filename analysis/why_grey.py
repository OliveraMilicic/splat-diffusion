"""What is the trained stage3_full model actually getting wrong about colour?

Decomposes the sampled-colour error into the two parts the dataset
decomposition says are different problems:

    GLOBAL   the object's mean colour  -> "what colour is this chair"
    LOCAL    per-Gaussian deviation from that mean -> shading and texture

and reports predicted vs true chroma, so "collapsing to grey" becomes a
number rather than an impression. Runs both samplers: DPM-Solver
(deterministic ODE) and DDPM ancestral (stochastic).

CPU, reduced N -- this measures WHICH kind of error dominates, a ratio that
does not depend on N, not the headline metric.
"""
import sys, time, json
from pathlib import Path as _Path; sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import container_env  # noqa
import numpy as np
import torch

from diffusion.sampling import sample as dpm_sample, sample_ddpm
from eval.load_run import load_run
from train.dataset import ChairPointDataset
from stage_config import load_norm_stats

N_OBJ = int(sys.argv[1]) if len(sys.argv) > 1 else 10
N_PTS = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 25

dev = torch.device("cpu")
model, cfg, meta = load_run("stage3_full", dev)
cfg.n_gaussians = N_PTS
ns = load_norm_stats(cfg.shard_path)
rgb_mean = np.asarray(ns["mean"])[11:14]
rgb_std = np.asarray(ns["std"])[11:14]
# where r,g,b sit inside the stage-3 TARGET vector (log_s x3, opacity, r,g,b)
ci = [cfg.target_ch.index(c) for c in (11, 12, 13)]
print(f"stage {cfg.stage} target {cfg.target_names}; rgb at target idx {ci}")

ds = ChairPointDataset(cfg, split="test", augment=False)
print(f"{len(ds)} test objects, using {N_OBJ} at N={N_PTS}, {STEPS} steps\n")


def raw_rgb(std_target):
    """standardized target -> raw [0,1] rgb"""
    a = std_target.detach().cpu().numpy()[:, ci]
    return np.clip(a * rgb_std + rgb_mean, 0, 1)


def chroma(x):
    return float((x.max(1) - x.min(1)).mean())


rows = []
t0 = time.time()
for i in range(N_OBJ):
    it = ds[i]
    cond = it["cond"].to(dev)
    off = torch.tensor([it["n"]], dtype=torch.long, device=dev)
    gt = raw_rgb(it["target"])
    out = dict(name=it["name"], n=int(it["n"]))

    for tag, fn in (("dpm", dpm_sample), ("ddpm", sample_ddpm)):
        with torch.no_grad():
            pred = fn(model, cond, off, T=cfg.T, num_steps=STEPS, seed=0, device=dev)
        pr = raw_rgb(pred)
        g_mu, p_mu = gt.mean(0), pr.mean(0)          # object mean colour
        g_loc, p_loc = gt - g_mu, pr - p_mu          # deviation from it
        out[tag] = dict(
            gt_chroma=chroma(gt), pred_chroma=chroma(pr),
            gt_mu_chroma=float(g_mu.max() - g_mu.min()),
            pred_mu_chroma=float(p_mu.max() - p_mu.min()),
            err_global=float(((p_mu - g_mu) ** 2).sum()),
            err_local=float(((p_loc - g_loc) ** 2).sum(1).mean()),
            var_global=float(((g_mu - rgb_mean) ** 2).sum()),
            var_local=float((g_loc ** 2).sum(1).mean()),
        )
    rows.append(out)
    d = out["dpm"]
    print(f"  [{i+1:2d}/{N_OBJ}] {it['name'][-10:]}  "
          f"chroma gt {d['gt_chroma']:.3f} -> dpm {d['pred_chroma']:.3f} / "
          f"ddpm {out['ddpm']['pred_chroma']:.3f}   ({time.time()-t0:.0f}s)")

print()
for tag in ("dpm", "ddpm"):
    R = [r[tag] for r in rows]
    eg = np.mean([x["err_global"] for x in R])
    el = np.mean([x["err_local"] for x in R])
    vg = np.mean([x["var_global"] for x in R])
    vl = np.mean([x["var_local"] for x in R])
    print(f"--- {tag.upper()} ---------------------------------------------")
    print(f"  GLOBAL (object mean colour)  err {eg:.5f}  vs baseline var {vg:.5f}"
          f"   -> explains {max(0,1-eg/vg):5.1%}")
    print(f"  LOCAL  (deviation from mean) err {el:.5f}  vs baseline var {vl:.5f}"
          f"   -> explains {max(0,1-el/vl):5.1%}")
    print(f"  chroma:  ground truth {np.mean([x['gt_chroma'] for x in R]):.4f}"
          f"   predicted {np.mean([x['pred_chroma'] for x in R]):.4f}"
          f"   ratio {np.mean([x['pred_chroma'] for x in R])/max(1e-9,np.mean([x['gt_chroma'] for x in R])):.2f}")
    print(f"  object-mean chroma: gt {np.mean([x['gt_mu_chroma'] for x in R]):.4f}"
          f"   predicted {np.mean([x['pred_mu_chroma'] for x in R]):.4f}")

json.dump(rows, open("/tmp/claude-1565/-home-mls04-psiml-splat-diffusion/"
                     "7d6a7574-1e1e-4fc2-a9bb-caae2deb57f3/scratchpad/why_grey.json", "w"),
          indent=1)
print("\nsaved why_grey.json")
