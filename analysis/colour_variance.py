"""Where does the RGB variance in this dataset actually live?

The question behind the colour collapse: if a model predicts each object's
GLOBAL colour perfectly but nothing else, how much MSE does it save? And if
it predicts local detail perfectly but gets the global colour wrong, how
much does it save? Those two numbers decide what an MSE-trained model will
spend its capacity on.

Variance decomposition, per RGB channel, over the train split:

    Var_total = E_i[ Var_within(object i) ]  +  Var_i[ mean(object i) ]
                \_______ local detail ______/   \____ global hue ______/
"""
import sys, time
from pathlib import Path as _Path; sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import numpy as np
from splat_utils import find_ply_files, read_splat
from stage_config import DATA_DIR, SHARD_PATH, load_norm_stats, load_split

stats = load_norm_stats(SHARD_PATH)
mean = np.asarray(stats["mean"])[11:14]
std = np.asarray(stats["std"])[11:14]
print(f"norm stats  rgb mean {mean.round(4)}  std {std.round(4)}")

names = load_split(SHARD_PATH)["train"]
n2f = {f.stem: f for f in find_ply_files(DATA_DIR)}

rows = []
t0 = time.time()
for i, n in enumerate(names):
    s = read_splat(n2f[n])
    rgb = np.clip(s["rgb"], 0, 1).astype(np.float64)
    w = s["opacity"].astype(np.float64)          # opacity-weighted throughout
    w = w / max(w.sum(), 1e-9)
    mu = (w[:, None] * rgb).sum(0)                       # (3,) object mean
    var = (w[:, None] * (rgb - mu) ** 2).sum(0)          # (3,) within-object
    rows.append((mu, var, s["n"]))
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(names)}  {time.time()-t0:.0f}s")

mu = np.stack([r[0] for r in rows])          # (N_obj, 3)
within = np.stack([r[1] for r in rows])      # (N_obj, 3)

print(f"\n{len(rows)} objects\n")
print("RAW rgb units [0,1]")
print(f"  {'chan':>5} {'within (mean)':>14} {'between':>10} {'total':>10} "
      f"{'between/total':>14}")
tot_b = tot_w = 0.0
for c, name in enumerate("rgb"):
    w_ = within[:, c].mean()
    b_ = mu[:, c].var()
    tot_w += w_; tot_b += b_
    print(f"  {name:>5} {w_:14.5f} {b_:10.5f} {w_+b_:10.5f} "
          f"{b_/(w_+b_):13.1%}")
print(f"  {'ALL':>5} {tot_w:14.5f} {tot_b:10.5f} {tot_w+tot_b:10.5f} "
      f"{tot_b/(tot_w+tot_b):13.1%}")

print("\nSTANDARDIZED units (what the loss sees: divide by per-channel std)")
for c, name in enumerate("rgb"):
    w_ = within[:, c].mean() / std[c] ** 2
    b_ = mu[:, c].var() / std[c] ** 2
    print(f"  {name:>5}  within {w_:.4f}   between {b_:.4f}   sum {w_+b_:.4f}")

# The headline: an MSE-optimal predictor that knows the object's global
# colour but nothing else vs one that knows local detail but not the colour.
b_frac = tot_b / (tot_w + tot_b)
print(f"\n=> a predictor that nails GLOBAL colour and nothing else removes "
      f"{b_frac:.1%} of RGB MSE")
print(f"=> a predictor that nails LOCAL detail and outputs the dataset-mean "
      f"hue removes {1-b_frac:.1%}")

# How grey is the dataset, in the units the model is scored in?
chroma_obj = mu.max(1) - mu.min(1)
print(f"\nper-object mean-colour chroma (max-min of the object's mean rgb):")
for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
    print(f"    q{q:<5.2f} {np.quantile(chroma_obj, q):.4f}")
print(f"    frac of objects with chroma < 0.05: "
      f"{(chroma_obj < 0.05).mean():.1%}")
print(f"    frac of objects with chroma < 0.10: "
      f"{(chroma_obj < 0.10).mean():.1%}")

np.savez("/tmp/claude-1565/-home-mls04-psiml-splat-diffusion/"
         "7d6a7574-1e1e-4fc2-a9bb-caae2deb57f3/scratchpad/colour_var.npz",
         mu=mu, within=within, names=np.array(names))
print("\nsaved colour_var.npz")
