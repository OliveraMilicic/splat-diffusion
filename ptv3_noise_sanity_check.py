"""
ptv3_noise_sanity_check.py — real chairs through PTv3 at random init.

Reuses ptv3.py and splat_utils.render_points as-is (no changes to either).
Not EDM preconditioning (no c_in scaling on the network input, no
c_skip/c_out combination on the output) -- just x_t = x0 + sigma*eps fed
straight into the network, per the pending-Day-2 scoping. Purely a "does
the untrained network produce sane-shaped, finite numbers" diagnostic.

Runs the SAME fixed random-init model (one seed, built once) across
several chairs spanning a range of Gaussian counts, so the single-chair
finding can be checked for consistency rather than trusted as one sample.

Run: python ptv3_noise_sanity_check.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from splat_utils import render_points
from preprocessing import standardize, unstandardize, tokens_to_splat_dict
from ptv3 import PTv3

SHARDS = Path("/lustre/data/mls04/ShapeSplatsV1/shards")
STATS_CSV = Path("/lustre/data/mls04/ShapeSplatsV1/results/object_stats.csv")
OUT_DIR = Path("/lustre/data/mls04/ShapeSplatsV1/results/random_init_samples")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SEED = 0       # fixes the network's random init -- ONE set of weights, reused for every chair
SELECT_SEED = 0      # fixes which chair is picked within each Gaussian-count bin
NOISE_SEED_BASE = 0  # per-chair noise draw = NOISE_SEED_BASE + chair index (reproducible, not shared)
N_CHAIRS = 7
SIGMA = 1.0
UP_AXIS = 2

CHANNEL_NAMES = ["x", "y", "z", "log_sx", "log_sy", "log_sz",
                 "qw", "qx", "qy", "qz", "op_logit", "r", "g", "b"]
GROUPS = {
    "position": (0, 3), "log-scale": (3, 6), "rotation": (6, 10),
    "opacity (logit)": (10, 11), "color": (11, 14),
}
CHAIR_COLORS = plt.cm.tab10.colors

# ------------------------------------------------------------- load shards + pick chairs spanning n_gaussians
tmeta = json.loads((SHARDS / "tokens_meta.json").read_text())
names = tmeta["names"]
shape = tuple(tmeta["shape"])
mm = np.memmap(SHARDS / "tokens.dat", dtype=np.float32, mode="r", shape=shape)
split = json.loads((SHARDS / "split.json").read_text())
norm_stats = json.loads((SHARDS / "norm_stats.json").read_text())
object_meta = json.loads((SHARDS / "object_meta.json").read_text())

train_names = set(split["train"])
stats = pd.read_csv(STATS_CSV)
stats = stats[stats["name"].isin(train_names)].sort_values("n_gaussians").reset_index(drop=True)

# Force-include the true min and max (equal-frequency binning alone can miss the extreme
# tail of a right-skewed distribution -- checked, it did: an earlier version of this
# selection landed at 19,600-41,482, missing the train split's actual 11,100-76,333 range).
# Remaining slots: equal-frequency bins across what's left, one seeded random pick per bin,
# so the middle is still spread rather than clustered near the median.
select_rng = np.random.default_rng(SELECT_SEED)
extremes = stats.iloc[[0, -1]]  # sorted ascending: smallest, largest
middle = stats.iloc[1:-1].reset_index(drop=True)
n_middle = N_CHAIRS - 2
edges = np.linspace(0, len(middle), n_middle + 1).astype(int)

chosen = [(extremes.iloc[0]["name"], int(extremes.iloc[0]["n_gaussians"])),
          (extremes.iloc[1]["name"], int(extremes.iloc[1]["n_gaussians"]))]
for i in range(n_middle):
    bin_rows = middle.iloc[edges[i]:edges[i + 1]]
    pick = bin_rows.iloc[select_rng.integers(len(bin_rows))]
    chosen.append((pick["name"], int(pick["n_gaussians"])))
chosen.sort(key=lambda t: t[1])

print(f"selected {N_CHAIRS} chairs spanning the train split's Gaussian-count range:")
for nm, ng in chosen:
    print(f"  {nm}  ({ng:,} gaussians)")

# ------------------------------------------------------------- build PTv3 ONCE, fixed random init
torch.manual_seed(MODEL_SEED)
model = PTv3(in_ch=14, dim=384, depth=12, heads=6, patch_size=1024, mlp_ratio=4, k=16)
model.eval()
print(f"\nPTv3 built once with MODEL_SEED={MODEL_SEED} -- identical weights reused for every chair below")


def decode(tok_std, meta, label):
    """Un-standardize -> un-normalize (§3.4) -> activate -> splat dict. Same decode path
    the round-trip check used (preprocessing.py::tokens_to_splat_dict)."""
    tok_unstd = unstandardize(tok_std, norm_stats)
    n_nan = int(np.isnan(tok_unstd).sum())
    n_inf = int(np.isinf(tok_unstd).sum())
    safe = np.nan_to_num(tok_unstd, nan=0.0, posinf=1e4, neginf=-1e4)
    splat = tokens_to_splat_dict(safe, np.asarray(meta["centre"], np.float32), meta["radius"], name=label)
    return splat, n_nan, n_inf


# ------------------------------------------------------------- per-chair: noise, forward, decode, render
per_chair = []  # list of dicts: name, n_gaussians, x0_np, xt_np, out_np, splats, nan/inf counts

for i, (name, n_gaussians) in enumerate(chosen):
    row = names.index(name)
    meta = object_meta[name]
    raw_tok = np.asarray(mm[row, 0])  # (4096, 14) stored, per-object-normalized, unstandardized
    x0 = torch.from_numpy(standardize(raw_tok, norm_stats)).float().unsqueeze(0)  # (1,4096,14)

    noise_gen = torch.Generator().manual_seed(NOISE_SEED_BASE + i)
    eps = torch.randn(x0.shape, generator=noise_gen)
    x_t = x0 + SIGMA * eps
    sigma_t = torch.full((1,), SIGMA)

    with torch.no_grad():
        out = model(x_t, sigma_t)  # raw F_theta(x_t, sigma) -- no c_in/c_skip/c_out, same as before

    x0_np, xt_np, out_np = x0[0].numpy(), x_t[0].numpy(), out[0].numpy()

    splat_clean, _, _ = decode(x0_np, meta, "clean")
    splat_noisy, _, _ = decode(xt_np, meta, "noisy")
    splat_output, n_nan, n_inf = decode(out_np, meta, "output")
    if n_nan or n_inf:
        print(f"  [{name}] output decode: {n_nan} NaN, {n_inf} Inf (clipped to +/-1e4 before render)")

    # per-chair PNG
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    titles = [f"clean\n{name[:18]}", f"noisy input (sigma={SIGMA:g})",
              "raw random-init output\n(EXPECTED to look like noise, not a chair)"]
    for ax, splat, title in zip(axes, [splat_clean, splat_noisy, splat_output], titles):
        img = render_points(splat, azim=40, elev=15, up_axis=UP_AXIS, size=300)
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    per_chair_path = OUT_DIR / f"{name}_clean_noisy_output.png"
    plt.savefig(per_chair_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {per_chair_path}")

    per_chair.append(dict(name=name, n_gaussians=n_gaussians, x0_np=x0_np, xt_np=xt_np, out_np=out_np,
                           splats=(splat_clean, splat_noisy, splat_output), n_nan=n_nan, n_inf=n_inf))

# ------------------------------------------------------------- combined grid: rows=chairs, cols=3
fig, axes = plt.subplots(N_CHAIRS, 3, figsize=(11, 3.4 * N_CHAIRS))
for row_i, rec in enumerate(per_chair):
    col_titles = ["clean", f"noisy (sigma={SIGMA:g})", "raw output"]
    for col_i, (splat, ctitle) in enumerate(zip(rec["splats"], col_titles)):
        img = render_points(splat, azim=40, elev=15, up_axis=UP_AXIS, size=260)
        ax = axes[row_i, col_i]
        ax.imshow(img)
        ax.axis("off")
        if row_i == 0:
            ax.set_title(ctitle, fontsize=10)
        if col_i == 0:
            ax.text(-0.05, 0.5, f"{rec['name'][:14]}\n({rec['n_gaussians']:,}g)", fontsize=8,
                     transform=ax.transAxes, ha="right", va="center")
plt.tight_layout()
grid_path = OUT_DIR / "all_chairs_grid.png"
plt.savefig(grid_path, dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nwrote {grid_path}")

# ------------------------------------------------------------- combined per-channel range plot, all chairs overlaid
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes_flat = axes.flat
flags = []

all_clean = np.stack([r["x0_np"] for r in per_chair])   # (N_CHAIRS, 4096, 14)
all_noisy = np.stack([r["xt_np"] for r in per_chair])
all_output = np.stack([r["out_np"] for r in per_chair])

for ax, (gname, (lo, hi)) in zip(axes_flat, GROUPS.items()):
    n_ch = hi - lo
    base = np.arange(n_ch)
    any_flag_in_group = False

    for ci, c in enumerate(range(lo, hi)):
        # pooled clean+noisy reference band, all chairs combined
        ref = np.concatenate([all_clean[:, :, c].ravel(), all_noisy[:, :, c].ravel()])
        ref_lo, ref_hi = ref.min(), ref.max()
        ref_span = max(ref_hi - ref_lo, 1e-6)
        ax.bar(ci, ref_hi - ref_lo, bottom=ref_lo, width=0.7, color="lightgrey", alpha=0.6, zorder=0)

        # per-chair output range bar (p5-p95, robust to single-Gaussian tails) + median marker
        chair_medians = []
        n_nan_total, n_inf_total = 0, 0
        for i, rec in enumerate(per_chair):
            col = all_output[i, :, c]
            n_nan_total += int(np.isnan(col).sum())
            n_inf_total += int(np.isinf(col).sum())
            finite = col[np.isfinite(col)]
            if finite.size == 0:
                continue
            p5, p50, p95 = np.percentile(finite, [5, 50, 95])
            chair_medians.append(p50)
            x = ci + (i - (N_CHAIRS - 1) / 2) * (0.6 / N_CHAIRS)
            ax.plot([x, x], [p5, p95], color=CHAIR_COLORS[i % 10], lw=1.5, alpha=0.85, zorder=2)
            ax.plot(x, p50, "o", color=CHAIR_COLORS[i % 10], ms=3, zorder=3)

        # numeric consistency check: chair-to-chair spread of the median vs. a typical chair's own spread
        chair_medians = np.asarray(chair_medians)
        chair_intra_stds = [np.std(all_output[i, :, c][np.isfinite(all_output[i, :, c])])
                             for i in range(N_CHAIRS)]
        inter_spread = chair_medians.std() if len(chair_medians) > 1 else 0.0
        typical_intra = np.mean(chair_intra_stds) if chair_intra_stds else 1e-6
        consistency_ratio = inter_spread / max(typical_intra, 1e-6)

        wildly_out = any(
            (all_output[i, :, c][np.isfinite(all_output[i, :, c])].max() > ref_hi + 5 * ref_span
             or all_output[i, :, c][np.isfinite(all_output[i, :, c])].min() < ref_lo - 5 * ref_span)
            for i in range(N_CHAIRS) if np.isfinite(all_output[i, :, c]).any()
        )
        inconsistent = consistency_ratio > 1.0

        if n_nan_total or n_inf_total or wildly_out or inconsistent:
            any_flag_in_group = True
            flags.append(dict(channel=CHANNEL_NAMES[c], n_nan=n_nan_total, n_inf=n_inf_total,
                               wildly_out=wildly_out, consistency_ratio=float(consistency_ratio),
                               inconsistent=inconsistent))

    ax.set_xticks(base)
    ax.set_xticklabels([CHANNEL_NAMES[c] for c in range(lo, hi)])
    title_color = "crimson" if any_flag_in_group else "black"
    suffix = "  *** FLAGGED ***" if any_flag_in_group else ""
    ax.set_title(gname + suffix, color=title_color, fontsize=10)
    ax.axhline(0, color="grey", lw=0.5, ls=":")

axes_flat[-1].axis("off")
handles = [plt.Line2D([0], [0], color=CHAIR_COLORS[i % 10], marker="o", ms=4,
                       label=f"{per_chair[i]['name'][:12]} ({per_chair[i]['n_gaussians']:,}g)")
           for i in range(N_CHAIRS)]
handles.append(plt.Rectangle((0, 0), 1, 1, fc="lightgrey", alpha=0.6, label="clean+noisy ref. range (all chairs)"))
axes_flat[-1].legend(handles=handles, loc="center", fontsize=8,
                      title="per-chair raw output (p5-p95 + median)")

plt.tight_layout()
range_path = OUT_DIR / "all_chairs_channel_ranges.png"
plt.savefig(range_path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"wrote {range_path}")

# ------------------------------------------------------------- verdict
print("\n" + "=" * 78)
print(f"Chairs used ({N_CHAIRS}, spanning the train split's Gaussian-count range):")
for rec in per_chair:
    print(f"  {rec['name']:<45} {rec['n_gaussians']:>7,} gaussians"
          + (f"   [{rec['n_nan']} NaN, {rec['n_inf']} Inf on decode]" if rec["n_nan"] or rec["n_inf"] else ""))

if flags:
    print(f"\nFLAGGED: {len(flags)} channel(s) do NOT look consistently like plain untrained-random "
          f"across all {N_CHAIRS} chairs:")
    for f in flags:
        reasons = []
        if f["n_nan"] or f["n_inf"]:
            reasons.append(f"{f['n_nan']} NaN / {f['n_inf']} Inf")
        if f["wildly_out"]:
            reasons.append("output range >5x the pooled clean+noisy range for at least one chair")
        if f["inconsistent"]:
            reasons.append(f"chair-to-chair median spread ({f['consistency_ratio']:.2f}x a typical "
                            f"chair's own internal spread) exceeds the within-chair spread")
        print(f"  {f['channel']:>10}: " + "; ".join(reasons))
    print("\n=> The single-chair finding does NOT hold uniformly -- see flagged channels above.")
else:
    print(f"\nNO channels flagged across any of the {N_CHAIRS} chairs: every channel stays finite, "
          f"within a reasonable multiple of the pooled clean/noisy range, AND consistent "
          f"chair-to-chair (no object shows a systematically different output distribution "
          f"than the others for any channel).")
    print("=> The single-chair finding HOLDS across all chairs tested -- untrained-random, not broken, "
          "and not a coincidence of the one chair originally picked.")
print("=" * 78)
