"""
analyze_chairs.py — dataset stats for the ShapeSplatsV1 chairs.

No install, no config file, no CLI framework. Edit the two paths below and run:

    python analyze_chairs.py

Needs: numpy, plyfile, pandas, matplotlib
    pip install numpy plyfile pandas matplotlib

Writes to OUT_DIR:
    object_stats.csv       one row per object
    figures/*.png          a handful of summary plots
    render_grid.png        a quick look at 12 random chairs
"""

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display on this node
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from splat_utils import find_ply_files, object_stats, read_splat, render_points

# ---------------------------------------------------------------- edit these
DATA_DIR = "/lustre/data/mls04/ShapeSplatsV1/extracted"   # where the .ply files live
OUT_DIR = "/lustre/data/mls04/ShapeSplatsV1/results"       # where to write results
UP_AXIS = 2            # 0=x, 1=y, 2=z — confirmed via up_axis_check.png on 3 objects
N_RENDER_SAMPLE = 12    # how many chairs in the quick-look grid
# -----------------------------------------------------------------------------

data_dir = Path(DATA_DIR)
out_dir = Path(OUT_DIR)
fig_dir = out_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------- 1. find files
files = find_ply_files(data_dir)
print(f"found {len(files)} .ply files under {data_dir}")
if not files:
    raise SystemExit(f"nothing found — check DATA_DIR ({data_dir})")

# ------------------------------------------------------- 2. per-object stats
csv_path = out_dir / "object_stats.csv"
if csv_path.exists():
    df = pd.read_csv(csv_path)
    print(f"reusing cached stats for {len(df)} objects (delete {csv_path} to rebuild)")
else:
    rows, bad, t0 = [], [], time.time()
    for i, f in enumerate(files):
        try:
            rows.append(object_stats(read_splat(f)))
        except Exception as e:
            bad.append((f.name, str(e)))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(files)}  ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\n{len(df)} objects in {time.time() - t0:.0f}s -> {csv_path}")
    if bad:
        print(f"{len(bad)} failed to load:")
        for name, err in bad[:10]:
            print(f"   {name}: {err}")

print(df[["n_gaussians", "radius", "opacity_mean", "aniso_median"]].describe())

# ---------------------------------------------------------------- 3. figures
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))

axes[0, 0].hist(df["n_gaussians"], bins=50, color="#4C72B0")
axes[0, 0].axvline(32768, color="crimson", ls="--", lw=1, label="32768 = 32³")
axes[0, 0].set(xlabel="gaussians per object", ylabel="objects", title="count")
axes[0, 0].legend(fontsize=7)

axes[0, 1].scatter(df["center_x"], df["center_y"], s=5, alpha=0.4)
axes[0, 1].axhline(0, color="k", lw=0.5)
axes[0, 1].axvline(0, color="k", lw=0.5)
axes[0, 1].set(xlabel="bbox centre x", ylabel="bbox centre y", title="centred?")

axes[0, 2].hist(df["radius"], bins=50, color="#55A868")
axes[0, 2].set(xlabel="radius", ylabel="objects", title="object size")

axes[1, 0].hist(df["log_scale_mean"], bins=50, color="#8172B2")
axes[1, 0].set(xlabel="mean log(scale)", title="scale (log space)")

axes[1, 1].hist(df["opacity_mean"], bins=50, color="#C44E52")
axes[1, 1].set(xlabel="mean opacity", title="opacity")

axes[1, 2].hist(df["frac_quat_w_negative"], bins=40, color="#DD8452")
axes[1, 2].axvline(0.5, color="k", ls="--", lw=1)
axes[1, 2].set(xlabel="fraction with quat w < 0", title="0.5 = not canonicalised")

plt.tight_layout()
plt.savefig(fig_dir / "overview.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"wrote {fig_dir / 'overview.png'}")

# -------------------------------------------------------- 4. quick-look grid
rng = np.random.default_rng(0)
picks = sorted(rng.choice(len(files), min(N_RENDER_SAMPLE, len(files)), replace=False))

fig, axes = plt.subplots(3, 4, figsize=(13, 10))
for ax, i in zip(axes.ravel(), picks):
    s = read_splat(files[i])
    ax.imshow(render_points(s, azim=40, elev=15, up_axis=UP_AXIS, size=260))
    ax.set_title(f"{s['name']} ({s['n'] // 1000}k)", fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig(out_dir / "render_grid.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out_dir / 'render_grid.png'}")

# ---------------------------------------------------------------- 5. summary
print(f"""
DATASET      {len(df)} objects

COUNT        median {df['n_gaussians'].median():,.0f}   range [{df['n_gaussians'].min():,} .. {df['n_gaussians'].max():,}]
GEOMETRY     radius mean {df['radius'].mean():.4f}   CV {df['radius'].std()/df['radius'].mean():.3f}
             bbox centre, max |offset| {df[['center_x','center_y','center_z']].abs().to_numpy().max():.4f}
SCALE        log-scale mean {df['log_scale_mean'].mean():.3f}
OPACITY      mean {df['opacity_mean'].mean():.3f}
ROTATION     fraction w<0: {df['frac_quat_w_negative'].mean():.3f}  (0.5 = not canonicalised)

Look at render_grid.png first — if the chairs are sideways, change UP_AXIS
at the top of this script and rerun (stats are cached, only the render redoes).
""")
