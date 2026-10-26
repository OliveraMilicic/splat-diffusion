"""
build_shards.py — Day 1 preprocessing: .ply files -> one memmap of tokens.

DESIGN_SPEC §3.4-3.6, §7.5. Edit the constants below and run:

    python build_shards.py

Skips work that's already on disk (split.json is never silently
regenerated; delete files by hand to force a rebuild — same idiom as
analyze_chairs.py's cached object_stats.csv).

Writes to OUT_DIR:
    split.json              train/val object names, fixed seed
    object_meta.json        per-object {centre, radius} (undoes §3.4)
    norm_stats.json         per-channel mean/std, sigma_data, axes
    tokens.dat               memmap (n_objects, 8, 4096, 14) float32
    tokens_meta.json         object name -> row index
    mirror_axis_check.png
    roundtrip_check.png
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from splat_utils import find_ply_files, read_splat, render_points
from preprocessing import (
    N_CHANNELS, normalize_object, canonicalize_quat, fps_indices_batch,
    build_token, tokens_to_splat_dict, load_or_make_split,
    compute_channel_stats, standardize, detect_mirror_axis,
)

# ---------------------------------------------------------------- edit these
DATA_DIR = "/lustre/data/mls04/ShapeSplatsV1/extracted"
OUT_DIR = "/lustre/data/mls04/ShapeSplatsV1/shards"
UP_AXIS = 2              # confirmed in analyze_chairs.py / up_axis_check.png
N_VAL = 170               # DESIGN_SPEC §2.3: ~170 held out
N_TOKENS = 4096
N_SUBSETS = 8
SEED = 0
N_ROUNDTRIP_SAMPLES = 4
# -----------------------------------------------------------------------------

data_dir = Path(DATA_DIR)
out_dir = Path(OUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)

files = find_ply_files(data_dir)
print(f"found {len(files)} .ply files under {data_dir}")
if not files:
    raise SystemExit(f"nothing found — check DATA_DIR ({data_dir})")

# canonical row order: sorted object name, fixed regardless of split
names = sorted(f.stem for f in files)
name_to_file = {f.stem: f for f in files}
n_objects = len(names)

# --------------------------------------------------------- 1. split (§2.3)
split = load_or_make_split(out_dir / "split.json", files, N_VAL, SEED)
train_set = set(split["train"])

# --------------------------------------------------- 2. mirror axis (§3.6)
mirror_axis_path = out_dir / "mirror_axis.json"
if mirror_axis_path.exists():
    mirror_info = json.loads(mirror_axis_path.read_text())
    print(f"reusing mirror axis {mirror_info['winner']} from {mirror_axis_path}")
else:
    print("detecting mirror axis (reflect + self-nearest-neighbour on a sample)...")
    winner, scores = detect_mirror_axis(files, up_axis=UP_AXIS, seed=SEED)
    mirror_info = dict(winner=winner, scores=scores, up_axis=UP_AXIS)
    mirror_axis_path.write_text(json.dumps(mirror_info, indent=2))
    fig, ax = plt.subplots(figsize=(4, 3.5))
    axes_lbl = [str(a) for a in scores]
    ax.bar(axes_lbl, [scores[int(a)] for a in axes_lbl], color=["#4C72B0", "#C44E52"])
    ax.set(xlabel="candidate axis", ylabel="mean symmetry score (lower = more symmetric)",
           title=f"mirror axis = {winner}")
    plt.tight_layout()
    plt.savefig(out_dir / "mirror_axis_check.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"mirror axis scores: {scores}  -> winner = {winner}")
    print(f"wrote {out_dir / 'mirror_axis_check.png'}")
MIRROR_AXIS = mirror_info["winner"]

# ----------------------------------------------------- 3. per-object pass
tokens_path = out_dir / "tokens.dat"
meta_path = out_dir / "object_meta.json"
tokens_meta_path = out_dir / "tokens_meta.json"

if tokens_path.exists() and meta_path.exists() and tokens_meta_path.exists():
    print(f"reusing existing shards at {tokens_path} (delete to rebuild)")
    object_meta = json.loads(meta_path.read_text())
else:
    shape = (n_objects, N_SUBSETS, N_TOKENS, N_CHANNELS)
    mm = np.memmap(tokens_path, dtype=np.float32, mode="w+", shape=shape)
    object_meta = {}

    t0 = time.time()
    for row, name in enumerate(names):
        s = read_splat(name_to_file[name])
        xyz_n, log_scale_n, centre, radius = normalize_object(s["xyz"], s["log_scale"])
        quat_c = canonicalize_quat(s["quat"])

        rng = np.random.default_rng(SEED + row)
        fps_idx = fps_indices_batch(xyz_n, N_TOKENS, N_SUBSETS, rng)  # (8, 4096)

        for sub in range(N_SUBSETS):
            idx = fps_idx[sub]
            tok = build_token(
                xyz_n[idx], log_scale_n[idx], quat_c[idx],
                s["opacity_logit"][idx], s["rgb"][idx],
            )
            mm[row, sub] = tok

        object_meta[name] = dict(centre=centre.tolist(), radius=radius)
        if (row + 1) % 100 == 0:
            print(f"  {row + 1}/{n_objects}  ({time.time() - t0:.0f}s)")

    mm.flush()
    del mm
    print(f"\n{n_objects} objects -> {tokens_path}  ({time.time() - t0:.0f}s total)")

    meta_path.write_text(json.dumps(object_meta, indent=2))
    tokens_meta_path.write_text(json.dumps(
        dict(shape=list(shape), names=names), indent=2))
    print(f"wrote {meta_path}, {tokens_meta_path}")

shape = tuple(json.loads(tokens_meta_path.read_text())["shape"])
mm = np.memmap(tokens_path, dtype=np.float32, mode="r", shape=shape)

# ------------------------------------------------- 4. per-channel stats (§3.5)
stats_path = out_dir / "norm_stats.json"
if stats_path.exists():
    stats = json.loads(stats_path.read_text())
    print(f"reusing existing norm_stats at {stats_path}")
else:
    train_rows = [i for i, n in enumerate(names) if n in train_set]
    print(f"computing per-channel stats over {len(train_rows)} train objects "
          f"(full train split, not a further subsample)...")
    train_tokens = np.asarray(mm[train_rows])  # (n_train, 8, 4096, 14) — fits in RAM (~1.9GB)
    stats = compute_channel_stats(train_tokens)

    standardized = standardize(train_tokens, stats)
    sigma_data = float(standardized.std())
    stats["sigma_data"] = sigma_data
    stats["up_axis"] = UP_AXIS
    stats["mirror_axis"] = MIRROR_AXIS
    stats["n_train"] = len(train_rows)
    stats["n_val"] = n_objects - len(train_rows)
    stats["seed"] = SEED

    ok = abs(sigma_data - 1.0) < 0.1
    print(f"sigma_data = {sigma_data:.4f}  (assert within 10% of 1.0: {'PASS' if ok else 'FAIL'})")
    if not ok:
        print("  *** sigma_data is NOT within 10% of 1.0 — DESIGN_SPEC §4.3 says use the "
              "measured value regardless, but this is worth a second look before Day 2. ***")

    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"wrote {stats_path}")
    del train_tokens, standardized

# --------------------------------------------------------- 5. round-trip check
print(f"\nround-trip check: normalize -> un-normalize -> render vs. original")
rng = np.random.default_rng(SEED + 999)
train_names = split["train"]
picks = rng.choice(len(train_names), min(N_ROUNDTRIP_SAMPLES, len(train_names)), replace=False)

fig, axes = plt.subplots(N_ROUNDTRIP_SAMPLES, 2, figsize=(6, 3 * N_ROUNDTRIP_SAMPLES))
for row_i, pick in enumerate(picks):
    name = train_names[pick]
    row = names.index(name)
    orig = read_splat(name_to_file[name])
    meta = object_meta[name]

    tok = mm[row, 0]  # first FPS subset, unstandardized (as stored)
    recon = tokens_to_splat_dict(tok, np.asarray(meta["centre"], np.float32),
                                  meta["radius"], name=name)

    img_orig = render_points(orig, azim=40, elev=15, up_axis=UP_AXIS, size=220)
    img_recon = render_points(recon, azim=40, elev=15, up_axis=UP_AXIS, size=220)

    axes[row_i, 0].imshow(img_orig)
    axes[row_i, 0].set_title(f"{name[:14]} original ({orig['n']//1000}k)", fontsize=8)
    axes[row_i, 0].axis("off")
    axes[row_i, 1].imshow(img_recon)
    axes[row_i, 1].set_title(f"round-trip ({N_TOKENS}k subset)", fontsize=8)
    axes[row_i, 1].axis("off")

plt.tight_layout()
plt.savefig(out_dir / "roundtrip_check.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out_dir / 'roundtrip_check.png'}")

print(f"""
SUMMARY
  objects        {n_objects}  (train {len(split['train'])} / val {len(split['val'])})
  shards         {tokens_path}  shape {shape}
  up_axis        {UP_AXIS}
  mirror_axis    {MIRROR_AXIS}
  sigma_data     {stats['sigma_data']:.4f}
""")
