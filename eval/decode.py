"""
eval/decode.py — STAGE_SPEC §11 build-order step 1, before any model code
exists. Implements check 1 and NORMALIZATION.md's check N2.

Exercises the part of NORMALIZATION.md §5's pipeline that doesn't need a
model or augmentation:

    read .ply (splat_utils.read_splat)
      -> normalize_object + canonicalize_quat + build_token
         (preprocessing.full_object_tokens)              RAW, unstandardized
      -> standardize (per-channel stats, stage_config.load_norm_stats)
      -> split into (cond, target) by stage config        <- this file
      -> recombine                                         <- tests this
      -> unstandardize
      -> tokens_to_splat_dict (denormalize to world units)
      -> render_points

Two checks:
  check 1  (STAGE_SPEC §9). Split -> recombine -> denormalize -> render
           must match rendering the original object directly. Split+
           recombine is pure re-indexing (no model in the loop yet), so
           the standardized round trip should be EXACT, not just close —
           any mismatch means a channel-index bug in split/recombine.
  check N2 (NORMALIZATION.md §6). standardize -> unstandardize round trip
           on RGB, max absolute error below ~1e-5.

Run (from repo root): `python eval/decode.py [--stage N] [--n N]`
"""

import sys
from pathlib import Path

# Direct script execution (`python eval/decode.py`) sets sys.path[0] to
# this file's own directory, not repo root, and never triggers
# eval/__init__.py (that only runs on `import eval.*`) — so `import
# preprocessing` etc. would fail without this. Do it before those imports.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse

import matplotlib
matplotlib.use("Agg")  # CLAUDE.md: no display on this node
import matplotlib.pyplot as plt
import numpy as np

from preprocessing import full_object_tokens, standardize, unstandardize, tokens_to_splat_dict
from splat_utils import find_ply_files, read_splat, render_points
from stage_config import StageConfig, load_norm_stats, load_split, split_channels, recombine_channels

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"


# ----------------------------------------------------------------- round trip
def round_trip_object(path: Path, cfg: StageConfig, norm_stats: dict, up_axis: int):
    """One object through the full split -> recombine -> denormalize ->
    render pipeline. Returns (img_orig, img_recon, errs) where errs has
    the two numeric checks."""
    splat = read_splat(path)
    tok_raw, centre, radius = full_object_tokens(splat)  # (N, 14), per-object normalized

    tok_std = standardize(tok_raw, norm_stats)
    cond_std, target_std = split_channels(tok_std, cfg)
    tok_std_recombined = recombine_channels(cond_std, target_std, cfg)

    # check 1's mechanical half: split+recombine is pure re-indexing, must
    # be exact (not just close) — any drift means a channel-index bug.
    split_recombine_max_err = float(np.abs(tok_std_recombined - tok_std).max())

    tok_raw_recon = unstandardize(tok_std_recombined, norm_stats)

    # check N2: standardize -> unstandardize round trip on RGB specifically
    rgb_idx = [11, 12, 13]
    n2_max_err = float(np.abs(tok_raw_recon[:, rgb_idx] - tok_raw[:, rgb_idx]).max())
    full_max_err = float(np.abs(tok_raw_recon - tok_raw).max())

    recon_splat = tokens_to_splat_dict(tok_raw_recon, centre, radius, name=splat["name"] + "_roundtrip")

    img_orig = render_points(splat, azim=40, elev=15, up_axis=up_axis, size=220)
    img_recon = render_points(recon_splat, azim=40, elev=15, up_axis=up_axis, size=220)

    errs = dict(
        split_recombine_max_err=split_recombine_max_err,
        n2_rgb_max_err=n2_max_err,
        full_channel_max_err=full_max_err,
    )
    return img_orig, img_recon, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--n", type=int, default=4, help="number of val objects to check")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = StageConfig(stage=args.stage)
    print(f"stage {cfg.stage}: target_ch={cfg.target_ch}  cond_ch={cfg.cond_ch}")

    norm_stats = load_norm_stats(cfg.shard_path)  # asserts against STAGE_SPEC §2's table
    print("norm_stats.json matches STAGE_SPEC §2's table — PASS")

    split = load_split(cfg.shard_path)
    up_axis = norm_stats["up_axis"]

    files = find_ply_files(cfg.data_dir)
    name_to_file = {f.stem: f for f in files}

    rng = np.random.default_rng(args.seed)
    val_names = split["val"]
    picks = rng.choice(len(val_names), min(args.n, len(val_names)), replace=False)
    picked_names = [val_names[i] for i in picks]

    fig, axes = plt.subplots(len(picked_names), 2, figsize=(6, 3 * len(picked_names)))
    if len(picked_names) == 1:
        axes = axes[None, :]

    all_errs = []
    for row, name in enumerate(picked_names):
        img_orig, img_recon, errs = round_trip_object(name_to_file[name], cfg, norm_stats, up_axis)
        all_errs.append(errs)
        print(f"  {name}: split/recombine max_err={errs['split_recombine_max_err']:.2e}  "
              f"N2 RGB max_err={errs['n2_rgb_max_err']:.2e}  "
              f"full-channel max_err={errs['full_channel_max_err']:.2e}")

        axes[row, 0].imshow(img_orig)
        axes[row, 0].set_title(f"{name[:14]} original", fontsize=8)
        axes[row, 0].axis("off")
        axes[row, 1].imshow(img_recon)
        axes[row, 1].set_title(f"stage {cfg.stage} split->recombine round-trip", fontsize=8)
        axes[row, 1].axis("off")

    plt.tight_layout()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / "decode_roundtrip.png"
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out_path}")

    split_recombine_worst = max(e["split_recombine_max_err"] for e in all_errs)
    n2_worst = max(e["n2_rgb_max_err"] for e in all_errs)

    print(f"\ncheck 1 (split->recombine exact):     worst max_err = {split_recombine_worst:.2e}")
    print(f"check N2 (standardize round trip, RGB): worst max_err = {n2_worst:.2e}  (must be < 1e-5)")

    assert split_recombine_worst < 1e-6, "split->recombine is not exact — channel-index bug"
    assert n2_worst < 1e-5, "standardize/unstandardize round trip on RGB exceeds NORMALIZATION.md §6 tolerance"
    print("\nALL DECODE CHECKS PASSED.")


if __name__ == "__main__":
    main()
