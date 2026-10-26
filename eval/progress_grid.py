"""
eval/progress_grid.py — the training-progress figure: the SAME object,
sampled from the model at successive points in training, side by side.

This is the figure that shows a diffusion model working. A loss curve says
the number went down; this says what the number bought. It is also the one
figure that cannot be rebuilt after the fact unless the run kept its
intermediate weights, which is why `train/loop.py` retains a step-named EMA
snapshot at every render boundary.

    python eval/progress_grid.py stage3_full
    python eval/progress_grid.py stage3_full --n-objects 3 --n-cols 6

Columns are `INPUT | step a | step b | ... | GROUND TRUTH`, one row per
object. INPUT is the conditioning with every target channel at the dataset
mean — the mean-fill baseline, i.e. what the model starts from — so the
progression reads left to right from "no appearance information" to
"ground truth", with the model's actual trajectory in between.

Rendered through `eval/render3d.py`, so at stage 3 the changing splat SIZE
and TRANSLUCENCY are visible, not just colour.
"""

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import container_env  # noqa: F401 -- MUST precede torch

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from diffusion.sampling import sample as diffusion_sample
from eval.load_run import load_run, train_names_of
from eval.render3d import gsplat_available
from eval.render_grid import (geometry_only_tokens, mean_fill_target,
                               render_tokens, to_tokens)
from stage_config import load_norm_stats
from train.dataset import ChairPointDataset

REPORT_DIR = Path(_REPO_ROOT) / "reports"


def find_snapshots(run_dir: Path, n_cols: int) -> list:
    """Step-named EMA snapshots, thinned to `n_cols` evenly spaced ones with
    the LAST always included — the final model is the point of the figure and
    must never be the one that gets dropped by the thinning."""
    snaps = sorted(run_dir.glob("snapshots/ema_step*.pt"))
    if not snaps:
        # Fall back to whatever single snapshot the run did leave behind, so
        # this still produces something on a run that predates snapshot
        # retention -- with one column instead of a progression.
        snaps = sorted(run_dir.glob("snapshot_step*.pt")) or \
            sorted(run_dir.glob("snapshot_latest.pt"))
    if not snaps:
        raise SystemExit(
            f"no snapshots under {run_dir}. A run started with "
            f"--no-keep-snapshots cannot produce this figure; nothing recovers "
            f"weights that were never written.")
    if len(snaps) <= n_cols:
        return snaps
    idx = np.linspace(0, len(snaps) - 1, n_cols).round().astype(int)
    idx[-1] = len(snaps) - 1
    return [snaps[i] for i in sorted(set(idx.tolist()))]


def step_of(path: Path) -> int:
    digits = "".join(c for c in path.stem if c.isdigit())
    return int(digits) if digits else 0


@torch.no_grad()
def build(run_name: str, n_objects: int = 2, n_cols: int = 6,
          sample_steps: int = None, seed: int = 0, image_size: int = 256,
          split: str = "test", device=None, out: Path = None,
          names: list = None) -> Path:
    """`names` restricts a `split="train"` figure to specific objects.
    Defaults to the run's OWN `train_subset.json` — without that, a figure
    built for a run trained on 8 named objects would render indices 0..k of
    the full 1,103-object split, i.e. objects that run never saw, and label
    them TRAIN."""
    run_dir = Path("runs") / run_name
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if names is None and split == "train":
        names = train_names_of(run_dir)
        if names:
            print(f"restricting to the run's own {len(names)} train objects")

    snaps = find_snapshots(run_dir, n_cols)
    print(f"{len(snaps)} snapshots: steps "
          f"{[step_of(s) for s in snaps]}")

    # Architecture and config from the LAST snapshot's own payload rather than
    # from config.json (eval/load_run.py) — every snapshot in a run shares the
    # architecture, so one load configures the model for all of them, and a run
    # older than a StageConfig default no longer builds the wrong network.
    model, cfg, _meta = load_run(run_name, device, snapshot=str(snaps[-1]))
    norm_stats = load_norm_stats(cfg.shard_path)
    sample_steps = sample_steps or cfg.sample_steps

    ds = ChairPointDataset(cfg, split=split, augment=False,
                           names=names if split == "train" else None)
    items = [ds[i] for i in range(min(n_objects, len(ds)))]
    # One row per object; columns are INPUT, then one per snapshot, then GT.
    grid = [[None] * (len(snaps) + 2) for _ in items]
    titles = [[""] * (len(snaps) + 2) for _ in items]

    for r, item in enumerate(items):
        cond_np = item["cond"].numpy()
        grid[r][0] = geometry_only_tokens(cond_np, cfg, norm_stats)
        titles[r][0] = "INPUT\n(geometry + mean appearance)"
        grid[r][-1] = to_tokens(cond_np, item["target"], cfg, norm_stats)
        titles[r][-1] = "GROUND TRUTH"

    t0 = time.time()
    for c, snap in enumerate(snaps, start=1):
        state = torch.load(snap, map_location=device, weights_only=False)
        key = next(k for k in ("ema_state", "ema", "model_state") if k in state)
        model.load_state_dict(state[key])
        step = step_of(snap)
        for r, item in enumerate(items):
            cond = item["cond"].to(device)
            offset = torch.tensor([item["n"]], dtype=torch.long, device=device)
            # SAME seed at every step and every object: the column-to-column
            # difference is then the model changing, not the noise draw. With a
            # fresh seed per column an unchanged model would still look like it
            # was doing something.
            pred = diffusion_sample(model, cond, offset, T=cfg.T,
                                    num_steps=sample_steps, seed=seed, device=device)
            mse = float(torch.mean((pred - item["target"].to(device)) ** 2))
            grid[r][c] = to_tokens(cond.cpu().numpy(), pred, cfg, norm_stats)
            titles[r][c] = f"step {step:,}\nMSE={mse:.4f}"
        print(f"  step {step:>7,}  ({time.time() - t0:.0f}s elapsed)")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    n_rows, n_cols_total = len(items), len(snaps) + 2
    fig, axes = plt.subplots(n_rows, n_cols_total,
                             figsize=(2.7 * n_cols_total, 3.1 * n_rows))
    axes = np.atleast_2d(axes)
    for r in range(n_rows):
        for c in range(n_cols_total):
            ax = axes[r, c]
            ax.imshow(render_tokens(grid[r][c], size=image_size, device=device))
            if r == 0:
                ax.set_title(titles[r][c], fontsize=8)
            elif c in (0, n_cols_total - 1):
                pass
            else:
                ax.set_title(titles[r][c].split("\n")[-1], fontsize=7)
            ax.axis("off")
        axes[r, 0].set_ylabel(items[r]["name"][:18], fontsize=7)

    backend = "gsplat" if gsplat_available() else "POINT-CLOUD fallback"
    where = ("HELD-OUT objects" if split == "test"
             else "objects the model was TRAINED ON")
    plt.suptitle(
        f"{run_name} — training progress on {where}   "
        f"stage {cfg.stage}: {', '.join(cfg.target_names)}\n"
        f"same object, same sampling seed, EMA weights at each step   "
        f"({sample_steps} DPM-Solver steps, renderer: {backend})",
        fontsize=11, y=1.0)
    plt.tight_layout()
    out = Path(out) if out else REPORT_DIR / f"progress_{run_name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    ap.add_argument("--n-objects", type=int, default=2)
    ap.add_argument("--n-cols", type=int, default=6,
                    help="how many training steps to show (snapshots are thinned "
                         "evenly; the final one is always kept)")
    ap.add_argument("--sample-steps", type=int, default=None)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = torch.device(args.device) if args.device else None
    path = build(args.run_name, n_objects=args.n_objects, n_cols=args.n_cols,
                 sample_steps=args.sample_steps, seed=args.seed,
                 image_size=args.image_size, split=args.split, device=device,
                 out=args.out)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
