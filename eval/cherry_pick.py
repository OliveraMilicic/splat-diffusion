"""
eval/cherry_pick.py — pick the best-looking objects from a finished run
and render them.

The mid-training grids and `test_eval.py` both render a FIXED set of
indices (0..k), on purpose: a figure whose objects change between
checkpoints can't be compared across a run. That makes them honest but
not flattering — index 0 is whichever object sorted first, not a good
one.

This script is the other thing: a deliberate, *labelled* cherry-pick for
a writeup figure. It scores every object in the run's train subset, then
selects. Two criteria, because "small loss" alone tends to return white
and grey chairs — an almost-colourless object is easy to get right and
looks like nothing:

    sample MSE   full DPM-Solver sampling from pure noise vs. ground
                 truth. The real criterion; the denoising loss is not,
                 since a low L_simple at small t says little about what
                 sampling from noise produces.
    chroma       opacity-weighted mean of (max(rgb) - min(rgb)) over the
                 GROUND-TRUTH colours. Zero for a grey/white chair, high
                 for a saturated one. Weighted by opacity so nearly
                 transparent Gaussians don't vote.

Default selection: keep the best `--mse-quantile` fraction by MSE, then
take the most colourful of those. So the picks are genuinely good
reconstructions, not just colourful ones.

Scoring samples all 100 objects, which costs a couple of minutes, so the
scores are cached to `cherry_pick_scores.json`. `--reuse-scores` re-picks
and re-renders from the cache without sampling everything again.

    python eval/cherry_pick.py stage1_100obj
    python eval/cherry_pick.py stage1_100obj --n-objects 8 --reuse-scores
    python eval/cherry_pick.py stage1_100obj --sort mse   # ignore colour
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
torch.multiprocessing.set_sharing_strategy('file_system')  # CLAUDE.md: 64MB /dev/shm

import numpy as np
import torch

from diffusion.sampling import sample as diffusion_sample
from eval.load_run import load_run
from eval.render_grid import geometry_only_tokens, render_rows, to_splat, to_tokens
from stage_config import load_norm_stats
from train.dataset import ChairPointDataset


def chroma_of(splat: dict) -> float:
    """Opacity-weighted mean saturation of a splat's colours, in [0, 1].

    max(rgb) - min(rgb) per Gaussian: 0 for any grey (r == g == b), 1 for
    a fully saturated primary. Opacity-weighted because a chair's visible
    colour comes from its opaque Gaussians -- the Day-1 EDA found opacity
    varying from 0.21 to 0.98 within a single object, so an unweighted
    mean is diluted by Gaussians nobody sees.
    """
    rgb = np.clip(splat["rgb"], 0.0, 1.0)
    w = splat["opacity"].astype(np.float64)
    c = rgb.max(axis=1) - rgb.min(axis=1)
    return float((w * c).sum() / max(w.sum(), 1e-9))


@torch.no_grad()
def score_objects(model, ds, cfg, norm_stats, device, seed: int = 0) -> list:
    """Sample every object once and score it. Returns a list of dicts."""
    scores = []
    t0 = time.time()
    for i in range(len(ds)):
        item = ds[i]
        cond = item["cond"].to(device)
        target = item["target"].to(device)
        offset = torch.tensor([item["n"]], dtype=torch.long, device=device)
        s0 = diffusion_sample(model, cond, offset, T=cfg.T,
                              num_steps=cfg.sample_steps, seed=seed, device=device)
        mse = float(torch.mean((s0 - target) ** 2))

        cond_np = cond.cpu().numpy()
        gt_splat = to_splat(cond_np, target, cfg, norm_stats,
                            item["centre"], item["radius"], item["name"])
        scores.append(dict(index=i, name=item["name"], n=int(item["n"]),
                           mse=mse, chroma=chroma_of(gt_splat)))
        if (i + 1) % 10 == 0 or i + 1 == len(ds):
            el = time.time() - t0
            print(f"  scored {i + 1:3d}/{len(ds)}  ({el:.0f}s, "
                  f"ETA {el / (i + 1) * (len(ds) - i - 1):.0f}s)")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return scores


def select(scores: list, k: int, mse_quantile: float, mode: str) -> list:
    """Pick k objects. `mode`:
        both  — best `mse_quantile` fraction by MSE, then most colourful
        mse   — lowest MSE, colour ignored
        chroma— most colourful, MSE ignored (a sanity contrast, not a figure)
    """
    if mode == "mse":
        return sorted(scores, key=lambda s: s["mse"])[:k]
    if mode == "chroma":
        return sorted(scores, key=lambda s: -s["chroma"])[:k]

    by_mse = sorted(scores, key=lambda s: s["mse"])
    n_keep = max(k, int(round(len(by_mse) * mse_quantile)))
    pool = by_mse[:n_keep]
    cutoff = pool[-1]["mse"]
    print(f"  MSE cutoff at quantile {mse_quantile:.2f}: {cutoff:.4f} "
          f"({len(pool)} of {len(scores)} objects eligible)")
    return sorted(pool, key=lambda s: -s["chroma"])[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    ap.add_argument("--snapshot", default=None,
                    help="default: the newest snapshot_*.pt in the run dir")
    ap.add_argument("--n-objects", type=int, default=6, help="rows in the figure")
    ap.add_argument("--mse-quantile", type=float, default=0.4,
                    help="fraction of objects (best by MSE) eligible to be picked")
    ap.add_argument("--sort", choices=["both", "mse", "chroma"], default="both")
    ap.add_argument("--reuse-scores", action="store_true",
                    help="re-pick from cherry_pick_scores.json instead of resampling")
    ap.add_argument("--split", default="train", choices=["train", "test"],
                    help="'train' uses the run's own subset -- these are objects the "
                         "model was fit on, and the figure MUST say so")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    run_dir = Path("runs") / args.run_name
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    # Config comes from the CHECKPOINT, not config.json: runs predating
    # `learn_sigma` would otherwise be rebuilt with a double-width output head
    # (see eval/load_run.py).
    model, cfg, meta = load_run(args.run_name, device, snapshot=args.snapshot)
    snap = meta["snapshot"]
    norm_stats = load_norm_stats(cfg.shard_path)

    subset_path = run_dir / "train_subset.json"
    train_names = (json.loads(subset_path.read_text())["names"]
                   if (subset_path.exists() and args.split == "train") else None)
    ds = ChairPointDataset(cfg, split=args.split, augment=False, names=train_names)
    print(f"{args.split} objects: {len(ds)}")

    scores_path = run_dir / f"cherry_pick_scores_{args.split}.json"
    if args.reuse_scores:
        if not scores_path.exists():
            raise SystemExit(f"--reuse-scores given but {scores_path} does not exist")
        scores = json.loads(scores_path.read_text())["scores"]
        print(f"reusing {len(scores)} cached scores from {scores_path.name}")
    else:
        print(f"scoring all {len(ds)} objects (sampling {cfg.sample_steps} steps each)...")
        scores = score_objects(model, ds, cfg, norm_stats, device)
        scores_path.write_text(json.dumps(
            dict(snapshot=str(snap), split=args.split, sample_steps=cfg.sample_steps,
                 scores=scores), indent=2))
        print(f"wrote {scores_path}")

    mse = np.array([s["mse"] for s in scores])
    chroma = np.array([s["chroma"] for s in scores])
    print(f"\nsample MSE : min {mse.min():.4f}  median {np.median(mse):.4f}  "
          f"max {mse.max():.4f}")
    print(f"chroma     : min {chroma.min():.4f}  median {np.median(chroma):.4f}  "
          f"max {chroma.max():.4f}")
    # Worth knowing before trusting the picks: if these are strongly
    # anti-correlated, "low MSE" really is selecting for colourless objects
    # and the two criteria are genuinely in tension.
    if len(scores) > 2:
        print(f"corr(mse, chroma) = {np.corrcoef(mse, chroma)[0, 1]:+.3f}  "
              f"(positive => colourful objects are HARDER)")

    picks = select(scores, args.n_objects, args.mse_quantile, args.sort)
    print(f"\npicked {len(picks)} objects (--sort {args.sort}):")
    print(f"  {'name':<26} {'MSE':>8} {'chroma':>8} {'N':>7}")
    for s in picks:
        print(f"  {s['name'][:26]:<26} {s['mse']:8.4f} {s['chroma']:8.4f} {s['n']:7d}")

    print("\nrendering...")
    rows = []
    for s in picks:
        item = ds[s["index"]]
        cond = item["cond"].to(device)
        target = item["target"].to(device)
        offset = torch.tensor([item["n"]], dtype=torch.long, device=device)
        with torch.no_grad():
            s0 = diffusion_sample(model, cond, offset, T=cfg.T,
                                  num_steps=cfg.sample_steps, seed=0, device=device)
            s1 = diffusion_sample(model, cond, offset, T=cfg.T,
                                  num_steps=cfg.sample_steps, seed=1, device=device)
        cond_np = cond.cpu().numpy()
        rows.append(dict(
            name=item["name"], tag=args.split.upper(), n=int(item["n"]),
            mse=float(torch.mean((s0 - target) ** 2)),
            div=float((s0 - s1).abs().mean()),
            # `render_rows` takes normalized-frame TOKENS now, not world-space
            # splat dicts: it renders through eval/render3d.py's fixed camera
            # rig, which only frames every object identically because every
            # object is in that shared normalized frame.
            tok_input=geometry_only_tokens(cond_np, cfg, norm_stats),
            tok_gt=to_tokens(cond_np, target, cfg, norm_stats),
            tok_s0=to_tokens(cond_np, s0, cfg, norm_stats),
            tok_s1=to_tokens(cond_np, s1, cfg, norm_stats),
        ))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = Path(args.out) if args.out else (
        Path("reports") / f"cherry_pick_{args.run_name}_{args.split}.png")
    # The title states the selection rule and the split. A cherry-picked
    # figure that doesn't say it is cherry-picked, on objects the model was
    # trained on, is the easiest way to mislead a reader (including us).
    rule = {"both": f"lowest-MSE {args.mse_quantile:.0%}, then most colourful",
            "mse": "lowest sample MSE", "chroma": "most colourful"}[args.sort]
    path = render_rows(
        rows, out_path=out, up_axis=norm_stats["up_axis"],
        title=(f"{args.run_name} — CHERRY-PICKED, {args.split.upper()} objects "
               f"({'SEEN in training' if args.split == 'train' else 'held out'})\n"
               f"selected from {len(scores)} by: {rule}   "
               f"(sampled with EMA weights, {cfg.sample_steps} steps)"))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
