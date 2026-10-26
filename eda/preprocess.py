#!/usr/bin/env python
"""
Convert a category of ShapeSplats PLY files into fixed-size normalised shards.

Why bother: Lustre is fast at large sequential reads and slow at thousands of
small random ones. Reading 889 individual PLY files every epoch will leave the
A100 waiting on metadata latency. This does the parse once and writes a few
large .npy files that memory-map instantly.

You need the fixed-N step for the model anyway, so this is not extra work.

    python preprocess.py --root /lustre/data/mls04/ModelNet_Splats/extracted/train \
                         --category chair --out ~/psiml/data --n 32768

Output
    <out>/<category>_train.npy      (M, N, C) float32   the tensor
    <out>/<category>_meta.json      channel names, normalisation constants,
                                    per-object original scale/centre

Channels (default, drop_sh=True):
    0:3   xyz          normalised position
    3:6   log_scale
    6:10  quaternion   canonicalised, w >= 0
    10    logit_opacity
    11:14 rgb
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import splatlib as sl

CHANNELS = ["x", "y", "z", "ls0", "ls1", "ls2", "qw", "qx", "qy", "qz",
            "opacity", "r", "g", "b"]


def importance(sp):
    """alpha * ellipsoid volume -- a cheap proxy for visual contribution.

    Opacity alone is misleading: a large faint Gaussian can matter more than a
    small opaque one.
    """
    return sp.opacity * np.prod(sp.scale, axis=1)


def to_fixed(sp, n_target, normalise_per_object=True):
    """Parse one object into an (n_target, 14) array. Returns (arr, meta)."""
    keep = np.argsort(importance(sp))[::-1][:n_target]

    xyz = sp.xyz[keep]
    log_scale = sp.raw_scale[keep]
    quat = sl.canonicalize_quat(sp.quat[keep])
    logit_op = sp.raw_opacity[keep]
    rgb = sp.rgb[keep]

    centre = 0.5 * (sp.xyz.min(0) + sp.xyz.max(0))
    radius = float(np.abs(sp.xyz - centre).max())
    if normalise_per_object:
        xyz = (xyz - centre) / radius
        log_scale = log_scale - np.log(radius)   # scales live in the same units

    arr = np.concatenate([xyz, log_scale, quat, logit_op[:, None], rgb], 1)

    n = len(arr)
    if n < n_target:                              # pad with invisible Gaussians
        pad = np.zeros((n_target - n, arr.shape[1]), np.float32)
        pad[:, 3:6] = -10.0                       # tiny
        pad[:, 6] = 1.0                           # identity quaternion
        pad[:, 10] = -20.0                        # opacity ~ 0
        arr = np.concatenate([arr, pad], 0)

    meta = dict(name=sp.name, n_original=len(sp), n_kept=int(n),
                centre=centre.tolist(), radius=radius,
                importance_kept=float(importance(sp)[keep].sum()
                                      / max(importance(sp).sum(), 1e-12)))
    return arr.astype(np.float32), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--category", default="chair")
    ap.add_argument("--out", default=str(Path.home() / "psiml" / "data"))
    ap.add_argument("--n", type=int, default=32768, help="fixed gaussians per object")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first K objects")
    ap.add_argument("--no-normalise", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = sl.find_objects(args.root, category=args.category)
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"no '{args.category}' files under {args.root}")
    print(f"{len(files)} objects -> {args.n} gaussians each, {len(CHANNELS)} channels")
    print(f"tensor will be {len(files) * args.n * len(CHANNELS) * 4 / 1e9:.2f} GB")

    data = np.zeros((len(files), args.n, len(CHANNELS)), np.float32)
    metas, t0 = [], time.time()
    for i, f in enumerate(files):
        arr, m = to_fixed(sl.load_ply(f), args.n, not args.no_normalise)
        data[i] = arr
        metas.append(m)
        if (i + 1) % 50 == 0 or i + 1 == len(files):
            el = time.time() - t0
            print(f"  {i+1}/{len(files)}  {el:.0f}s  "
                  f"~{el/(i+1)*(len(files)-i-1):.0f}s left", flush=True)

    # Global per-channel normalisation, computed on real (non-padded) entries.
    real = data[:, :, 10] > -15
    flat = data[real]
    mean = flat.mean(0)
    std = np.maximum(flat.std(0), 1e-6)
    mean[6:10], std[6:10] = 0.0, 1.0        # leave the unit quaternion alone

    npy = out / f"{args.category}_train.npy"
    np.save(npy, data)

    meta = dict(
        category=args.category, n_objects=len(files), n_gaussians=args.n,
        channels=CHANNELS, normalise_per_object=not args.no_normalise,
        channel_mean=mean.tolist(), channel_std=std.tolist(),
        note="apply (x - channel_mean)/channel_std at load time; "
             "quaternion channels are left unscaled",
        objects=metas,
    )
    (out / f"{args.category}_meta.json").write_text(json.dumps(meta, indent=2))

    kept = np.array([m["importance_kept"] for m in metas])
    trunc = sum(m["n_original"] > args.n for m in metas)
    print(f"\nwrote {npy}  ({npy.stat().st_size/1e9:.2f} GB)")
    print(f"     {out / f'{args.category}_meta.json'}")
    print(f"truncated {trunc}/{len(files)} objects; "
          f"importance retained: min {kept.min():.4f}  mean {kept.mean():.4f}")


if __name__ == "__main__":
    main()
