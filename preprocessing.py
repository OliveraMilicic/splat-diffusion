"""
preprocessing.py — turns loaded splats into normalized 14-channel tokens.

Plain functions, no classes, no config files — same idiom as splat_utils.py.
Implements DESIGN_SPEC §3.4 (per-object normalization), §3.2 (14-channel
token layout), §3.5 (per-channel standardization), and the FPS core of §3.6.
`build_shards.py` orchestrates these over the full dataset; `dataset.py`
uses `standardize`/`unstandardize` and the augmentation transforms at train
time.

Channel layout (14), DESIGN_SPEC §3.2:
    0:3   x, y, z             (normalized position)
    3:6   log sx, sy, sz      (normalized log-scale)
    6:10  qw, qx, qy, qz      (canonicalized, w >= 0)
    10    logit opacity
    11:14 r, g, b             (decoded SH-DC color, [0,1])
"""

import json
from pathlib import Path

import numpy as np

N_CHANNELS = 14
QUAT_SLICE = slice(6, 10)  # excluded from per-channel standardization


# --------------------------------------------------------------- §3.4
def normalize_object(xyz, log_scale):
    """Centre + unit-sphere scale position; shift log_scale to match.

    The log_scale shift is not optional (DESIGN_SPEC §3.4): scales are in
    world units, so rescaling position without rescaling scale silently
    changes the shape of every Gaussian.

    Returns (xyz_n, log_scale_n, centre, radius) — centre/radius saved
    separately so the normalization is invertible.
    """
    centre = 0.5 * (xyz.min(0) + xyz.max(0))
    radius = float(np.abs(xyz - centre).max())
    xyz_n = (xyz - centre) / radius
    log_scale_n = log_scale - np.log(radius).astype(np.float32)
    return xyz_n.astype(np.float32), log_scale_n.astype(np.float32), centre.astype(np.float32), radius


def canonicalize_quat(quat):
    """Flip sign where w < 0 so w >= 0 (DESIGN_SPEC §3.2). q and -q are the
    same rotation; this fixes the two-to-one mapping (not the discontinuity
    at w=0 — see DESIGN_SPEC §3.3, that's the log-Cholesky ablation)."""
    sign = np.where(quat[:, 0] < 0, -1.0, 1.0).astype(np.float32)
    return quat * sign[:, None]


# --------------------------------------------------------------- STAGE_SPEC §2/§3
def full_object_tokens(splat):
    """`splat_utils.read_splat()` output -> full-resolution (n_gaussians, 14)
    RAW (per-object normalized, unstandardized) token array, plus
    (centre, radius) so it round-trips back to world units.

    No subsampling here — STAGE_SPEC §3 samples N=min(16384, n) at train
    time via random choice, not FPS (unlike `build_shards.py`'s 8
    precomputed 4096-point FPS subsets, which are the wrong shape for this:
    "do not re-run FPS preprocessing at 16,384, it isn't worth the hours").
    This is the same per-object pipeline `build_shards.py` runs before its
    FPS step, just without that step, and reading directly from the .ply
    (`splat_utils.read_splat`) instead of a shard so it works before any
    shard exists.
    """
    xyz_n, log_scale_n, centre, radius = normalize_object(splat["xyz"], splat["log_scale"])
    quat_c = canonicalize_quat(splat["quat"])
    tok = build_token(xyz_n, log_scale_n, quat_c, splat["opacity_logit"], splat["rgb"])
    return tok, centre, radius


# --------------------------------------------------------------- FPS (§3.6)
def fps_indices_batch(points, k, n_starts, rng):
    """Farthest-point sampling, `n_starts` independent random-start runs
    batched together (each row of the batch is one full FPS run).

    Distances use the |a-b|^2 = |a|^2 + |b|^2 - 2*a.b expansion so the
    per-iteration update is one (n_starts,3) @ (3,N) BLAS matmul instead
    of an (n_starts,N,3) broadcast+reduce — ~23x faster in practice
    (benchmarked: 0.73s vs 16.8s per ~22k-point object). FPS is inherently
    sequential (each pick depends on the previous one) so the K=4096
    iterations can't be removed, only the per-iteration cost.

    Returns int64 array (n_starts, k) of indices into `points`.
    """
    points = points.astype(np.float32, copy=False)
    N = points.shape[0]
    points_sq = np.sum(points * points, axis=1)  # (N,)

    starts = rng.integers(0, N, size=n_starts)
    idx_out = np.empty((n_starts, k), dtype=np.int64)
    idx_out[:, 0] = starts

    min_dist = np.full((n_starts, N), np.inf, dtype=np.float32)
    last = points[starts].copy()  # (B, 3)

    for i in range(1, k):
        last_sq = np.sum(last * last, axis=1)          # (B,)
        dot = last @ points.T                            # (B, N)
        d = points_sq[None, :] + last_sq[:, None] - 2.0 * dot
        np.minimum(min_dist, d, out=min_dist)
        nxt = np.argmax(min_dist, axis=1)                 # (B,)
        idx_out[:, i] = nxt
        last = points[nxt]

    return idx_out


# --------------------------------------------------------------- tokens (§3.2)
def build_token(xyz, log_scale, quat, opacity_logit, rgb):
    """Assemble the 14-channel token from already-normalized/canonicalized
    per-Gaussian arrays. All inputs (N, ...) for the same N Gaussians."""
    return np.concatenate(
        [xyz, log_scale, quat, opacity_logit[:, None], rgb], axis=-1
    ).astype(np.float32)


def tokens_to_splat_dict(tokens, centre, radius, name="reconstructed"):
    """Inverse of the whole pipeline: standardized-or-not-doesn't-matter-here
    14-channel tokens (already UNstandardized — caller applies
    `unstandardize` first if needed) -> a dict compatible with
    `splat_utils.render_points` (world-space xyz, activated scale/opacity).

    Undoes, in order: per-object normalization (§3.4), then the log-scale
    and logit-opacity activations that read_splat() applies at load time.
    """
    xyz_n = tokens[:, 0:3]
    log_scale_n = tokens[:, 3:6]
    quat = tokens[:, 6:10]
    opacity_logit = tokens[:, 10]
    rgb = np.clip(tokens[:, 11:14], 0.0, 1.0)

    xyz = xyz_n * radius + centre
    log_scale = log_scale_n + np.log(radius).astype(np.float32)
    scale = np.exp(log_scale)
    opacity = 1.0 / (1.0 + np.exp(-opacity_logit))

    return dict(
        name=name,
        n=len(xyz),
        xyz=xyz.astype(np.float32),
        scale=scale.astype(np.float32),
        log_scale=log_scale.astype(np.float32),
        quat=quat.astype(np.float32),
        opacity=opacity.astype(np.float32),
        opacity_logit=opacity_logit.astype(np.float32),
        rgb=rgb.astype(np.float32),
    )


# --------------------------------------------------------------- split (§2.3)
def make_split(files, n_val, seed):
    """Deterministic train/val split by object name (not index — stable
    even if `files` ordering changes). Returns dict(train=[...], val=[...])
    of object stems, sorted."""
    names = sorted(f.stem for f in files)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(names))
    val_idx = set(perm[:n_val].tolist())
    train = sorted(names[i] for i in range(len(names)) if i not in val_idx)
    val = sorted(names[i] for i in range(len(names)) if i in val_idx)
    return dict(train=train, val=val, seed=seed)


def load_or_make_split(path, files, n_val, seed):
    """Never silently regenerate: if split.json exists, load and use it
    as-is (report size mismatch as a loud warning, not a silent overwrite)."""
    path = Path(path)
    if path.exists():
        split = json.loads(path.read_text())
        print(f"reusing existing split at {path}: "
              f"{len(split['train'])} train / {len(split['val'])} val "
              f"(delete the file to regenerate)")
        return split
    split = make_split(files, n_val, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2))
    print(f"wrote new split to {path}: {len(split['train'])} train / {len(split['val'])} val")
    return split


# --------------------------------------------------------------- §3.5
def compute_channel_stats(tokens):
    """Per-channel mean/std over `tokens` (any leading shape, last dim=14).
    Quaternion channels (6:10) are excluded from standardization — set to
    (mean=0, std=1) by fiat (DESIGN_SPEC §3.5): standardizing them with
    their own stats would destroy the unit-norm structure that makes them
    rotations.
    """
    flat = tokens.reshape(-1, N_CHANNELS)
    mean = flat.mean(0).astype(np.float64)
    std = flat.std(0).astype(np.float64)
    mean[QUAT_SLICE] = 0.0
    std[QUAT_SLICE] = 1.0
    std = np.maximum(std, 1e-8)
    return dict(mean=mean.tolist(), std=std.tolist())


def standardize(tokens, stats):
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return (tokens - mean) / std


def unstandardize(tokens, stats):
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return tokens * std + mean


# --------------------------------------------------------------- mirror axis (§3.6)
def detect_mirror_axis(files, up_axis, sample_size=40, n_check_points=2000, seed=0):
    """Which horizontal axis is the chair's left-right mirror plane?

    We know up_axis (z, from EDA) but not which of the remaining two axes
    is left-right (bilaterally symmetric) vs. front-back (broken by
    armrests/backrest) — nothing so far determines this, it must be
    checked, not assumed, the same way up_axis was.

    For a sample of objects, reflect across each candidate axis and score
    self-similarity via mean nearest-neighbour distance (both directions,
    using scipy cKDTree) between the reflected and original point sets,
    normalized by object radius. The true mirror plane should score much
    lower (near 0) than the wrong one.

    Returns (winning_axis, {axis: mean_score, ...}).
    """
    from scipy.spatial import cKDTree
    from splat_utils import read_splat

    rng = np.random.default_rng(seed)
    candidates = [a for a in range(3) if a != up_axis]
    idx = rng.choice(len(files), min(sample_size, len(files)), replace=False)

    scores = {a: [] for a in candidates}
    for i in idx:
        s = read_splat(files[i])
        xyz = s["xyz"]
        centre = 0.5 * (xyz.min(0) + xyz.max(0))
        radius = float(np.abs(xyz - centre).max())
        p = xyz - centre
        if len(p) > n_check_points:
            sel = rng.choice(len(p), n_check_points, replace=False)
            p = p[sel]

        tree = cKDTree(p)
        for a in candidates:
            mirrored = p.copy()
            mirrored[:, a] *= -1.0
            d_fwd, _ = cKDTree(mirrored).query(p, k=1)
            d_bwd, _ = tree.query(mirrored, k=1)
            score = 0.5 * (d_fwd.mean() + d_bwd.mean()) / radius
            scores[a].append(score)

    mean_scores = {a: float(np.mean(v)) for a, v in scores.items()}
    winner = min(mean_scores, key=mean_scores.get)
    return winner, mean_scores
