"""
stage_config.py — STAGE_SPEC.md config: the per-stage channel split (§1),
plus the paths and normalization stats every STAGE_SPEC module needs.

Deliberately separate from `config.py` (the pre-STAGE_SPEC EDM/N=4096
config): STAGE_SPEC §0 supersedes EDM preconditioning, fixed N=4096, and
EMA 0.9999 wholesale, so reusing that dataclass would mean overriding most
of its fields at every call site. `config.py` stays as-is for whatever
still imports it (`check_dataset.py`); this file is what
`eval/`, `train/`, `model/`, `diffusion/` import from instead.

Channel layout (14) — see `preprocessing.py` docstring / STAGE_SPEC §2:
    0:3   x, y, z             3   frozen conditioning, every stage
    3:6   log sx, sy, sz      3
    6:10  qw, qx, qy, qz      4   canonicalized, w >= 0, NOT standardized
    10    logit opacity       1
    11:14 r, g, b             3
"""

import dataclasses
import json
import os
from pathlib import Path

import numpy as np

N_CHANNELS = 14


def split_channels(tok: np.ndarray, cfg: "StageConfig"):
    """(N, 14) -> (cond (N, n_cond), target (N, n_target)), config-driven
    (STAGE_SPEC §1). Works on either raw or standardized tokens — it's
    just column indexing, applied AFTER augmentation
    (NORMALIZATION.md §5: rotation touches xyz and quat, both
    conditioning channels)."""
    return tok[:, cfg.cond_ch], tok[:, cfg.target_ch]


def recombine_channels(cond: np.ndarray, target: np.ndarray, cfg: "StageConfig") -> np.ndarray:
    """Inverse of `split_channels`: scatter cond/target columns back into
    a full (N, 14) array by the same index lists. `cond`/`target` can be
    numpy arrays or torch tensors (same call shape either way)."""
    n = cond.shape[0]
    assert target.shape[0] == n
    if hasattr(cond, "new_empty"):  # torch tensor
        out = cond.new_empty((n, N_CHANNELS))
    else:
        out = np.empty((n, N_CHANNELS), dtype=cond.dtype)
    out[:, cfg.cond_ch] = cond
    out[:, cfg.target_ch] = target
    return out

# ----------------------------------------------------------------- data location
# Resolved at import from the first candidate that exists, because the two
# machines this runs on see different filesystems:
#
#   login01 (slurm head node)  — /lustre mounted, /home mounted
#   GPU Jupyter container      — /home only, NO /lustre
#
# So a hardcoded /lustre path works on one and FileNotFoundErrors on the
# other. `/home/mls04/data/...` is the copy that both can see (the .ply
# set is ~8.9GB; `tokens.dat` deliberately NOT copied — it's the FPS@4096
# shard STAGE_SPEC §3 says not to reuse at N=16384, so nothing reads it).
# $SPLAT_SHARD_PATH / $SPLAT_DATA_DIR override if a third location shows up.
#
# Shard artifacts REUSED as-is: split.json, norm_stats.json (already
# matches STAGE_SPEC §2's table exactly — see `assert_stats_match` — so no
# recomputation, which NORMALIZATION.md supersedes as a no-op anyway).

_SHARD_CANDIDATES = [
    "/lustre/data/mls04/ShapeSplatsV1/shards",
    "/home/mls04/data/ShapeSplatsV1/shards",
]
_DATA_CANDIDATES = [
    "/lustre/data/mls04/ShapeSplatsV1/extracted",
    "/home/mls04/data/ShapeSplatsV1/extracted",
]


def _resolve(env_var: str, candidates: list, what: str) -> str:
    override = os.environ.get(env_var)
    if override:
        return override
    for path in candidates:
        if Path(path).is_dir():
            return path
    # Don't raise at import time — a caller may only need the config object
    # (e.g. channel splits) and never touch the filesystem. Return the first
    # candidate so the eventual failure names a concrete missing path.
    print(f"WARNING: no {what} found in any of {candidates}; "
          f"set ${env_var} to override. Falling back to {candidates[0]}.")
    return candidates[0]


SHARD_PATH = _resolve("SPLAT_SHARD_PATH", _SHARD_CANDIDATES, "shard dir")
DATA_DIR = _resolve("SPLAT_DATA_DIR", _DATA_CANDIDATES, "extracted .ply dir")

# ----------------------------------------------------------------- §1 channel split
# target/cond channel indices per stage. "a stage change is one line":
# StageConfig(stage=2) instead of StageConfig(stage=1), nothing else moves.
STAGES = {
    1: dict(name="rgb",            target=[11, 12, 13],
            cond=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    2: dict(name="rgb+opacity",    target=[10, 11, 12, 13],
            cond=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    3: dict(name="+log-scale",     target=[3, 4, 5, 10, 11, 12, 13],
            cond=[0, 1, 2, 6, 7, 8, 9]),
    4: dict(name="all",            target=list(range(3, 14)),
            cond=[0, 1, 2]),
}

for _s, _spec in STAGES.items():
    assert sorted(_spec["target"] + _spec["cond"]) == list(range(N_CHANNELS)), (
        f"stage {_s} ({_spec['name']}): target+cond don't partition 0..{N_CHANNELS - 1}"
    )
del _s, _spec

QUAT_IDX = [6, 7, 8, 9]  # never standardized regardless of which side of the split it's on

# Per-channel names, for logging and figure labels. Hardcoding ("r","g","b")
# at the call site was fine while stage 1 was the only target; at stage 3 the
# target is 7 channels and a per-channel loss labelled "r" that is actually
# log_sx is worse than no label at all.
CHANNEL_NAMES = ["x", "y", "z", "log_sx", "log_sy", "log_sz",
                 "qw", "qx", "qy", "qz", "opacity", "r", "g", "b"]
assert len(CHANNEL_NAMES) == N_CHANNELS


# ----------------------------------------------------------------- norm stats
# STAGE_SPEC §2's table, for `assert_stats_match` — catches a stale/wrong
# norm_stats.json loudly instead of silently training on the wrong scale.
EXPECTED_STATS = dict(
    mean=[-0.1004, 0.0022, 0.0462, -3.8603, -3.6891, -3.5918,
          0.0, 0.0, 0.0, 0.0, -0.3375, 0.3500, 0.2937, 0.2955],
    std=[0.4163, 0.4063, 0.4605, 1.8206, 1.5816, 1.5362,
         1.0, 1.0, 1.0, 1.0, 1.9068, 0.1927, 0.1989, 0.2034],
)


def load_norm_stats(shard_path: str = SHARD_PATH) -> dict:
    stats = json.loads((Path(shard_path) / "norm_stats.json").read_text())
    assert_stats_match(stats)
    return stats


def assert_stats_match(stats: dict, tol: float = 2e-3):
    mean, std = np.asarray(stats["mean"]), np.asarray(stats["std"])
    exp_mean, exp_std = np.asarray(EXPECTED_STATS["mean"]), np.asarray(EXPECTED_STATS["std"])
    assert mean.shape == (N_CHANNELS,) and std.shape == (N_CHANNELS,)
    bad_mean = np.abs(mean - exp_mean) > tol
    bad_std = np.abs(std - exp_std) > tol
    assert not (bad_mean.any() or bad_std.any()), (
        f"norm_stats.json doesn't match STAGE_SPEC §2's table (tol={tol}) — "
        f"mean mismatch at {np.where(bad_mean)[0].tolist()}, "
        f"std mismatch at {np.where(bad_std)[0].tolist()}. Measured number is "
        f"off from what the spec assumed — per CLAUDE.md, stop and say so "
        f"rather than silently proceeding."
    )


def load_split(shard_path: str = SHARD_PATH) -> dict:
    return json.loads((Path(shard_path) / "split.json").read_text())


# ----------------------------------------------------------------- run config
@dataclasses.dataclass
class StageConfig:
    # Stage 3 is the current build target: log-scale + opacity + RGB are
    # diffused, xyz + quaternion stay frozen conditioning. Stages 1/2 remain
    # selectable (`--stage`) and the gate runs recorded in PLAN.md were stage 1.
    stage: int = 3
    shard_path: str = SHARD_PATH
    data_dir: str = DATA_DIR

    # data — STAGE_SPEC §3, §6
    n_gaussians: int = 16384        # N = min(16384, n_gaussians), random subsample
    mirror_prob: float = 0.5        # STAGE_SPEC §6: "random mirror about axis 1"
    num_workers: int = 4            # TRAINING_SPEC §4: start at 4, not 0
    n_train_objects: int = 0        # 0 = the whole train split. >0 trains on a
                                     # seeded random subset, recorded to
                                     # runs/<name>/train_subset.json (train/loop.py).
                                     # The VAL split is never subset, so runs at
                                     # different train sizes stay comparable.

    # model — STAGE_SPEC §4
    d_model: int = 384
    n_blocks: int = 12
    n_heads: int = 6
    patch_size: int = 1024
    mlp_ratio: int = 4
    xcpe_k: int = 16

    # diffusion — STAGE_SPEC §5 + GaussianCube's loss (diffusion/losses.py)
    T: int = 1000                   # cosine schedule, zero terminal SNR
    ema_decay: float = 0.999        # NOT 0.9999 — see STAGE_SPEC §5
    learn_sigma: bool = True        # GaussianCube configs/shapenet_uncond.yml:
                                     # learn_sigma true -> out_channels = 2*target,
                                     # loss = MSE + VB (diffusion/losses.py)
    min_snr_gamma: float = 0.0      # 0 = off, which is what GaussianCube ships
                                     # (min_snr=False); 5.0 is their disabled path
    clip_input_p: float = 0.0       # GaussianCube dataset.py's `dynamic_clip`: a
                                     # per-object per-channel symmetric quantile
                                     # clip of the standardized target. Their value
                                     # is 0.995. **OFF here, decided on a
                                     # measurement, not by default-copying** (see
                                     # check_target_stats.py): the standardized
                                     # stage-3 target's worst tail over 40 train
                                     # objects is max|x| = 6.7 with only 2-3% of
                                     # log-scale values past 3 sigma and NOTHING
                                     # past 10 — the raw anisotropy tail (p99 39.3,
                                     # max 240.8) is a scale RATIO and largely
                                     # disappears under log + standardization.
                                     # p=0.995 would clip every object at ~3.7 sigma,
                                     # i.e. deliberately shrink each object's 0.5%
                                     # largest Gaussians, which are the big flat
                                     # splats covering seats and backs. Real cost,
                                     # no measured benefit. Set >0 to re-enable if
                                     # training turns unstable.

    # optimization
    batch_size: int = 8             # STAGE_SPEC §3 memory budget at N=16384
    lr: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 0
    run_name: str = "stage1_rgb"

    # full training run (train/loop.py)
    max_steps: int = 60_000
    warmup_steps: int = 1000        # GaussianCube's `num_warmup_steps = 1000`
                                     # (train.py:135). Was 500 for no stated
                                     # reason; matched to theirs to remove an
                                     # unexplained difference.
    amp: str = "bf16"               # measured 2.1x faster than fp32 AND less memory
                                     # (see PLAN.md perf section) -- fp32 only for gates
    log_every: int = 10
    render_every: int = 2000        # TRAINING_SPEC §7: every ~2000-5000 steps
    ckpt_every: int = 1000
    n_render_objects: int = 4       # TRAIN objects in the mid-training grid
    # NO validation set and no val_every: nothing during training selects a
    # model (no hyperparameter search, no early stopping, run length fixed by
    # wall clock), so a validation split would have no job. The held-out
    # objects are the TEST set, touched once at the end — eval/test_eval.py.
    sample_steps: int = 50          # DPM-Solver steps for render/eval sampling

    @property
    def target_ch(self) -> list:
        return STAGES[self.stage]["target"]

    @property
    def cond_ch(self) -> list:
        return STAGES[self.stage]["cond"]

    @property
    def n_target_ch(self) -> int:
        return len(self.target_ch)

    @property
    def n_cond_ch(self) -> int:
        return len(self.cond_ch)

    @property
    def target_names(self) -> list:
        """Human-readable names for the target channels, in target order —
        for per-channel loss keys and figure labels."""
        return [CHANNEL_NAMES[c] for c in self.target_ch]

    @property
    def n_model_out_ch(self) -> int:
        """What the network's final Linear must emit. Doubled under
        `learn_sigma` (GaussianCube: `in_channels: 14, out_channels: 28`) —
        the second half is `LEARNED_RANGE` variance, see
        `diffusion/losses.py`."""
        return 2 * self.n_target_ch if self.learn_sigma else self.n_target_ch

    def run_dir(self, root: str = "runs") -> Path:
        return Path(root) / self.run_name

    def save(self, root: str = "runs") -> Path:
        out_dir = self.run_dir(root)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "config.json"
        path.write_text(json.dumps(dataclasses.asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, path: str) -> "StageConfig":
        return cls(**json.loads(Path(path).read_text()))


def lr_at(step: int, cfg: StageConfig) -> float:
    """Linear warmup, then constant.

    Matches GaussianCube: `LambdaLR` with
    `lr_lambda = min(step / num_warmup_steps, 1)` over `num_warmup_steps =
    1000` (train.py:135-141), and `_anneal_lr` is a literal `return` — no
    decay. Lives here rather than in either training script because BOTH
    need it and they must not drift: `train/overfit.py` had no warmup at all
    while `train/loop.py` did, so the gate was testing a different optimizer
    schedule than the run it gates.
    """
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    return cfg.lr
