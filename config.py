"""
config.py — the training config. One dataclass, no Hydra (TRAINING_SPEC §8:
a 5-day sprint doesn't need multi-run composition).

Every run writes its resolved config to `runs/<run_name>/config.json`
before training starts, via `TrainConfig.save()` — not after, so a crashed
run still tells you what it was trying to do (TRAINING_SPEC §2).
"""

import dataclasses
import json
from pathlib import Path


@dataclasses.dataclass
class TrainConfig:
    # data
    shard_path: str = "/lustre/data/mls04/ShapeSplatsV1/shards"
    channels: int = 14          # or 13 under the log-Cholesky ablation (DESIGN_SPEC §3.3)
    n_gaussians: int = 4096

    # model — DESIGN_SPEC §7.1
    d_model: int = 384
    n_blocks: int = 12
    n_heads: int = 6
    patch_size: int = 1024

    # diffusion — DESIGN_SPEC §4
    sigma_data: float = 0.9264   # MEASURED Day 1 value (shards/norm_stats.json), not the 1.0 placeholder
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    p_mean: float = -1.2
    p_std: float = 1.2

    # optimization — DESIGN_SPEC §7.1
    batch_size: int = 32
    lr: float = 1e-4
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    max_steps: int = 150_000

    # augmentation — DESIGN_SPEC §3.6, applied per-__getitem__ (TRAINING_SPEC §4)
    mirror_prob: float = 0.5
    rotation_aug: bool = True
    scale_jitter_range: tuple = (0.9, 1.1)
    position_jitter_std: float = 0.002

    # run mode
    overfit_n: int | None = None   # e.g. 64 — see TRAINING_SPEC §12 phase G
    seed: int = 0
    run_name: str = "default"
    resume_from: str | None = None
    num_workers: int = 4            # TRAINING_SPEC §4: start at 4, not 0

    def run_dir(self, root: str = "runs") -> Path:
        return Path(root) / self.run_name

    def save(self, root: str = "runs") -> Path:
        """Write the resolved config to `<root>/<run_name>/config.json`.
        Call this before training starts, not after (TRAINING_SPEC §2)."""
        out_dir = self.run_dir(root)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "config.json"
        path.write_text(json.dumps(dataclasses.asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, path: str) -> "TrainConfig":
        data = json.loads(Path(path).read_text())
        data["scale_jitter_range"] = tuple(data["scale_jitter_range"])
        return cls(**data)
