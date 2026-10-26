"""
model/conditioning.py — STAGE_SPEC §11 build-order step 4 (part 1 of 2,
with `denoiser.py`). The channel-split glue between `stage_config.py`
and `model/ptv3.py`'s generic `(x_in, pos, offset, t)` interface:
assembling the per-point network input from (cond, noisy target), and
extracting `pos` from it for the backbone's serialization.
"""

import torch

from stage_config import StageConfig


def assert_xyz_first(cfg: StageConfig):
    """`model/ptv3.py::PTv3` reads `pos` as `cond[:, 0:3]` — valid only
    because every `STAGES` entry (`stage_config.py`) lists channels 0,1,2
    first (STAGE_SPEC §1: xyz is frozen conditioning at every stage).
    Cheap guard against a future stage definition breaking that
    assumption silently."""
    assert cfg.cond_ch[:3] == [0, 1, 2], (
        f"stage {cfg.stage}: cond_ch does not start with [0,1,2] ({cfg.cond_ch[:3]}) — "
        f"extract_pos()'s cond[:, 0:3]=xyz assumption no longer holds"
    )


def extract_pos(cond: torch.Tensor, cfg: StageConfig) -> torch.Tensor:
    """xyz is always the first 3 columns of `cond` (see `assert_xyz_first`)."""
    assert_xyz_first(cfg)
    return cond[:, 0:3]


def build_input(cond: torch.Tensor, target_noisy: torch.Tensor) -> torch.Tensor:
    """The backbone's per-point input: conditioning channels concatenated
    with the noisy target `y_t` (NORMALIZATION.md §5's last step before
    `model(y_t, t, cond, offset)`)."""
    return torch.cat([cond, target_noisy], dim=-1)
