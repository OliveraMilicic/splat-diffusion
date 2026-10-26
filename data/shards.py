"""
data/shards.py — read-only wrapper around the Day-1 shard layout built by
`build_shards.py`. Doesn't rebuild anything; loads what's already on disk:

    <shard_path>/split.json          train/val object names, fixed seed
    <shard_path>/object_meta.json    per-object {centre, radius} (undoes §3.4)
    <shard_path>/norm_stats.json     per-channel mean/std, sigma_data, axes
    <shard_path>/tokens.dat          memmap (n_objects, n_subsets, N, C) float32
    <shard_path>/tokens_meta.json    object name -> row index, shape

`ShardStore` is the thing `data/dataset.py` wraps in a torch Dataset. Kept
separate (TRAINING_SPEC §1) so it's testable/inspectable on its own,
independent of torch or augmentation.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np


@dataclasses.dataclass
class ShardStore:
    tokens: np.memmap            # (n_objects, n_subsets, N, C) float32, RAW (unstandardized)
    names: list                  # row index -> object name
    name_to_row: dict            # object name -> row index
    split: dict                  # {"train": [...], "val": [...], "seed": int}
    norm_stats: dict              # per-channel mean/std, sigma_data, up_axis, mirror_axis
    object_meta: dict             # name -> {centre, radius}

    @property
    def n_subsets(self) -> int:
        return self.tokens.shape[1]

    @property
    def n_gaussians_stored(self) -> int:
        return self.tokens.shape[2]

    @property
    def up_axis(self) -> int:
        return self.norm_stats["up_axis"]

    @property
    def mirror_axis(self) -> int:
        return self.norm_stats["mirror_axis"]

    def names_for(self, split: str) -> list:
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        return self.split[split]

    def raw(self, name: str, subset: int) -> np.ndarray:
        """One (N, C) raw (unstandardized, per-object-normalized) token
        array, as stored — copy, not a memmap view, safe to mutate
        (augmentation does)."""
        row = self.name_to_row[name]
        return np.array(self.tokens[row, subset])


def load_shard_store(shard_path: str) -> ShardStore:
    shard_dir = Path(shard_path)

    tmeta = json.loads((shard_dir / "tokens_meta.json").read_text())
    names = tmeta["names"]
    shape = tuple(tmeta["shape"])
    tokens = np.memmap(shard_dir / "tokens.dat", dtype=np.float32, mode="r", shape=shape)

    split = json.loads((shard_dir / "split.json").read_text())
    norm_stats = json.loads((shard_dir / "norm_stats.json").read_text())
    object_meta = json.loads((shard_dir / "object_meta.json").read_text())

    return ShardStore(
        tokens=tokens,
        names=names,
        name_to_row={n: i for i, n in enumerate(names)},
        split=split,
        norm_stats=norm_stats,
        object_meta=object_meta,
    )
