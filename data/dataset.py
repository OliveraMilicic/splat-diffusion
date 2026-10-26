"""
data/dataset.py — torch Dataset over the shard store, wiring
`data/shards.py` (read) and `data/augment.py` (per-item transform)
together (TRAINING_SPEC §1, §4).

Augmentation re-rolls every `__getitem__` call, not once at construction
(TRAINING_SPEC §4): which of the 8 precomputed FPS subsets, plus
mirror/rotation/jitter on top, are all re-sampled per draw so the same
object looks different across epochs. Each call builds its own
`np.random.default_rng()` with no fixed seed — it pulls fresh OS entropy,
which is also what keeps augmentation streams independent across
`DataLoader` worker processes (a per-index-seeded RNG would make results
deterministic across epochs and workers alike, which is exactly what
TRAINING_SPEC §4 says not to do; full reproducibility is deliberately not
a goal here, TRAINING_SPEC §9).
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from preprocessing import standardize
from .augment import augment_object
from .shards import ShardStore, load_shard_store


class ChairTokenDataset(Dataset):
    def __init__(self, store: ShardStore, cfg, split: str = "train"):
        self.store = store
        self.cfg = cfg
        names = store.names_for(split)
        if cfg.overfit_n is not None:
            if cfg.overfit_n > len(names):
                raise ValueError(
                    f"overfit_n={cfg.overfit_n} > {len(names)} objects in split {split!r}"
                )
            names = names[: cfg.overfit_n]
        self.names = names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        rng = np.random.default_rng()  # fresh entropy every call, see module docstring
        subset = int(rng.integers(0, self.store.n_subsets))
        tok = self.store.raw(name, subset)

        tok = augment_object(tok, self.store.mirror_axis, self.store.up_axis, rng, self.cfg)

        n = self.cfg.n_gaussians
        stored_n = tok.shape[0]
        if n < stored_n:
            sel = rng.choice(stored_n, n, replace=False)
            tok = tok[sel]
        elif n > stored_n:
            raise ValueError(f"cfg.n_gaussians={n} > stored subset size {stored_n}")

        tok = standardize(tok, self.store.norm_stats)
        return torch.from_numpy(np.ascontiguousarray(tok, dtype=np.float32))


def build_dataloader(cfg, split: str = "train", shuffle: bool = None) -> DataLoader:
    """Construct the ShardStore + Dataset + DataLoader for one split.
    Caller (scripts/train.py) is responsible for the
    `torch.multiprocessing.set_sharing_strategy('file_system')` call at
    the very top of the entry script, before this runs — TRAINING_SPEC §4,
    the /dev/shm 64MB limit crashes `num_workers>0` otherwise.
    """
    store = load_shard_store(cfg.shard_path)
    dataset = ChairTokenDataset(store, cfg, split=split)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        drop_last=(split == "train"),
        pin_memory=torch.cuda.is_available(),
    )
