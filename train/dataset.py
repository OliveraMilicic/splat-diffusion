"""
train/dataset.py — STAGE_SPEC §11 build-order step 2.

Reads full-resolution objects straight from `.ply`
(`preprocessing.full_object_tokens`), not the old `tokens.dat` shard —
that shard is FPS-subsampled to 4096 points, the wrong shape for this
stage's `N = min(16384, n)` (STAGE_SPEC §3). At ~35ms/object this is fast
enough for on-the-fly loading; no new shard format needed.

**Flagging a real discrepancy between two specs rather than silently
picking one (CLAUDE.md: "if something contradicts the spec, stop and say
so")**: `NORMALIZATION.md` §5's pipeline diagram lists standardization
*before* augmentation:

    load shard (already ... per-channel standardized)
      -> random subsample -> augment: mirror, rotate ...

but `data/augment.py`'s docstring — backed by a passing exact-to-float
Sigma-conjugation test, `check_augmentation.py` — requires the opposite
order, and gives a concrete reason: xyz's three channels have unequal
per-channel std (0.4163 / 0.4063 / 0.4605), so a standardized xyz vector
is not an isotropically-scaled space, and rotating it as one introduces
a real (~13%, from the std ratio) distortion. `NORMALIZATION.md`'s own
"two ordering traps" paragraph right below the diagram doesn't call out
standardize-vs-augment as one of them, which reads as the diagram being
a loose sketch rather than a literal, checked sequence — unlike the
"augment before serialize" / "split after augment" traps it does call
out explicitly.

**Resolution used here**: augment on RAW (unstandardized) tokens, same
as `data/augment.py` already assumes and already tested; standardize
after. Order actually implemented:

    read .ply -> full_object_tokens (RAW, per-object normalized)
      -> random subsample to min(cfg.n_gaussians, n)
      -> augment (RAW): mirror (axis 1, prob cfg.mirror_prob),
         rotate (axis 2, uniform [0, 2pi)), re-canonicalize quat w>=0
      -> standardize (per-channel stats)
      -> split into (cond, target) by stage config
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from batching import lengths_to_offset, offset_to_batch_idx
from data.augment import mirror, rotate_about_axis
from diffusion.schedule import dynamic_clip
from preprocessing import full_object_tokens, standardize
from splat_utils import find_ply_files, read_splat
from stage_config import StageConfig, load_norm_stats, load_split, split_channels


class ChairPointDataset(Dataset):
    """One item = one object's (cond, target) token arrays, full
    variable length (`n` in `[min(11100, cfg.n_gaussians), cfg.n_gaussians]`
    depending on the object) — no padding (STAGE_SPEC §10). Collate with
    `collate_variable_n` to concatenate a batch into the
    offset/batch-index sparse-tensor convention (STAGE_SPEC §4).

    Augmentation re-rolled every `__getitem__` call (fresh
    `np.random.default_rng()`, no fixed seed) — same rationale as
    `data/dataset.py`: keeps epochs varying and keeps worker-process
    augmentation streams independent without extra plumbing. Reproduce
    an exact draw by seeding `np.random` at the call site if ever needed;
    not a goal here (TRAINING_SPEC §9).
    """

    def __init__(self, cfg: StageConfig, split: str, augment: bool = None,
                 names: list = None):
        """`names` restricts the split to a subset (the 100-object
        experiment). It must be a subset of the split's own names —
        checked, because a typo'd or stale name list would otherwise
        silently train on fewer objects than intended. The caller owns
        choosing and persisting the subset (train/loop.py writes it to
        `runs/<name>/train_subset.json` once); this class only applies it,
        so the split file on disk is never rewritten."""
        self.cfg = cfg
        self.split = split
        self.augment = (split == "train") if augment is None else augment

        split_dict = load_split(cfg.shard_path)
        # "test" reads the on-disk "val" key. There is no validation set in
        # this project -- no hyperparameter search and no early stopping, so
        # a third split would have no job -- and the 170 held-out objects are
        # the TEST set, touched once at the end of a run. The key in
        # split.json stays "val" because that file is written once and never
        # regenerated (CLAUDE.md never-cut list); renaming it would mean
        # rewriting the split, which is the one thing not to do.
        key = "val" if split == "test" else split
        if key not in split_dict:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        self.names = split_dict[key]

        if names is not None:
            unknown = sorted(set(names) - set(self.names))
            if unknown:
                raise ValueError(
                    f"{len(unknown)} requested names are not in the {split!r} split, "
                    f"e.g. {unknown[:3]} — a subset must be a subset, or the run "
                    f"would train on a different set than it records."
                )
            self.names = list(names)

        self.norm_stats = load_norm_stats(cfg.shard_path)  # asserts vs STAGE_SPEC §2's table
        self.up_axis = self.norm_stats["up_axis"]
        self.mirror_axis = self.norm_stats["mirror_axis"]

        files = find_ply_files(cfg.data_dir)
        self.name_to_file = {f.stem: f for f in files}
        missing = [n for n in self.names if n not in self.name_to_file]
        if missing:
            raise FileNotFoundError(f"{len(missing)} split objects have no .ply under "
                                     f"{cfg.data_dir}, e.g. {missing[:3]}")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        splat = read_splat(self.name_to_file[name])
        # centre/radius are returned to the caller: per-object normalisation
        # (preprocessing §3.4) is only invertible with THIS object's pair, and
        # decoding a prediction with the wrong one silently yields a
        # mis-scaled splat rather than an error. The render callback in
        # train/loop.py needs them.
        tok, centre, radius = full_object_tokens(splat)  # RAW, (n_total, 14)

        rng = np.random.default_rng()  # fresh entropy every call, see class docstring
        n_total = tok.shape[0]
        n = min(self.cfg.n_gaussians, n_total)
        if n < n_total:
            sel = rng.choice(n_total, n, replace=False)
            tok = tok[sel]
        # else: n == n_total, keep everything — the "genuinely variable N" case
        # (STAGE_SPEC §3: smallest objects have 11,100 < 16,384 Gaussians)

        if self.augment:
            if rng.random() < self.cfg.mirror_prob:
                tok = mirror(tok, self.mirror_axis)
            theta = rng.uniform(0.0, 2 * np.pi)
            tok = rotate_about_axis(tok, self.up_axis, theta)

        tok_std = standardize(tok, self.norm_stats)
        cond, target = split_channels(tok_std, self.cfg)

        # GaussianCube's outlier guard, applied to the TARGET only — the
        # channels being diffused. Deliberately not applied to `cond`:
        # clipping xyz would move the geometry the model is conditioned on
        # AND that its output is rendered against, and clipping the
        # quaternion channels (unstandardized by design, STAGE_SPEC §2)
        # would break unit norm. Off by default, see `cfg.clip_input_p`.
        if self.cfg.clip_input_p:
            target = dynamic_clip(torch.from_numpy(np.ascontiguousarray(target)),
                                   self.cfg.clip_input_p).numpy()

        return dict(
            cond=torch.from_numpy(np.ascontiguousarray(cond, dtype=np.float32)),
            target=torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32)),
            n=cond.shape[0],
            name=name,
            centre=centre,
            radius=float(radius),
        )


def collate_variable_n(batch: list) -> dict:
    """List of per-object dicts -> one batch, concatenated along the
    point dimension (STAGE_SPEC §4's sparse-tensor convention, not
    padding — STAGE_SPEC §10 non-goal). `offset[i]` = cumulative point
    count through object i inclusive; `batch_idx` gives each point's
    object index directly (see `batching.py`)."""
    lengths = [b["n"] for b in batch]
    cond = torch.cat([b["cond"] for b in batch], dim=0)
    target = torch.cat([b["target"] for b in batch], dim=0)
    offset = lengths_to_offset(lengths)
    batch_idx = offset_to_batch_idx(offset)
    return dict(
        cond=cond,
        target=target,
        offset=offset,
        batch_idx=batch_idx,
        lengths=torch.as_tensor(lengths, dtype=torch.long),
        names=[b["name"] for b in batch],
        centres=[b["centre"] for b in batch],   # per-object, see __getitem__
        radii=[b["radius"] for b in batch],
    )


def build_dataloader(cfg: StageConfig, split: str, augment: bool = None,
                      shuffle: bool = None, names: list = None,
                      num_workers: int = None) -> DataLoader:
    """Caller (entry script) is responsible for
    `torch.multiprocessing.set_sharing_strategy('file_system')` at the
    very top, before this runs — CLAUDE.md / TRAINING_SPEC §4, the
    /dev/shm 64MB limit crashes `num_workers>0` otherwise.

    **`file_system` does not escape that 64MB ceiling**, it only changes
    the failure into a legible `RuntimeError: unable to allocate shared
    memory` instead of a bare `Bus error`. The budget still has to be
    respected: at stage 1, N=32768, one object is ~1.8MB of tensors, a
    batch of 4 is 7.3MB, and `num_workers` workers each eagerly fill
    `prefetch_factor=2` batches — so 8 workers means up to 117MB in
    flight against 64MB of /dev/shm. Training survives that only because
    the main process drains the queue as fast as the workers fill it; the
    dangerous moment is loader STARTUP, when every worker bursts at once.

    `num_workers` overrides `cfg.num_workers`. **Pass 0 for evaluation
    loaders.** They ran with 8 and blew /dev/shm at the end of
    `colourful8`, after training had finished — and they gain nothing from
    workers anyway, since an evaluation is dominated by GPU sampling, not
    by the ~35ms it takes to read a .ply.
    """
    dataset = ChairPointDataset(cfg, split=split, augment=augment, names=names)
    if shuffle is None:
        shuffle = split == "train"
    if num_workers is None:
        num_workers = cfg.num_workers
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_variable_n,
        drop_last=(split == "train"),
        pin_memory=torch.cuda.is_available(),
    )
