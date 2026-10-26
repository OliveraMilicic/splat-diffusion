"""
batching.py — offset/batch-index conversions for the "concatenate points
across objects" sparse-tensor batching convention (STAGE_SPEC §4:
"Batch by concatenating points across objects with an offset /
batch-index array").

Same convention as `reference/ptv3-official/model.py`'s
`offset2batch`/`batch2offset` (consulted for structural correctness per
TRAINING_SPEC §0, not imported): `offset[i]` is the cumulative point
count through object `i` inclusive, so `offset[-1] == total_points` and
`offset[0] == len(object_0)`. `batch_idx` is the same information
expanded to one entry per point — which object each point belongs to —
used to broadcast a per-object embedding (e.g. the timestep embedding)
onto every point via indexing.

    lengths      [n_0, n_1, ..., n_{B-1}]
    offset       [n_0, n_0+n_1, ..., sum(n)]            same length as lengths
    batch_idx    [0]*n_0 + [1]*n_1 + ... + [B-1]*n_{B-1}   length sum(n)
"""

import torch


def lengths_to_offset(lengths) -> torch.Tensor:
    """[n_0, ..., n_{B-1}] -> cumulative offset, int64, shape (B,)."""
    lengths = torch.as_tensor(lengths, dtype=torch.long)
    return torch.cumsum(lengths, dim=0)


def offset_to_lengths(offset: torch.Tensor) -> torch.Tensor:
    """Inverse of `lengths_to_offset`: offset -> per-object counts via
    a leading-zero diff (matches reference `offset2bincount`)."""
    zero = torch.zeros(1, dtype=offset.dtype, device=offset.device)
    return torch.diff(offset, prepend=zero)


def offset_to_batch_idx(offset: torch.Tensor) -> torch.Tensor:
    """offset (B,) -> batch_idx (sum(n),): which object each point in the
    concatenated array belongs to."""
    lengths = offset_to_lengths(offset)
    return torch.arange(len(lengths), device=offset.device).repeat_interleave(lengths)


def batch_idx_to_offset(batch_idx: torch.Tensor) -> torch.Tensor:
    """Inverse of `offset_to_batch_idx`."""
    return torch.cumsum(batch_idx.bincount(), dim=0).long()
