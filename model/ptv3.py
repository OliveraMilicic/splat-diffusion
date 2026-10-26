"""
model/ptv3.py — STAGE_SPEC §11 build-order step 3: flat PTv3-style
backbone, serialization + patch attention (STAGE_SPEC §4a-f). Checks 2
(serialization locality) and 3 (serialization round-trip) live in
`check_ptv3_stage.py`.

Structural differences from the old (pre-pivot) `ptv3.py`, all from
STAGE_SPEC superseding EDM/fixed-N:

  - No dense `(B, N, C)` tensor anywhere. Positions are frozen
    conditioning (never noised, STAGE_SPEC §1), and batching concatenates
    variable-length objects into one flat `(total_N, C)` array with an
    `offset`/`batch_idx` pair (`batching.py`) — the "sparse-tensor
    convention" STAGE_SPEC §4 asks for. Patch attention chunks that flat
    array directly; there is no per-object batch axis to carry through
    the blocks.
  - xCPE is **dropped, not approximated**: STAGE_SPEC §4d says use it
    "only if spconv or MinkowskiEngine is already installed... skip
    otherwise, do not compile sparse-conv libraries." Confirmed absent
    (PLAN.md, this session) in every Python on this box. The old
    `ptv3.py`'s k-NN xCPE approximation predates that check.
  - Serialization uses STAGE_SPEC §4a's exact `part1by2` bit-interleave
    (21-bit, not the old code's 10-bit `_dilate3`) — kept verbatim since
    the spec gives the literal function.
  - Timestep conditioning embeds a standard DDPM-style sinusoidal
    embedding of the integer timestep `t` directly (STAGE_SPEC §4f says
    "sinusoidal"), not the old EDM `c_noise(sigma)` — there is no
    `sigma` here, `t` is 0..999 (STAGE_SPEC §5).
  - Output projects to `out_ch` (the TARGET channel count for the
    active stage, e.g. 3 for stage 1's RGB), not back to all 14 channels
    — positions/other conditioning channels are never predicted.

**Padding, to keep patches from spanning objects** (STAGE_SPEC §4b:
"pad to patch boundary only"): each object's own serialized sequence is
padded up to a multiple of `patch_size` independently, using wrap-around
duplicates of that object's own (already Morton-sorted) points — never
zeros, never another object's points, so attention needs no mask.
Reconciled against `reference/ptv3-official/model.py`'s
`SerializedAttention.get_padding_and_inverse` (TRAINING_SPEC §0): same
per-object-padding idea, simplified — the reference builds one shared
`pad`/`unpad` index pair reused across all layers; `build_patch_index`
here is called fresh per block (since each block picks its own
serialization ordering per §4c) and unshuffles by direct scatter into
original point order rather than a second gather, which is simpler to
verify exactly (check 3) at the cost of doing the (cheap, index-only)
work once per block instead of once per forward.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from batching import offset_to_batch_idx, offset_to_lengths

# ----------------------------------------------------------------- serialization (§4a)
# STAGE_SPEC §4a's bit-interleave, verbatim (21-bit: 3*21=63 bits fits int64).
def part1by2(n: torch.Tensor) -> torch.Tensor:
    n = n & 0x1fffff
    n = (n | (n << 32)) & 0x1f00000000ffff
    n = (n | (n << 16)) & 0x1f0000ff0000ff
    n = (n | (n << 8)) & 0x100f00f00f00f00f
    n = (n | (n << 4)) & 0x10c30c30c30c30c3
    n = (n | (n << 2)) & 0x1249249249249249
    return n


# Four orderings (STAGE_SPEC §4a: "'trans' variants by permuting axis
# priority; four orderings total, Hilbert optional" — Hilbert skipped,
# same deviation the old ptv3.py already flagged: Z-order is simpler to
# implement correctly and DESIGN_SPEC/STAGE_SPEC name it first).
ORDERINGS = [(0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1)]

# Positions are per-object normalized to the unit sphere (elementwise
# |coord| <= 1 by construction, preprocessing.normalize_object), EXCEPT
# augmentation's z-rotation mixes x and y: a single point can have
# |x|<=1 and |y|<=1 simultaneously without their joint (rotation-
# invariant) sqrt(x^2+y^2) being <=1, so a rotated coordinate can reach
# up to sqrt(2) in the worst case. 1.5 covers that with margin; the grid
# is 21 bits either way (~1.4e-6 per cell over this range), far finer
# than needed — clamping the rare point beyond it only costs a small
# ordering perturbation, not correctness.
GRID_VALUE_RANGE = 1.5
GRID_BITS = 21


def morton_code(pos: torch.Tensor, axes=(0, 1, 2),
                 bits: int = GRID_BITS, value_range: float = GRID_VALUE_RANGE) -> torch.Tensor:
    """(N, 3) positions -> (N,) int64 Z-order code along `axes` priority."""
    q = ((pos.clamp(-value_range, value_range) + value_range)
         / (2 * value_range) * (2 ** bits - 1)).long().clamp(0, 2 ** bits - 1)
    qa, qb, qc = q[..., axes[0]], q[..., axes[1]], q[..., axes[2]]
    return part1by2(qa) | (part1by2(qb) << 1) | (part1by2(qc) << 2)


def build_patch_index(offset: torch.Tensor, code: torch.Tensor, patch_size: int):
    """Per-object Morton sort + per-object pad-to-patch-boundary (module
    docstring). Pure index computation, no learnable parameters, no
    dependence on the point features themselves — only on `pos` (via
    `code`) and object boundaries (`offset`).

    Returns:
      serial_idx        (total_padded,) int64 — gather index into the
                         ORIGINAL (total_N, C) array; `x[serial_idx]` is
                         the patch-ready, patch_size-aligned sequence.
      real_original_idx (total_N,) int64 — a permutation of
                         `arange(total_N)`: scatter target for recovering
                         original point order from the "real" (non-pad)
                         subset of the processed sequence.
      real_positions    (total_N,) int64 — positions within
                         `serial_idx`/the processed sequence holding the
                         real (non-duplicated) points, in serialized
                         order; `out[real_positions]` lines up 1:1 with
                         `real_original_idx` for the final scatter.
    """
    device = code.device
    offset = offset.to(device)
    lengths = offset_to_lengths(offset).tolist()
    starts = (offset - torch.as_tensor(lengths, device=device)).tolist()

    serial_chunks, real_orig_chunks, real_pos_chunks = [], [], []
    cursor = 0
    for start, n in zip(starts, lengths):
        local_idx = torch.arange(start, start + n, device=device)
        sort_i = torch.argsort(code[local_idx])
        sorted_idx = local_idx[sort_i]                       # (n,) original indices, serialized order

        pad_needed = (-n) % patch_size
        if pad_needed > 0:
            wrap = torch.arange(pad_needed, device=device) % n
            padded_idx = torch.cat([sorted_idx, sorted_idx[wrap]])
        else:
            padded_idx = sorted_idx

        serial_chunks.append(padded_idx)
        real_orig_chunks.append(sorted_idx)                  # the real points, in this object's serial order
        real_pos_chunks.append(torch.arange(n, device=device) + cursor)  # their position in serial_idx
        cursor += padded_idx.shape[0]

    serial_idx = torch.cat(serial_chunks)
    real_original_idx = torch.cat(real_orig_chunks)
    real_positions = torch.cat(real_pos_chunks)
    assert serial_idx.shape[0] % patch_size == 0

    # Collapse the un-shuffle from a gather+scatter pair into ONE gather.
    # `final[real_original_idx] = out[real_positions]` needs two indexed ops,
    # each carrying an atomic-scatter backward. Since real_original_idx is a
    # permutation of arange(total_N), inverting it lets the same result be
    # written `final = out[unshuffle_idx]`:
    #     final[j] = out[real_positions[inv[j]]]   where inv[real_original_idx[k]] = k
    inv = torch.argsort(real_original_idx)
    unshuffle_idx = real_positions[inv]
    return serial_idx, real_original_idx, real_positions, unshuffle_idx


# ----------------------------------------------------------------- embeddings
class SinusoidalTimestepEmbedding(nn.Module):
    """Standard DDPM-style sinusoidal embedding of an integer timestep
    (STAGE_SPEC §4f: "sinusoidal"). `t`: (B,) int64/float in [0, T)."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class PositionalFourier(nn.Module):
    """NeRF-style Fourier features of xyz (STAGE_SPEC §4d: "L=10 -> 63
    dims incl. raw"). Deterministic (log-spaced) frequencies, not a
    random projection — matches "Fourier features of xyz" read as the
    NeRF positional-encoding sense used throughout §4d, distinct from
    the timestep embedding's own sinusoidal construction."""

    def __init__(self, num_freqs: int = 10):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(num_freqs, dtype=torch.float32) * math.pi)

    @property
    def out_dim(self) -> int:
        return 3 * (1 + 2 * len(self.freqs))

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        args = pos[..., None] * self.freqs  # (N, 3, L)
        enc = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (N, 3, 2L)
        return torch.cat([pos, enc.reshape(*pos.shape[:-1], -1)], dim=-1)  # (N, 3 + 3*2L)


# ----------------------------------------------------------------- AdaLN-Zero (§4f)
class AdaLNZero(nn.Module):
    """emb -> (shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp,
    gate_mlp), one 6-way split per PTv3Block. Final linear zero-init so
    every block starts as identity (gate=0) — the "Zero" in AdaLN-Zero,
    and check 4/LOGGING.md panel 2's reason gates must be watched: they
    start at 0 and must grow, or timestep conditioning is dead."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, emb: torch.Tensor):
        return self.net(emb).chunk(6, dim=-1)  # each (B, dim)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """`x` is (total_N, dim); `shift`/`scale` are whatever
    `broadcast_per_object` returned — either (1, dim) to broadcast, or
    (total_N, dim). `(1 + scale)`, NOT `scale` (STAGE_SPEC §4f: the
    bare-`scale` version multiplies the signal by zero at init)."""
    return x * (1 + scale) + shift


def broadcast_per_object(per_obj: torch.Tensor, lengths: list) -> torch.Tensor:
    """(B, dim) per-object vector -> per-point, for `lengths` points each.

    Deliberately NOT `per_obj[batch_idx]`, which is the obvious way to
    write this and was measured to cost 89% of the entire training step.
    Its backward is an atomic scatter-add of `total_N` rows into `B` rows;
    at B=1 all total_N threads contend for a single address, which is the
    worst case an atomic scatter has. Profiled at N=32768: 1.83s of a
    2.06s step in `indexing_backward_kernel`, over 96 calls (72 of them
    from this broadcast -- 6 modulation tensors x 12 blocks).

    Both paths here backward through a plain sum reduction instead:
      B == 1  -> return (1, dim) and let elementwise ops broadcast it;
                 broadcasting's backward is `sum(dim=0)`.
      B > 1   -> cat of expands. `expand` is a view (backward = sum),
                 `cat` backward is a slice. No atomics either way.
    """
    if len(lengths) == 1:
        return per_obj  # (1, dim) broadcasts against (total_N, dim)
    return torch.cat([per_obj[i:i + 1].expand(n, -1) for i, n in enumerate(lengths)], dim=0)


# ----------------------------------------------------------------- patch attention (§4b)
class PatchAttention(nn.Module):
    def __init__(self, dim: int, heads: int, patch_size: int):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.patch_size = patch_size
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, serial_idx: torch.Tensor,
                unshuffle_idx: torch.Tensor, patch_size: int = None) -> torch.Tensor:
        total_N, C = x.shape
        # `patch_size` is the EFFECTIVE size for this batch, which PTv3.forward
        # clamps to the shortest object (see there). Attention has no
        # patch-size-dependent parameters, so varying it per batch is a pure
        # reshape change.
        P = self.patch_size if patch_size is None else patch_size

        x_s = x[serial_idx]                              # (total_padded, C)
        n_patches = x_s.shape[0] // P
        x_s = x_s.view(n_patches, P, C)

        qkv = self.qkv(x_s).view(n_patches, P, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                  # (3, n_patches, heads, P, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        out = F.scaled_dot_product_attention(q, k, v)      # (n_patches, heads, P, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(n_patches * P, C)
        out = self.proj(out)

        # One gather, instead of the `final = zeros; final[roi] = out[rp]`
        # gather+scatter pair -- see build_patch_index. Equivalent by
        # construction, and check 3 verifies it exactly.
        return out[unshuffle_idx]


# ----------------------------------------------------------------- PTv3 block
class PTv3Block(nn.Module):
    """Pre-norm, adaLN-modulated. No xCPE term (§4d, see module docstring):

        h = h + gate_attn * Attn( modulate(LN(h), emb) )
        h = h + gate_mlp  * MLP ( modulate(LN(h), emb) )
    """

    def __init__(self, dim: int, heads: int, patch_size: int, mlp_ratio: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)  # required: STAGE_SPEC §4f
        self.attn = PatchAttention(dim, heads, patch_size)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
        )
        self.adaln = AdaLNZero(dim, cond_dim)
        self.patch_size = patch_size
        self.last_gate_mean = None  # LOGGING.md panel 2: mean(|gate|), the actual computed
                                     # values for the batch just processed, not the static
                                     # weight norm — stashed here since AdaLNZero's output
                                     # depends on `emb` (hence `t`), not just its own params.
                                     # Kept as a device TENSOR, not a float: `.item()` forces a
                                     # GPU->CPU sync, and this runs once per block per forward
                                     # (12x/step). `PTv3.gate_means()` does one batched sync
                                     # instead, at logging time.

    def forward(self, h: torch.Tensor, lengths: list, emb: torch.Tensor,
                patch_index, patch_size: int = None) -> torch.Tensor:
        """`patch_index` is the precomputed `(serial_idx, real_original_idx,
        real_positions)` triple for this block's chosen ordering, built once
        per forward by `PTv3.forward` rather than here — see its docstring
        for why that's both correct and much faster."""
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaln(emb)   # each (B, dim)
        with torch.no_grad():
            self.last_gate_mean = 0.5 * (gate1.detach().abs().mean() + gate2.detach().abs().mean())
        # per-object -> per-point. See broadcast_per_object: doing this with
        # `x[batch_idx]` instead was 89% of the whole training step.
        shift1 = broadcast_per_object(shift1, lengths)
        scale1 = broadcast_per_object(scale1, lengths)
        gate1 = broadcast_per_object(gate1, lengths)
        shift2 = broadcast_per_object(shift2, lengths)
        scale2 = broadcast_per_object(scale2, lengths)
        gate2 = broadcast_per_object(gate2, lengths)

        serial_idx, _real_original_idx, _real_positions, unshuffle_idx = patch_index

        h = h + gate1 * self.attn(modulate(self.norm1(h), shift1, scale1),
                                   serial_idx, unshuffle_idx, patch_size)
        h = h + gate2 * self.mlp(modulate(self.norm2(h), shift2, scale2))
        return h


# ----------------------------------------------------------------- full backbone
class PTv3(nn.Module):
    """`forward(x_in, pos, offset, t)`:
      x_in  (total_N, in_ch)   already-assembled per-point input
                                (conditioning.py's job: concat cond +
                                noisy target — NOT built here, this class
                                takes whatever `in_ch` it's told)
      pos   (total_N, 3)       frozen xyz, read from `x_in`'s caller-known
                                slice by convention (cond[:, 0:3] is xyz
                                at every stage, STAGE_SPEC §1) — passed
                                separately so this class stays agnostic
                                to which columns of `x_in` are which
      offset (B,)              STAGE_SPEC §4 sparse-tensor convention
      t     (B,)                integer timestep per object

    Returns (total_N, out_ch) — `out_ch` is the TARGET channel count.
    """

    def __init__(self, in_ch: int, out_ch: int, d_model: int = 384, depth: int = 12,
                 heads: int = 6, patch_size: int = 1024, mlp_ratio: int = 4,
                 cond_dim: int = 384, pos_fourier_L: int = 10):
        super().__init__()
        self.pos_fourier = PositionalFourier(pos_fourier_L)
        self.in_proj = nn.Linear(in_ch + self.pos_fourier.out_dim, d_model)

        self.t_embed = SinusoidalTimestepEmbedding(cond_dim)
        self.t_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )

        self.patch_size = patch_size  # shared by every block; `forward` builds the
                                       # patch index once per ordering, see there
        self.blocks = nn.ModuleList([
            PTv3Block(d_model, heads, patch_size, mlp_ratio, cond_dim) for _ in range(depth)
        ])
        self.norm_out = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, out_ch)

    def forward(self, x_in: torch.Tensor, pos: torch.Tensor, offset: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        # `offset`/`t` are small bookkeeping tensors that callers naturally
        # build on CPU (`lengths_to_offset` from a python list,
        # `torch.randint` without a device). Indexing a CUDA tensor with a
        # CPU index tensor is a hard error, and it would surface deep inside
        # a block rather than here — so normalize device once, at the entry.
        offset = offset.to(x_in.device)
        t = t.to(x_in.device)

        h = self.in_proj(torch.cat([x_in, self.pos_fourier(pos)], dim=-1))
        emb = self.t_mlp(self.t_embed(t))
        # python list, computed once: used by broadcast_per_object in every
        # block, and keeping it host-side avoids a per-block GPU sync
        lengths = offset_to_lengths(offset).tolist()

        # Pick each block's serialization ordering first (§4c: random per
        # layer during training so patch boundaries differ and information
        # crosses patches; fixed cycling at eval for determinism), then build
        # the patch index once per DISTINCT ordering used.
        #
        # Assignment matches `reference/ptv3-official/model.py` exactly
        # (TRAINING_SPEC §0): the reference permutes the ORDER LIST once per
        # forward (`Point.serialization(shuffle_orders=True)`, line 128) and
        # gives block i `order_index = i % len(order)` (line 905) — a cycle
        # through a shuffled list, NOT an independent draw per block.
        # Independent draws (what this did before) hand consecutive blocks the
        # same ordering 1-in-4 of the time, and a block whose patch boundaries
        # match its predecessor's contributes no cross-patch mixing at all —
        # which §4c calls the thing that makes patches non-isolated. Cycling a
        # permutation guarantees all four orderings appear before any repeats.
        #
        # This is a pure speed change, not a semantic one: `build_patch_index`
        # depends only on (pos, offset, axes, patch_size), and `pos` is frozen
        # conditioning that is never noised — so within one forward pass the
        # index for a given ordering is identical no matter which block asks
        # for it. Computing it per block instead recomputed the same thing up
        # to 12x, and each build costs two GPU->CPU syncs (`.tolist()`) plus a
        # sort. Measured 1242 ms/step at N=16384 before this change.
        #
        # NOT the caching STAGE_SPEC §4a's CORRECTION warns against: that's
        # about reusing an ordering ACROSS samples/steps, where augmentation
        # has moved the positions in between. Nothing is retained past the end
        # of this call.
        if self.training:
            perm = torch.randperm(len(ORDERINGS)).tolist()
        else:
            perm = list(range(len(ORDERINGS)))   # deterministic at eval (§4c)
        block_axes = [ORDERINGS[perm[i % len(ORDERINGS)]] for i in range(len(self.blocks))]

        # Patch size is clamped to the shortest object in the batch, matching
        # the reference's non-flash path (`SerializedAttention.forward`:
        # `patch_size = min(offset2bincount(offset).min(), patch_size_max)`).
        # Without it an object with fewer points than `patch_size` gets padded
        # with many wrap-around copies of itself — at n=100, patch=1024 every
        # point appears ~10 times inside one attention window, which is a
        # silently different computation rather than an error. Never binds in
        # the real config (smallest object has 11,100 points vs patch 1024);
        # it makes reduced-N smoke tests measure the real thing.
        patch_size = min(self.patch_size, min(lengths))
        patch_cache = {}
        for axes in set(block_axes):
            code = morton_code(pos, axes=axes)
            patch_cache[axes] = build_patch_index(offset, code, patch_size)

        for block, axes in zip(self.blocks, block_axes):
            h = block(h, lengths, emb, patch_cache[axes], patch_size)

        return self.out_proj(self.norm_out(h))

    def gate_means(self) -> list:
        """mean(|gate|) per block from the last forward call (LOGGING.md
        panel 2) — `None` entries if `forward` hasn't run yet. One
        batched device->host transfer for all blocks, not one per block
        (see `PTv3Block.last_gate_mean`)."""
        vals = [b.last_gate_mean for b in self.blocks]
        if any(v is None for v in vals):
            return [None if v is None else float(v) for v in vals]
        return torch.stack(vals).cpu().tolist()
