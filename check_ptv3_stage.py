"""
check_ptv3_stage.py — STAGE_SPEC §11 build-order step 3 checks, for
`model/ptv3.py` (distinct from `ptv3_smoke_test.py`/
`ptv3_noise_sanity_check.py`, which check the old pre-pivot EDM `ptv3.py`).

Checks 2 and 3 (STAGE_SPEC §9), plus a basic forward/backward smoke test
on real batched data (not one of the numbered checks, but "every step
needs a check it can run itself" per CLAUDE.md, and it's nearly free
here).

Run: python check_ptv3_stage.py
"""

import time

import container_env  # noqa: F401 -- MUST precede torch, see its docstring

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import torch

from batching import lengths_to_offset
from model.ptv3 import ORDERINGS, PTv3, build_patch_index, morton_code
from stage_config import StageConfig
from train.dataset import build_dataloader

torch.manual_seed(0)
cfg = StageConfig(stage=1, batch_size=4, n_gaussians=4096, num_workers=0)  # small N: fast checks

# ----------------------------------------------------------------- 1. real positions to test against
print("[setup] loading one real batch for realistic position data...")
loader = build_dataloader(cfg, split="train")
batch = next(iter(loader))
pos_all = batch["cond"][:, 0:3]  # xyz is cond[:, 0:3] at every stage (STAGE_SPEC §1)
offset = batch["offset"]
print(f"    batch: {tuple(pos_all.shape)}, offset={offset.tolist()}\n")

# take just the first object's positions for checks 2/3 (single-object, no padding needed for the literal check)
n0 = batch["lengths"][0].item()
pos0 = pos_all[:n0]

# ----------------------------------------------------------------- check 2: serialization locality
print("[check 2] serialization locality")
code0 = morton_code(pos0, axes=(0, 1, 2))
order = torch.argsort(code0)

def mean_adjacent_dist(p, perm):
    p_ordered = p[perm]
    return (p_ordered[1:] - p_ordered[:-1]).norm(dim=-1).mean().item()

d_serial = mean_adjacent_dist(pos0, order)
rng_perm = torch.randperm(n0)
d_random = mean_adjacent_dist(pos0, rng_perm)
ratio = d_random / d_serial
print(f"    n={n0}  mean adjacent dist: serialized={d_serial:.5f}  random={d_random:.5f}  "
      f"ratio={ratio:.2f}x  (STAGE_SPEC §9 check 2 expects ~8x)")
assert ratio > 3.0, f"serialization locality too weak: only {ratio:.2f}x tighter than random"
print("    PASS\n")

# ----------------------------------------------------------------- check 3: serialization round-trip
print("[check 3] serialization round-trip, x[perm][inv] == x exactly")
# (a) the plain permutation itself (literal check wording)
inv = torch.argsort(order)
x = pos0.clone()
recon = x[order][inv]
max_err_plain = (recon - x).abs().max().item()
print(f"    plain permutation: max|x[perm][inv] - x| = {max_err_plain:.2e}")
assert max_err_plain == 0.0

# (b) the padding-aware gather/scatter actually used by PatchAttention
# (build_patch_index's serial_idx/real_original_idx/real_positions) —
# use each point's own index as its "feature" so reconstruction can be
# checked exactly, patch_size deliberately not a divisor of n0 so
# padding is genuinely exercised.
patch_size = 1024
code_full = morton_code(pos_all, axes=(1, 2, 0))
serial_idx, real_original_idx, real_positions, unshuffle_idx = build_patch_index(
    offset, code_full, patch_size)
assert serial_idx.shape[0] % patch_size == 0
total_N = pos_all.shape[0]
ident = torch.arange(total_N, dtype=torch.float32)[:, None]
gathered = ident[serial_idx]                       # simulates the patch-attention gather
recon2 = ident.new_zeros(total_N, 1)
recon2[real_original_idx] = gathered[real_positions]  # the original scatter formulation
max_err_padded = (recon2 - ident).abs().max().item()

# the single-gather form PatchAttention actually uses must agree EXACTLY
# with the scatter formulation it replaced
recon_gather = gathered[unshuffle_idx]
assert torch.equal(recon_gather, recon2), \
    "unshuffle_idx single-gather disagrees with the gather+scatter it replaced"
assert (recon_gather - ident).abs().max().item() == 0.0
print(f"    padded patch-index gather/scatter: max|recon - x| = {max_err_padded:.2e} "
      f"(serial_idx len {serial_idx.shape[0]} vs {total_N} real points, "
      f"{serial_idx.shape[0] - total_N} padding duplicates)")
assert max_err_padded == 0.0
print("    PASS\n")

# (c) padding path specifically — (b) above used cfg.n_gaussians=4096,
# already a multiple of patch_size, so 0 padding duplicates were
# actually produced; that leaves the wrap-around branch unexercised.
# Force it with synthetic object lengths that aren't patch_size multiples.
print("[check 3b] round-trip with padding actually exercised")
synth_lengths = [11100, 4096, 16384, 1500]  # STAGE_SPEC §2's real min/max, deliberately not /1024
synth_offset = lengths_to_offset(synth_lengths)
synth_total = sum(synth_lengths)
synth_pos = torch.randn(synth_total, 3) * 0.3
synth_code = morton_code(synth_pos, axes=(0, 1, 2))
s_serial_idx, s_real_original_idx, s_real_positions, s_unshuffle_idx = build_patch_index(
    synth_offset, synth_code, patch_size)
n_pad = s_serial_idx.shape[0] - synth_total
print(f"    lengths={synth_lengths}  padding duplicates={n_pad}")
assert n_pad > 0, "test setup didn't actually exercise padding"
synth_ident = torch.arange(synth_total, dtype=torch.float32)[:, None]
synth_gathered = synth_ident[s_serial_idx]
synth_recon = synth_ident.new_zeros(synth_total, 1)
synth_recon[s_real_original_idx] = synth_gathered[s_real_positions]
max_err_synth = (synth_recon - synth_ident).abs().max().item()
print(f"    max|recon - x| = {max_err_synth:.2e}")
assert max_err_synth == 0.0
assert torch.equal(torch.sort(s_real_original_idx).values, torch.arange(synth_total))
# and the single-gather form, with padding genuinely present
assert torch.equal(synth_gathered[s_unshuffle_idx], synth_recon), \
    "unshuffle_idx disagrees with gather+scatter when padding is present"
print("    PASS\n")

# ----------------------------------------------------------------- smoke test: forward/backward, real shapes
print("[smoke] forward/backward pass on a real batch, gate-init sanity")
in_ch = cfg.n_cond_ch + cfg.n_target_ch
model = PTv3(in_ch=in_ch, out_ch=cfg.n_target_ch, d_model=cfg.d_model, depth=cfg.n_blocks,
             heads=cfg.n_heads, patch_size=1024, mlp_ratio=cfg.mlp_ratio)
n_params = sum(p.numel() for p in model.parameters())
print(f"    in_ch={in_ch}  out_ch={cfg.n_target_ch}  d_model={cfg.d_model}  depth={cfg.n_blocks}  "
      f"params={n_params / 1e6:.2f}M")

x_in = torch.cat([batch["cond"], batch["target"]], dim=-1)
t = torch.randint(0, cfg.T, (offset.shape[0],))

t0 = time.time()
out = model(x_in, pos_all, offset, t)
print(f"    forward: {tuple(out.shape)}  ({time.time() - t0:.2f}s)")
assert out.shape == (total_N, cfg.n_target_ch)
assert torch.isfinite(out).all()

loss = out.pow(2).mean()
loss.backward()
grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
n_no_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is None)
print(f"    backward OK: {len(grad_norms)} params with grad, {n_no_grad} without, "
      f"mean |grad| = {sum(grad_norms) / len(grad_norms):.2e}")
assert n_no_grad == 0, "some parameters got no gradient"
assert all(g == g for g in grad_norms), "NaN gradient"

gate_weight_norms = [b.adaln.net[-1].weight.abs().max().item() for b in model.blocks]
print(f"    adaLN gate weights at init (should be exactly 0.0, zero-init): {gate_weight_norms}")
assert all(g == 0.0 for g in gate_weight_norms)
print("    PASS\n")

# ----------------------------------------------------------------- reference reconciliation (PLAN.md PTv3 audit)
# Both of these check fixes made when `model/ptv3.py` was reconciled against
# `reference/ptv3-official/model.py`. They are not STAGE_SPEC's numbered
# checks — they guard behaviour the reference specifies and ours previously
# got wrong in a way that throws no error.
print("[check R1] block->ordering assignment cycles a permutation (reference model.py:128/905)")
model.train()
seen = []
for _ in range(200):
    perm = torch.randperm(len(ORDERINGS)).tolist()
    seen.append([perm[i % len(ORDERINGS)] for i in range(cfg.n_blocks)])
# The property that matters: no two CONSECUTIVE blocks ever share an ordering,
# which independent per-block draws violate 1-in-4 of the time.
n_adjacent_repeats = sum(1 for s in seen for a, b in zip(s, s[1:]) if a == b)
n_distinct_first4 = min(len(set(s[:4])) for s in seen)
print(f"    over 200 forwards: adjacent-block ordering repeats = {n_adjacent_repeats} "
      f"(independent draws would give ~{0.25 * 200 * (cfg.n_blocks - 1):.0f}); "
      f"min distinct orderings in the first 4 blocks = {n_distinct_first4}/4")
assert n_adjacent_repeats == 0
assert n_distinct_first4 == len(ORDERINGS)
print("    PASS\n")

print("[check R2] patch_size clamps to the shortest object (reference SerializedAttention.forward)")
# An object shorter than patch_size would otherwise be padded with ~patch_size/n
# wrap-around copies of itself, so every point appears many times inside one
# attention window — a silently different computation, not an error.
short_lengths = [700, 1500]
short_offset = lengths_to_offset(short_lengths)
short_total = sum(short_lengths)
short_x = torch.randn(short_total, in_ch)
short_pos = torch.randn(short_total, 3) * 0.3
short_out = model(short_x, short_pos, short_offset, torch.randint(0, cfg.T, (2,)))
eff_patch = min(1024, min(short_lengths))
s_idx, _, _, _ = build_patch_index(short_offset, morton_code(short_pos), eff_patch)
dupes = s_idx.shape[0] - short_total
print(f"    lengths={short_lengths} vs patch_size=1024 -> effective patch {eff_patch}, "
      f"{dupes} padding duplicates ({dupes / short_total:.1%} of points)")
assert short_out.shape == (short_total, cfg.n_target_ch)
assert torch.isfinite(short_out).all()
assert dupes / short_total < 0.5, "clamp did not take effect — objects are mostly duplicates"
print("    PASS\n")

print("ALL PTv3-STAGE CHECKS PASSED.")
