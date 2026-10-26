"""
bench_model.py — where does a training step actually spend its time?

`bench_attention.py` settled that attention is NOT the bottleneck: 8.9 ms
per call x 12 blocks = ~107 ms of a ~2361 ms step (4.5%). ~1900 ms is
unaccounted for, and three attempts to locate it by reasoning about FLOPs
and memory were all wrong. So: profile the real thing.

Runs the actual Denoiser fwd+bwd at the benchmark config and reports the
CUDA kernels that dominate, plus a section-level breakdown of the parts
that aren't matmuls (serialization, gather/scatter, positional features)
-- the code we wrote, as opposed to the code torch wrote.

Run on the GPU box: python bench_model.py [--n-gaussians 32768] [--amp bf16]
"""

import argparse

import container_env  # noqa: F401 -- MUST precede torch

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from batching import lengths_to_offset, offset_to_batch_idx
from diffusion.schedule import add_noise, l_simple, make_train_scheduler
from model.denoiser import Denoiser
from model.ptv3 import ORDERINGS, build_patch_index, morton_code
from stage_config import StageConfig

ap = argparse.ArgumentParser()
ap.add_argument("--n-gaussians", type=int, default=32768)
ap.add_argument("--amp", choices=["none", "bf16"], default="none")
ap.add_argument("--iters", type=int, default=10)
ap.add_argument("--rows", type=int, default=18)
args = ap.parse_args()

if not torch.cuda.is_available():
    raise SystemExit("needs CUDA")
device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Report whether TF32 actually took effect -- the allow_tf32 flags are
# deprecated in newer torch in favour of the fp32_precision API, and if they
# are silently ignored we would be running full fp32 matmuls without knowing.
print(f"torch {torch.__version__}, {torch.cuda.get_device_name(device)}")
print(f"  matmul.allow_tf32     = {torch.backends.cuda.matmul.allow_tf32}")
for attr in ("fp32_precision",):
    if hasattr(torch.backends.cuda.matmul, attr):
        print(f"  matmul.{attr} = {getattr(torch.backends.cuda.matmul, attr)}")
if hasattr(torch, "get_float32_matmul_precision"):
    print(f"  float32_matmul_precision = {torch.get_float32_matmul_precision()}")
print(f"  N={args.n_gaussians}, amp={args.amp}\n")

torch.manual_seed(0)
cfg = StageConfig(stage=1, batch_size=1, n_gaussians=args.n_gaussians)
n = args.n_gaussians
model = Denoiser(cfg).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = make_train_scheduler(T=cfg.T)

cond = torch.randn(n, cfg.n_cond_ch, device=device)
cond[:, 0:3] = torch.randn(n, 3, device=device) * 0.3   # plausible xyz for serialization
target = torch.randn(n, cfg.n_target_ch, device=device)
offset = lengths_to_offset([n]).to(device)
batch_idx = offset_to_batch_idx(offset)

amp_ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) \
    if args.amp == "bf16" else torch.enable_grad


def step():
    t = torch.randint(0, cfg.T, (1,), device=device)
    noise = torch.randn_like(target)
    y_t = add_noise(scheduler, target, noise, t, batch_idx)
    with amp_ctx():
        out = model(cond, y_t, t, offset)
    # Plain L_simple on the mean half, NOT the real hybrid loss: this script
    # measures where the training STEP's time goes, and the hybrid loss adds
    # a handful of elementwise ops on the output that would only dilute the
    # backbone timing this exists to profile. Keeping it L_simple also keeps
    # the ms/step numbers comparable against the ones in PLAN.md's perf table.
    pred = model.split_output(out)[0]
    loss = l_simple(pred, target)
    opt.zero_grad()
    loss.backward()
    opt.step()


for _ in range(3):  # warmup: autotune, allocator, lazy kernel load
    step()
torch.cuda.synchronize()

# ---------------------------------------------------------------- whole step
start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
start.record()
for _ in range(args.iters):
    step()
end.record()
torch.cuda.synchronize()
total_ms = start.elapsed_time(end) / args.iters
peak_alloc = torch.cuda.max_memory_allocated() / 1e9
peak_res = torch.cuda.max_memory_reserved() / 1e9
print(f"=== full training step: {total_ms:.0f} ms   "
      f"peak GPU {peak_alloc:.2f} GB allocated / {peak_res:.2f} GB reserved ===\n")

# ---------------------------------------------------------------- our own non-matmul code
# Timed separately because it's what we wrote, and it's the part a FLOP
# estimate would never account for.
pos = cond[:, 0:3]


def timeit(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


code = morton_code(pos, axes=ORDERINGS[0])
print("--- our non-matmul code, per call ---")
print(f"  morton_code                 {timeit(lambda: morton_code(pos, axes=ORDERINGS[0])):7.2f} ms"
      f"   (x4 per fwd)")
print(f"  build_patch_index           "
      f"{timeit(lambda: build_patch_index(offset, code, cfg.patch_size)):7.2f} ms   (x4 per fwd)")
print(f"  pos_fourier                 "
      f"{timeit(lambda: model.backbone.pos_fourier(pos)):7.2f} ms   (x1 per fwd)")
si, roi, rp, unsh = build_patch_index(offset, code, cfg.patch_size)
h = torch.randn(n, cfg.d_model, device=device)
print(f"  gather x[serial_idx]        {timeit(lambda: h[si]):7.2f} ms   (x12 per fwd)")
padded = h.new_zeros(si.shape[0], cfg.d_model)
print(f"  unshuffle out[unshuffle_idx]{timeit(lambda: padded[unsh]):7.2f} ms   (x12 per fwd, "
      f"replaced a gather+scatter pair)\n")

# ---------------------------------------------------------------- kernel profile
print(f"=== top {args.rows} CUDA kernels by self time (1 step) ===")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=False) as prof:
    step()
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=args.rows))
