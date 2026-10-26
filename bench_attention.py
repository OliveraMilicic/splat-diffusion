"""
bench_attention.py — which SDPA backend does `PatchAttention` actually get,
and what does each one cost?

Written because the same question was answered twice by inference (from
memory arithmetic) and got the wrong answer both times. This measures.

Reproduces `model/ptv3.py::PatchAttention`'s exact q/k/v construction —
same shapes, same `.view()` + `.permute()` chain, hence the same strides,
which is the thing most likely to disqualify a fast kernel — then for each
dtype:

  * asks `torch.backends.cuda.can_use_flash_attention` whether flash is
    even eligible, and prints WHY NOT if it isn't (that's the diagnosis)
  * forces each backend in turn via `sdpa_kernel` and times fwd+bwd,
    reporting peak memory, so "which is fastest" is measured not assumed
  * repeats with contiguous q/k/v, to test whether the permute's strides
    are what's blocking the fast path

Run on the GPU box: python bench_attention.py
"""

import container_env  # noqa: F401 -- MUST precede torch

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

N_POINTS = 32768        # matches the benchmark runs
PATCH = 1024
D_MODEL = 384
HEADS = 6
HEAD_DIM = D_MODEL // HEADS
ITERS = 20


def make_qkv(dtype, device, contiguous: bool):
    """Exactly PatchAttention's construction (model/ptv3.py)."""
    n_patches = N_POINTS // PATCH
    x_s = torch.randn(n_patches, PATCH, D_MODEL, device=device, dtype=dtype)
    qkv_proj = nn.Linear(D_MODEL, D_MODEL * 3, device=device, dtype=dtype)
    qkv = qkv_proj(x_s).view(n_patches, PATCH, 3, HEADS, HEAD_DIM)
    qkv = qkv.permute(2, 0, 3, 1, 4)          # (3, n_patches, heads, P, head_dim)
    q, k, v = qkv[0], qkv[1], qkv[2]
    if contiguous:
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    return (q.detach().requires_grad_(True),
            k.detach().requires_grad_(True),
            v.detach().requires_grad_(True))


def time_backend(q, k, v, backend, label):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    try:
        ctx = sdpa_kernel(backend) if backend is not None else torch.enable_grad()
        with ctx:
            for _ in range(3):  # warmup
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                out.sum().backward()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(ITERS):
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                out.sum().backward()
            end.record()
            torch.cuda.synchronize()
        ms = start.elapsed_time(end) / ITERS
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"    {label:<22} {ms:8.2f} ms   peak {peak:5.2f} GB")
        return ms
    except Exception as e:
        msg = str(e).split("\n")[0][:95]
        print(f"    {label:<22} UNAVAILABLE: {msg}")
        return None


def main():
    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")
    device = torch.device("cuda")
    print(f"{torch.cuda.get_device_name(device)}, torch {torch.__version__}")
    print(f"shape per patch-attention call: "
          f"({N_POINTS // PATCH}, {HEADS}, {PATCH}, {HEAD_DIM})  "
          f"[n_patches, heads, patch, head_dim]\n")

    for dtype in (torch.float32, torch.bfloat16):
        for contiguous in (False, True):
            tag = f"{str(dtype).replace('torch.', '')}, " \
                  f"{'contiguous' if contiguous else 'permuted (as in PatchAttention)'}"
            print(f"=== {tag} ===")
            q, k, v = make_qkv(dtype, device, contiguous)

            # THE diagnosis: is flash eligible at all, and if not, why not?
            params = torch.backends.cuda.SDPAParams(q, k, v, None, 0.0, False, False)
            flash_ok = torch.backends.cuda.can_use_flash_attention(params, debug=True)
            print(f"    can_use_flash_attention: {flash_ok}"
                  f"{'   (reason printed above by torch)' if not flash_ok else ''}")

            time_backend(q, k, v, None, "default dispatch")
            time_backend(q, k, v, SDPBackend.FLASH_ATTENTION, "forced FLASH")
            time_backend(q, k, v, SDPBackend.EFFICIENT_ATTENTION, "forced MEM_EFFICIENT")
            time_backend(q, k, v, SDPBackend.CUDNN_ATTENTION, "forced CUDNN")
            time_backend(q, k, v, SDPBackend.MATH, "forced MATH")
            print()
            del q, k, v
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
