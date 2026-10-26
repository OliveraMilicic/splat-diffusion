"""
train/overfit.py — STAGE_SPEC §11 build-order step 6. HARD GATE: do not
proceed past this point until it passes (STAGE_SPEC §11 / CLAUDE.md's
"day-2 overfit test" framing, carried over to this pivot).

Batch 1, no augmentation, no EMA (STAGE_SPEC §11, §5), ~2000 steps,
logging every step (LOGGING.md "Overfit-run specifics"). Checks 7, 8, 9.

`--n-gaussians` defaults to the production **16384** (STAGE_SPEC §3:
"do the final overfit check at the real N so the gate tests the real
configuration"). Objects with fewer Gaussians than that keep all of
them and are simply shorter — no dataset-level padding anywhere
(STAGE_SPEC §10). The only padding in the system is internal to patch
attention, which pads each object's serialized sequence up to a
multiple of `patch_size` with wrap-around duplicates of that object's
own points; that's required for fixed-size attention patches and is
verified exact by check 3.

STAGE_SPEC §3 does allow a smaller N for speed while iterating — a
CPU-only N=2048 run passed all three checks earlier — but that is
explicitly not the real gate.

Run: python train/overfit.py [--n-gaussians 16384] [--steps 2000]
"""

import argparse
import contextlib
import dataclasses
import json
import sys
import time
from pathlib import Path

# Direct script execution (`python train/overfit.py`) sets sys.path[0] to
# this file's own directory, not repo root, and never triggers
# train/__init__.py (that only runs on `import train.*`) -- same fix as
# eval/decode.py.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import container_env  # noqa: F401 -- MUST precede torch, see its docstring

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import matplotlib
matplotlib.use("Agg")  # CLAUDE.md: no display on this node
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from batching import lengths_to_offset, offset_to_batch_idx
from diffusion.losses import DiffusionLoss
from diffusion.render_loss import RenderLoss
from diffusion.sampling import sample as diffusion_sample
from diffusion.schedule import (add_noise, dynamic_clip, l_simple,
                                 make_train_scheduler)
from eval.render_grid import (active_backend, geometry_only_tokens, render_tokens,
                               to_tokens)
from model.denoiser import Denoiser
from preprocessing import full_object_tokens, standardize
from splat_utils import find_ply_files, read_splat
from stage_config import (StageConfig, load_norm_stats, load_split, lr_at,
                           split_channels)
from train.logging_utils import (StageLogger, TimestepBinTracker,
                                  per_term_grad_norms)

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"

# `to_splat` (the recombine -> unstandardize -> tokens_to_splat_dict decode
# chain) and `geometry_only` moved to eval/render_grid.py so this gate script
# and train/loop.py decode predictions identically -- a divergence there would
# make "does the full run look as good as the gate?" unanswerable by eye.
# The panel LAYOUTS stay separate on purpose: this script renders 3 seeds for
# check 9, the training loop renders 2 seeds across TRAIN and VAL rows.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=3,
                     help="3 = the current build target (log-scale + opacity + RGB)")
    ap.add_argument("--no-learn-sigma", action="store_true",
                     help="plain L_simple instead of GaussianCube's hybrid loss")
    # L_image (diffusion/render_loss.py). ON here too, so the gate tests the
    # loss the full run will actually use -- a gate that passes on a different
    # objective than the run is not a gate.
    ap.add_argument("--no-render-loss", action="store_true",
                     help="drop L_image and gate on the diffusion loss alone")
    ap.add_argument("--render-l1-weight", type=float, default=10.0)
    ap.add_argument("--render-lpips-weight", type=float, default=10.0)
    ap.add_argument("--render-loss-start", type=int, default=None,
                     help="step at which L_image ramps in. Default: 25%% of --steps. "
                          "train/loop.py's 10,000 would never fire in a 2,000-step "
                          "overfit, so it is scaled to the run instead of copied.")
    ap.add_argument("--render-loss-ramp", type=int, default=None,
                     help="default: 10%% of --steps")
    ap.add_argument("--render-loss-views", type=int, default=1)
    ap.add_argument("--render-loss-size", type=int, default=128)
    ap.add_argument("--clip-input-p", type=float, default=None,
                     help="GaussianCube's dynamic_clip: per-object per-channel "
                          "symmetric clip of the standardized TARGET at this "
                          "quantile of |value|. THEY PASS --clip_input for "
                          "ShapeNet, i.e. 0.995. Off here (0.0) on a measurement "
                          "-- the standardized stage-3 target's worst tail is "
                          "6.7 sigma, so 0.995 would clip every object at ~3.7 "
                          "sigma and shrink its largest Gaussians. Set 0.995 to "
                          "match them exactly.")
    ap.add_argument("--render-pixel-norm", choices=["l1", "l2"], default="l1",
                     help="pixel term of L_image. 'l1' = GaussianCube's CODE "
                          "(nn.L1Loss, train.py:285) and what produced their "
                          "published numbers; 'l2' = their PAPER eq. (4), which "
                          "writes ||I_pred - I_gt||_2. They disagree; the code wins "
                          "by default.")
    ap.add_argument("--grad-probe-every", type=int, default=200,
                     help="measure each loss term's gradient norm separately every "
                          "N steps (0 = off). One extra backward per term, so ~1%% "
                          "overhead at the default cadence. This is what answers "
                          "'is L2 dominating?' -- loss values cannot.")
    ap.add_argument("--n-gaussians", type=int, default=16384,
                     help="production N (STAGE_SPEC §3). Objects with fewer keep all of theirs.")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=None,
                     help="linear LR warmup. Default: 5%% of --steps capped at "
                          "GaussianCube's 1000 (train.py:135). Their 1000 is 0.1%% "
                          "of a 1M-step run; applied literally to a 2000-step "
                          "overfit it would be half the run below target LR.")
    ap.add_argument("--object", type=str, default=None,
                     help="object name; default: first train object(s) of the split")
    ap.add_argument("--n-objects", type=int, default=1,
                     help="how many objects to overfit jointly, as ONE batch (STAGE_SPEC §9 "
                          "check 7 is 1 object; >1 additionally exercises the B>1 "
                          "broadcast path, real variable-N batching, and makes check 9 "
                          "load-bearing since the model can no longer memorise one answer)")
    ap.add_argument("--run-name", type=str, default="overfit1_stage3")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--print-every", type=int, default=100)
    ap.add_argument("--device", type=str, default=None,
                     help="cuda / cpu / cuda:0. Default: cuda when available, else cpu.")
    ap.add_argument("--sample-steps", type=int, default=50, help="sampler steps for checks 8/9")
    ap.add_argument("--no-tf32", action="store_true", help="strict fp32 matmuls on CUDA")
    ap.add_argument("--skip-checks", action="store_true",
                     help="train only: skip sampling + renders. For clean perf benchmarking, "
                          "where the 3x50 sampler forwards would otherwise dominate a short run.")
    ap.add_argument("--amp", choices=["none", "bf16"], default="none",
                     help="bf16 autocast for the training forward/loss. Master weights stay "
                          "fp32 and no GradScaler is needed (bf16 has fp32's exponent range). "
                          "Sampling deliberately stays fp32. STAGE_SPEC §3's memory budget "
                          "assumes bf16; 'none' is the default so the gate stays fp32.")
    args = ap.parse_args()

    torch.manual_seed(0)
    cfg = StageConfig(stage=args.stage, batch_size=args.n_objects,
                       n_gaussians=args.n_gaussians, run_name=args.run_name, lr=args.lr,
                       learn_sigma=not args.no_learn_sigma)
    if args.clip_input_p is not None:
        cfg.clip_input_p = args.clip_input_p
    # GaussianCube's warmup is 1000 steps of a ~1,000,000-step run (0.1%), and
    # StageConfig carries that absolute number for the 60k full run (1.7%).
    # Applied literally to a 2,000-step overfit it would be HALF the run spent
    # below target LR -- measured: at step 500 the LR would still be 5e-5, and
    # the loss barely moves. So scale it to the run and cap at their value.
    cfg.warmup_steps = (args.warmup_steps if args.warmup_steps is not None
                        else max(1, min(args.steps // 20, cfg.warmup_steps)))
    print(f"LR: {cfg.lr:g} with {cfg.warmup_steps}-step linear warmup "
          f"({cfg.warmup_steps / args.steps:.0%} of the run)")

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"device: {device} — {props.name}, {props.total_memory / 1e9:.1f} GB")
        if not args.no_tf32:
            # A100 tensor cores. Standard for training (this is what
            # `torch.set_float32_matmul_precision('high')` selects); ~5x on
            # fp32 matmuls, at reduced mantissa precision in the matmul only.
            # --no-tf32 restores strict fp32 if a numerical check ever needs it.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("        TF32 matmuls enabled (--no-tf32 to disable)")
    else:
        print(f"device: {device}  (no CUDA visible — this will be slow at N={args.n_gaussians})")

    norm_stats = load_norm_stats(cfg.shard_path)
    split = load_split(cfg.shard_path)
    up_axis = norm_stats["up_axis"]
    files = find_ply_files(cfg.data_dir)
    name_to_file = {f.stem: f for f in files}

    if args.object:
        obj_names = [args.object]
        if args.n_objects > 1:
            raise SystemExit("--object names a single object; drop it to use --n-objects>1")
    else:
        obj_names = split["train"][: args.n_objects]
    print(f"overfitting on {len(obj_names)} object(s): {', '.join(o[:26] for o in obj_names)}")

    rng = np.random.default_rng(cfg.seed)
    conds_np, targets_np, lengths, metas = [], [], [], []
    for name in obj_names:
        splat = read_splat(name_to_file[name])
        tok_raw_i, centre_i, radius_i = full_object_tokens(splat)
        n_total_i = tok_raw_i.shape[0]
        n_i = min(cfg.n_gaussians, n_total_i)
        if n_i < n_total_i:
            tok_raw_i = tok_raw_i[rng.choice(n_total_i, n_i, replace=False)]
            note = f"random subsample of {n_total_i}"
        else:
            # STAGE_SPEC §3/§10: shorter objects stay short, no padding to a fixed N
            note = f"kept all {n_total_i}, fewer than {cfg.n_gaussians}, no padding"
        c_i, t_i = split_channels(standardize(tok_raw_i, norm_stats), cfg)
        # This script assembles its batch directly rather than going through
        # ChairPointDataset, so the dataset's clip has to be applied here too
        # -- otherwise --clip-input-p would appear to work and silently do
        # nothing, and the gate would train on different data than the run.
        if cfg.clip_input_p:
            t_i = dynamic_clip(torch.from_numpy(np.ascontiguousarray(t_i)),
                               cfg.clip_input_p).numpy()
        conds_np.append(c_i)
        targets_np.append(t_i)
        lengths.append(n_i)
        metas.append((name, centre_i, radius_i))
        print(f"    {name[:30]}: N={n_i} ({note})")

    if len(set(lengths)) > 1:
        print(f"  -> batch genuinely mixes lengths {lengths}: exercises variable-N batching")

    # single concatenated batch, the STAGE_SPEC §4 sparse-tensor convention
    cond_np = np.concatenate(conds_np, axis=0)
    target_np = np.concatenate(targets_np, axis=0)
    n = int(sum(lengths))
    cond = torch.from_numpy(np.ascontiguousarray(cond_np, dtype=np.float32)).to(device)
    target = torch.from_numpy(np.ascontiguousarray(target_np, dtype=np.float32)).to(device)
    offset = lengths_to_offset(lengths).to(device)
    batch_idx = offset_to_batch_idx(offset)
    obj_name = obj_names[0] if len(obj_names) == 1 else f"{len(obj_names)}objs"
    n_total = n

    model = Denoiser(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params / 1e6:.2f}M  (depth={cfg.n_blocks}, d_model={cfg.d_model})")

    # no EMA (STAGE_SPEC §11/§5: "Disable for overfit")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = make_train_scheduler(T=cfg.T, learn_sigma=cfg.learn_sigma)
    loss_fn = DiffusionLoss(scheduler, learn_sigma=cfg.learn_sigma,
                             min_snr_gamma=cfg.min_snr_gamma, device=device)
    render_loss, rl_start, rl_ramp = None, 0, 1
    if not args.no_render_loss:
        rl_start = args.render_loss_start if args.render_loss_start is not None \
            else max(1, args.steps // 4)
        rl_ramp = args.render_loss_ramp if args.render_loss_ramp is not None \
            else max(1, args.steps // 10)
        render_loss = RenderLoss(
            cfg, norm_stats, device, image_size=args.render_loss_size,
            n_views=args.render_loss_views, l1_weight=args.render_l1_weight,
            lpips_weight=args.render_lpips_weight,
            start_step=rl_start, ramp_steps=rl_ramp, seed=cfg.seed,
            pixel_norm=args.render_pixel_norm)
        print(f"L_image ON: {args.render_l1_weight:g}*L1 + "
              f"{args.render_lpips_weight:g}*LPIPS, {args.render_loss_views} view(s) "
              f"@ {args.render_loss_size}px, backend={render_loss.backend}, "
              f"ramping in over steps {rl_start}-{rl_start + rl_ramp}")
    else:
        print("L_image OFF (--no-render-loss)")

    print(f"stage {cfg.stage}: diffusing {cfg.n_target_ch} channels "
          f"({', '.join(cfg.target_names)}); loss = GaussianCube hybrid"
          f"{' (MSE + VB, learned variance)' if cfg.learn_sigma else ' DISABLED -> plain L_simple'}")

    logger = StageLogger(cfg, run_name=cfg.run_name, use_wandb=not args.no_wandb)
    logger.log_summary(
        n_params=n_params, n_gaussians_used=n, n_gaussians_total=n_total,
        object_name=obj_name, spconv_available=False,
        n_train_objects=len(split["train"]), device=str(device),
        gpu_available=torch.cuda.is_available(), amp=args.amp,
    )

    t_bins = TimestepBinTracker(n_bins=10, T=cfg.T)

    if args.amp == "bf16":
        if device.type != "cuda":
            raise SystemExit("--amp bf16 needs CUDA")
        if not torch.cuda.is_bf16_supported():
            raise SystemExit("this GPU does not support bf16")
        def amp_ctx():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        print("        bf16 autocast ON for the training forward "
              "(master weights fp32; sampling stays fp32)")
    else:
        amp_ctx = contextlib.nullcontext

    # ------------------------------------------------- L_image feasibility probe
    # L_image only fires once the ramp opens (25% of the run by default), so
    # anything wrong with the render path -- a shape gsplat rejects, a
    # disconnected graph, a config that does not fit -- surfaces hundreds of
    # steps in, AFTER a ~2 minute first-call gsplat JIT compile. That is
    # exactly how the gsplat `backgrounds` shape bug was found, at step 500 of
    # a real run. One real render-loss forward+backward here moves that to the
    # first seconds. Same probe train/loop.py runs; it should have been in both
    # from the start.
    if render_loss is not None:
        try:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            # Deliberately a WRONG prediction: with a perfect one the L1
            # subgradient at 0 is 0, so a fully disconnected loss would give
            # the same all-zero gradient and pass.
            p_pred = (target + 0.5 * torch.randn_like(target)).requires_grad_(True)
            probe_terms = render_loss(p_pred, cond, target, lengths,
                                       step=rl_start + rl_ramp)
            probe_terms["render_total"].backward()
            g = p_pred.grad
            assert g is not None and torch.isfinite(g).all(), \
                "L_image produced no finite gradient — see check_render_loss.py"
            dead = [n for n, v in zip(cfg.target_names, g.abs().mean(0).tolist())
                    if v == 0.0]
            assert not dead, (
                f"L_image gives ZERO gradient to {dead} — it cannot teach those "
                f"channels. Run check_render_loss.py.")
            mem = (f", peak {torch.cuda.max_memory_allocated(device) / 1e9:.1f} GB"
                   if device.type == "cuda" else "")
            print(f"  L_image probe OK: on a deliberately wrong prediction, "
                  f"L1={probe_terms['render_l1'].item():.4f} "
                  f"LPIPS={probe_terms['render_lpips'].item():.4f}, gradient "
                  f"reaches all {len(cfg.target_names)} generated channels{mem}")
        except torch.OutOfMemoryError:
            raise SystemExit(
                f"L_image does not fit at N={cfg.n_gaussians}. Lower "
                f"--render-loss-size (currently {args.render_loss_size}) or pass "
                f"--no-render-loss.")
        finally:
            p_pred = probe_terms = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

    WARMUP_STEPS = min(5, max(1, args.steps // 10))
    warmup_done_t = None

    model.train()
    t0 = time.time()
    losses, mse_losses = [], []
    render_l1s, render_lpips_s = [], []
    B = len(lengths)
    for step in range(args.steps):
        # Linear LR warmup, same as train/loop.py and same as GaussianCube
        # (`LambdaLR` over `num_warmup_steps = 1000`, train.py:135). This was
        # missing here: the gate ran at full LR from step 0 while the full run
        # warmed up, so the gate was testing a different optimizer schedule
        # than the run it is supposed to gate.
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        # one INDEPENDENT timestep per object -- that's the real training
        # condition (diffusion noise level is per-object, not per-batch), and
        # it's what add_noise's batch_idx broadcast exists for
        t = torch.randint(0, cfg.T, (B,), device=device)
        noise = torch.randn_like(target)
        y_t = add_noise(scheduler, target, noise, t, batch_idx)

        # Forward + loss under autocast; backward outside it (standard: the
        # backward reuses the dtypes autocast chose on the forward). F.mse_loss
        # is on autocast's fp32 list, so the loss itself is still accumulated
        # in fp32 even when the network runs bf16.
        with amp_ctx():
            pred = model(cond, y_t, t, offset)
        # Loss outside autocast: the VB term's exp(logvar1-logvar2) needs more
        # than bf16's ~3 decimal digits. See train/loop.py for the same note.
        terms = loss_fn(pred.float(), target, y_t, t, lengths)
        loss = terms["loss"]

        pred_y0_mean = model.split_output(pred.float())[0]
        terms_render = None
        if render_loss is not None:
            terms_render = render_loss(pred_y0_mean, cond, target, lengths, step)
            loss = loss + terms_render["render_total"]

        # Which term is actually driving training? Loss VALUES cannot answer
        # that -- a small term with a steep slope dominates a large flat one,
        # and clip_grad_norm_ rescales the SUM, so the term with the bigger
        # gradient gets proportionally more say whatever the printed losses
        # look like. Measured every --grad-probe-every steps: one extra
        # backward per term, so ~1% overhead at the default cadence.
        grad_terms = {}
        if args.grad_probe_every and step % args.grad_probe_every == 0:
            probe_src = dict(terms)
            if terms_render is not None:
                probe_src["render_total"] = terms_render["render_total"]
            grad_terms = per_term_grad_norms(probe_src, model.parameters())

        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        loss_val = loss.item()
        mse_val = terms["mse"].item()
        losses.append(loss_val)
        mse_losses.append(mse_val)
        render_l1s.append(terms_render["render_l1"].item() if terms_render else 0.0)
        render_lpips_s.append(terms_render["render_lpips"].item() if terms_render else 0.0)
        # Bin the MSE term, not the total: the VB term's t=0 decoder NLL is
        # tens of bits and lands entirely in bin 0, swamping exactly the
        # low-t-vs-high-t comparison LOGGING.md panel 1 exists for.
        for t_i in t.tolist():   # one bucket update per object's own timestep
            t_bins.update(t_i, mse_val)

        # Mean half only — the variance half is not a prediction of the target.
        per_channel = ((pred_y0_mean.detach() - target) ** 2).mean(dim=0).tolist()
        gate_means = [g for g in model.gate_means() if g is not None]

        metrics = dict(
            loss=loss_val,
            loss_mse=mse_val,
            loss_vb=terms["vb"].item() if "vb" in terms else 0.0,
            render_l1=terms_render["render_l1"].item() if terms_render else 0.0,
            render_lpips=terms_render["render_lpips"].item() if terms_render else 0.0,
            render_weight=terms_render["render_weight"] if terms_render else 0.0,
            grad_norm=float(grad_norm),
            lr=opt.param_groups[0]["lr"],
            step_time=(time.time() - t0) / (step + 1),
            gate_mean=float(np.mean(gate_means)) if gate_means else 0.0,
            n_points=n,
        )
        for c, ch_name in enumerate(cfg.target_names):
            metrics[f"loss_{ch_name}"] = per_channel[c]
        metrics.update(grad_terms)
        if device.type == "cuda":
            # allocated = live tensors; reserved = what the caching allocator
            # holds from the driver, i.e. roughly what nvidia-smi shows and
            # what actually has to fit in the card.
            metrics["gpu_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
            metrics["gpu_mem_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 1e9
        for b, m in enumerate(t_bins.means()):
            if not np.isnan(m):
                metrics[f"loss_t_bin_{b}"] = float(m)
        logger.log(step, **metrics)  # every step, LOGGING.md "Overfit-run specifics"

        # steady-state ms/step, excluding the first few steps (cuDNN autotune,
        # allocator warmup and lazy kernel loading make step 1 badly
        # unrepresentative -- 1507ms vs 1057ms steady, measured)
        if step == WARMUP_STEPS - 1:
            warmup_done_t = time.time()

        if (step + 1) % args.print_every == 0 or step == 0 or step == args.steps - 1:
            mem = (f"  mem={metrics['gpu_mem_gb']:.1f}/{metrics['gpu_mem_reserved_gb']:.1f}GB"
                   if device.type == "cuda" else "")
            print(f"  step {step + 1:5d}/{args.steps}  loss={loss_val:.4f} "
                  f"(mse={mse_val:.4f})  "
                  f"grad_norm={grad_norm:.3f}  gate_mean={metrics['gate_mean']:.2e}"
                  f"  {metrics['step_time'] * 1000:.0f}ms/step{mem}")

    dt = time.time() - t0
    print(f"\ntraining done: {args.steps} steps in {dt:.0f}s ({dt / args.steps * 1000:.0f} ms/step)")
    # Check 7's number is the MSE term, not the hybrid total. Two reasons,
    # both about comparability: the VB term is in BITS and the MSE is in
    # squared standardized units, so their sum has no unit; and every gate
    # figure recorded in PLAN.md (check 7 = 0.00033 at the 16k gate) is an
    # L_simple number, which stays comparable only if this one is too.
    final_loss = float(np.mean(mse_losses[-20:]))
    final_total = float(np.mean(losses[-20:]))
    print(f"final L_simple (mean of last 20 steps) = {final_loss:.4f}   "
          f"(hybrid total incl. VB = {final_total:.4f})")

    # ----------------------------------------------------------------- perf / feasibility report
    steady_ms = None
    if warmup_done_t is not None and args.steps > WARMUP_STEPS:
        steady_ms = (time.time() - warmup_done_t) / (args.steps - WARMUP_STEPS) * 1000
    print(f"\n--- perf at N={n}, batch {cfg.batch_size}, amp={args.amp} ---")
    print(f"  ms/step (all steps)        : {dt / args.steps * 1000:.0f}")
    if steady_ms:
        print(f"  ms/step (steady, skip {WARMUP_STEPS:>2}) : {steady_ms:.0f}")
    if device.type == "cuda":
        alloc = torch.cuda.max_memory_allocated(device) / 1e9
        reserved = torch.cuda.max_memory_reserved(device) / 1e9
        total = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"  peak GPU allocated         : {alloc:.2f} GB")
        print(f"  peak GPU reserved          : {reserved:.2f} GB   of {total:.1f} GB "
              f"({reserved / total * 100:.0f}%)")
        # Attention memory here is LINEAR in N, not quadratic: patches are a
        # fixed 1024 points, so n_patches = N/1024 grows while each patch's
        # PxP attention stays the same size. Everything else (MLP, projections,
        # activations) is linear in N too, so headroom scales ~linearly.
        per_point_mb = reserved / n * 1e3
        print(f"  => ~{per_point_mb:.3f} MB per point per unit batch "
              f"(all terms are linear in N -- patch count grows, patch size fixed)")
        for probe_n in (16384, 32768, 65536, 76333):
            for probe_b in (1, 2, 4, 8):
                est = per_point_mb * probe_n * probe_b / 1e3
                if est <= total * 0.92:
                    fits = f"~{est:.1f} GB"
                else:
                    fits = f"~{est:.1f} GB  DOES NOT FIT"
                if probe_b in (1, 8) or est > total * 0.5:
                    print(f"     N={probe_n:6d} batch {probe_b}: {fits}")
        print("  (linear extrapolation from ONE measured config -- treat as a "
              "planning estimate, confirm before committing to a long run)")

    # ----------------------------------------------------------------- checkpoint
    ckpt_dir = cfg.run_dir()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "overfit_final.pt"
    torch.save(dict(
        model_state=model.state_dict(),
        optimizer_state=opt.state_dict(),
        cfg=dataclasses.asdict(cfg),
        object_name=obj_name,
        object_names=obj_names,
        lengths=lengths,
        n_gaussians_used=n,
        step=args.steps,
        final_loss=final_loss,
    ), ckpt_path)
    cfg.save()
    print(f"checkpoint saved: {ckpt_path}")

    if args.skip_checks:
        # Benchmark mode: sampling is 3 x --sample-steps forward passes, which
        # would dominate (and distort the reported timing of) a short run.
        print("\n--skip-checks: sampling, checks 7/8/9 and renders skipped. "
              "This run is a perf measurement, NOT a gate.")
        logger.log_summary(final_loss=final_loss, train_seconds=dt,
                            ms_per_step=dt / args.steps * 1000,
                            ms_per_step_steady=steady_ms, skipped_checks=True)
        logger.close()
        return

    # ----------------------------------------------------------------- checks 7, 8, 9
    model.eval()

    print(f"\n[check 7] final training loss (mean, last 20 steps) = {final_loss:.4f} "
          f"(near 0 expected; render below is the real criterion -- \"a brown chair must render brown\")")

    print(f"[check 8] sampling from pure noise (DPMSolverMultistep, {args.sample_steps} steps), "
          "comparing to ground truth...")
    pred_sample0 = diffusion_sample(model, cond, offset, T=cfg.T,
                                     num_steps=args.sample_steps, seed=0, device=device)
    sample_mse = F.mse_loss(pred_sample0, target).item()
    print(f"    sampled-vs-ground-truth MSE (standardized RGB) = {sample_mse:.4f} "
          f"(render below is the real criterion)")

    print("[check 9] sampling twice more with different seeds...")
    pred_seed1 = diffusion_sample(model, cond, offset, T=cfg.T,
                                   num_steps=args.sample_steps, seed=1, device=device)
    pred_seed2 = diffusion_sample(model, cond, offset, T=cfg.T,
                                   num_steps=args.sample_steps, seed=2, device=device)
    diversity = (pred_seed1 - pred_seed2).abs().mean().item()
    print(f"    mean |seed1 - seed2| (standardized RGB) = {diversity:.4f}")
    print("    NOTE (STAGE_SPEC §9 check 9 is a dataset-level mode-collapse check --")
    print("    \"most chairs near-grey, predict muddy grey-brown always\" is about averaging")
    print("    ACROSS objects during full training). For a single memorized object there is only")
    print("    one true answer, so LOW diversity here is the expected, correct outcome of a")
    print("    successful overfit, not evidence of collapse -- reported as a diagnostic number,")
    print("    not gated pass/fail. This check becomes load-bearing again at the full-dataset")
    print("    training run (step 9), not here.")

    # ----------------------------------------------------------------- render
    # One row per object. Each object's slice of the concatenated batch is
    # decoded with ITS OWN centre/radius -- per-object normalisation
    # (preprocessing §3.4) means using the wrong object's radius silently
    # produces a wrongly-scaled splat rather than an error.
    fig, axes = plt.subplots(len(obj_names), 5,
                              figsize=(3.6 * 5, 4.4 * len(obj_names)))
    if len(obj_names) == 1:
        axes = axes[None, :]

    cursor = 0
    per_object_mse = []
    for row, (name, centre_i, radius_i) in enumerate(metas):
        n_i = lengths[row]
        sl = slice(cursor, cursor + n_i)
        cond_i = cond_np[sl]

        gt_tok = to_tokens(cond_i, target_np[sl], cfg, norm_stats)
        s0 = to_tokens(cond_i, pred_sample0[sl], cfg, norm_stats)
        s1 = to_tokens(cond_i, pred_seed1[sl], cfg, norm_stats)
        s2 = to_tokens(cond_i, pred_seed2[sl], cfg, norm_stats)

        mse_i = F.mse_loss(pred_sample0[sl], target[sl]).item()
        div_i = (pred_seed1[sl] - pred_seed2[sl]).abs().mean().item()
        per_object_mse.append((name, mse_i, div_i))

        # The INPUT the model actually gets: the conditioning channels as
        # given, every TARGET channel replaced by the dataset mean. At stage 3
        # that withholds size and opacity as well as colour -- a panel that
        # only greyed out the colour would be showing the viewer generated
        # geometry and calling it input.
        geom_tok = geometry_only_tokens(cond_i, cfg, norm_stats)

        panels = [
            (geom_tok, f"INPUT: conditioning only\n(target ch = dataset mean)\n{name[:22]}  N={n_i}"),
            (gt_tok, "GROUND TRUTH\n(the training target)"),
            (s0, f"SAMPLED, seed 0\nMSE={mse_i:.4f}"),
            (s1, "SAMPLED, seed 1"),
            (s2, f"SAMPLED, seed 2\n|s1-s2|={div_i:.4f}"),
        ]
        for ax, (tok_i, title) in zip(axes[row], panels):
            ax.imshow(render_tokens(tok_i, size=260, device=device))
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        cursor += n_i

    if len(obj_names) > 1:
        print("\n    per-object (the thing a 1-object run cannot show):")
        for name, mse_i, div_i in per_object_mse:
            print(f"      {name[:34]:<34} MSE={mse_i:.4f}  seed-diversity={div_i:.4f}")

    plt.suptitle(f"overfit: {obj_name}   stage {cfg.stage} ({', '.join(cfg.target_names)})\n"
                  f"{args.steps} steps, {len(obj_names)} object(s), N={lengths}, "
                  f"L_simple={final_loss:.5f}, amp={args.amp}, renderer={active_backend()}",
                  fontsize=11, y=1.10 if len(obj_names) == 1 else 1.02)
    plt.tight_layout()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / f"overfit_render_{cfg.run_name}.png"
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out_path}")

    # loss curve -- the trajectory shape matters, not just the final number
    # (LOGGING.md). Raw per-step loss is very noisy here because each step
    # draws a random t: high-t steps are intrinsically harder than low-t
    # ones, so the spread is mostly t-sampling, not instability. The
    # running-median overlay is what shows the actual trend.
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 4.2))
    ax2.plot(losses, linewidth=0.5, alpha=0.35, label="per-step (noisy: random t each step)")
    win = max(1, len(losses) // 50)
    smoothed = [float(np.median(losses[max(0, i - win):i + 1])) for i in range(len(losses))]
    ax2.plot(smoothed, linewidth=1.8, color="C3", label=f"running median (window {win})")
    ax2.plot(mse_losses, linewidth=0.9, color="C2", alpha=0.8, label="MSE term only")
    # L_image's two halves, unweighted. The weighted sum is 10x each and would
    # squash the MSE trace it needs to be read against; the raw numbers are the
    # ones that mean something. The vertical line is where the ramp opens.
    if render_loss is not None:
        rl1 = np.array([r for r in render_l1s])
        rlp = np.array([r for r in render_lpips_s])
        on = rl1 > 0
        if on.any():
            steps_arr = np.arange(len(rl1))
            ax2.plot(steps_arr[on], rl1[on], lw=1.0, color="C6", ls="-.",
                     label="render L1 (unweighted)")
            ax2.plot(steps_arr[on], rlp[on], lw=1.0, color="C9", ls="-.",
                     label="render LPIPS (unweighted)")
            ax2.axvline(int(steps_arr[on][0]), color="C6", ls=":", lw=1.2,
                        alpha=0.7)
    ax2.set(xlabel="step", ylabel="loss", yscale="log",
            title=f"overfit loss: {obj_name[:24]}")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # loss by timestep bucket -- LOGGING.md panel 1, "the single most
    # important panel": a flat near-zero high-t tail is what would justify
    # NORMALIZATION.md §4's input scaling.
    bin_means = t_bins.means()
    bin_centres = [(b + 0.5) * cfg.T / t_bins.n_bins for b in range(t_bins.n_bins)]
    ax3.plot(bin_centres, bin_means, "o-", color="C0")
    ax3.set(xlabel="diffusion timestep t", ylabel="mean MSE term in bucket", yscale="log",
            title="loss by timestep bucket (LOGGING.md panel 1)")
    ax3.grid(alpha=0.3)
    plt.tight_layout()
    loss_curve_path = REPORT_DIR / f"overfit_loss_curve_{cfg.run_name}.png"
    plt.savefig(loss_curve_path, dpi=110, bbox_inches="tight")
    plt.close(fig2)
    print(f"wrote {loss_curve_path}")

    logger.log_image(args.steps, "overfit_render", plt.imread(out_path))
    logger.log_summary(
        final_loss=final_loss, sample_mse=sample_mse, seed_diversity=diversity,
        check7_final_loss=final_loss, check8_sample_mse=sample_mse, check9_diversity=diversity,
        train_seconds=dt, ms_per_step=dt / args.steps * 1000,
    )
    logger.close()

    print(f"\n{'=' * 66}")
    print(f"check 7 (final loss near 0):          {final_loss:.5f}")
    print(f"check 8 (sampler MSE vs ground truth): {sample_mse:.5f}")
    print(f"check 9 (seed diversity, informational for a 1-object overfit): {diversity:.5f}")
    print(f"{'=' * 66}")
    print(f"renders:  {out_path}")
    print(f"curves:   {loss_curve_path}")
    print(f"ckpt:     {ckpt_path}")
    print("The render is the actual pass/fail call -- \"a brown chair must render")
    print("brown\" is a visual criterion, not a threshold.")


if __name__ == "__main__":
    main()
