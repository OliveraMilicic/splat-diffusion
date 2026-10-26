"""
train/loop.py — STAGE_SPEC §11 build-order step 9: the real multi-object
training run. Everything the overfit gate deliberately disabled is on
here: augmentation, EMA, checkpoint/resume, periodic render grids.

**Train and test only — no validation set.** Nothing during training
selects a model: no hyperparameter search, no early stopping, run length
fixed by wall clock. A validation split would have no job, so held-out
data is not touched during the run at all. The held-out 170 objects are
the TEST set, evaluated ONCE when training finishes (`eval/test_eval.py`),
and the mid-training render grids show TRAIN objects only.

Per-step order (TRAINING_SPEC §5, adapted to STAGE_SPEC's schedule since
EDM preconditioning is superseded):

    1. batch of objects -> (total_N, C) concat + offset  [augmented]
    2. t ~ U{0, T-1}, one INDEPENDENT timestep per object
    3. eps ~ N(0, I)
    4. y_t = sqrt(abar_t)*y_0 + sqrt(1-abar_t)*eps        [diffusion/schedule]
    5. out = model(cond, y_t, t, offset)   [y0_hat, and the learned variance
                                            when cfg.learn_sigma]
    6. loss = GaussianCube's hybrid: mse * w(snr) + vb   [diffusion/losses.py]
    7. backward under bf16 autocast
    8. NaN/Inf guard BEFORE the optimizer step
    9. grad clip to cfg.grad_clip
   10. optimizer.step()
   11. EMA update (decay 0.999 -- STAGE_SPEC §5, not DESIGN_SPEC's 0.9999)
   12. log

Run:
    python train/loop.py --run-name stage3_full
    python train/loop.py --run-name stage3_full --resume     # after a drop
"""

import argparse
import contextlib
import gc
import json
import os
import signal
import sys
import time
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# MUST be set before CUDA initialises. This run alternates two very different
# allocation patterns -- training at batch 4 x N=32768 (~35.5GB peak), then a
# render doing 400 batch-1 sampling passes -- which fragments the caching
# allocator badly. `stage1_100obj` died of exactly that at step 2001: 30.8GiB
# allocated but 8.3GiB reserved-and-unusable, unable to find a contiguous
# 96MiB. Expandable segments let the allocator grow/shrink a segment instead
# of hunting for a contiguous block, which is the documented fix for this
# pattern. Respect an existing value rather than overriding a deliberate one.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import container_env  # noqa: F401 -- MUST precede torch

import torch.multiprocessing
# CLAUDE.md / TRAINING_SPEC §4: /dev/shm is 64MB here; num_workers>0 dies
# with a Bus error without this.
torch.multiprocessing.set_sharing_strategy('file_system')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from diffusion.losses import DiffusionLoss
from diffusion.render_loss import RenderLoss
from diffusion.schedule import add_noise, make_train_scheduler
from eval.image_eval import print_image_metrics, run_image_metrics
from eval.render_grid import sample_and_render
from eval.test_eval import print_results, run_test_eval
from model.denoiser import Denoiser
from stage_config import (CHANNEL_NAMES, StageConfig, load_norm_stats, load_split,
                           lr_at)
from train.checkpoint import load_training_state, save_snapshot, save_training_state
from train.dataset import ChairPointDataset, build_dataloader
from train.ema import EMA
from train.logging_utils import (StageLogger, TimestepBinTracker,
                                  per_term_grad_norms)
from train.plot_run import plot_run


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


def resolve_train_subset(cfg: StageConfig, run_dir: Path,
                          explicit: list = None) -> list:
    """The train-object subset for this run, or None for the whole split.

    Written to `runs/<name>/train_subset.json` the first time and RELOADED
    thereafter, so `--resume` can never silently continue on a different
    set of objects than the run started with — same rule as the
    train/val split itself (CLAUDE.md never-cut list: "a fixed
    train/val split written to disk once").

    `explicit` is a caller-chosen list of object names
    (`--train-names-file`), for experiments where WHICH objects matters and
    a seeded random draw will not do: the colourful-batch and brown-prior
    overfits pick their objects on measured colour
    (`experiments/select_chairs.py`), and the loss ablation must reuse the
    L2 run's exact 100 names or it is comparing two things at once.

    The TEST split is deliberately never subset: runs at different train
    sizes then share an identical held-out set, which is what makes their
    final test numbers directly comparable.
    """
    subset_path = run_dir / "train_subset.json"
    if subset_path.exists():
        names = json.loads(subset_path.read_text())["names"]
        # A resume that quietly swapped the object set would invalidate the
        # run without any visible symptom, which is the entire reason this
        # file is written once. Say so instead.
        if explicit is not None and sorted(explicit) != sorted(names):
            raise SystemExit(
                f"{subset_path} already records {len(names)} objects, but "
                f"--train-names-file asks for a different set of {len(explicit)}. "
                f"Refusing to continue a run on different data than it started "
                f"on — use a new --run-name, or delete that file deliberately.")
        print(f"train subset: {len(names)} objects (reloaded from {subset_path.name})")
        return names
    if explicit is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        subset_path.write_text(json.dumps(
            dict(seed=cfg.seed, n=len(explicit), source="explicit",
                 names=list(explicit)), indent=2))
        print(f"train subset: {len(explicit)} EXPLICIT objects -> {subset_path}")
        return list(explicit)
    if not cfg.n_train_objects:
        return None

    all_names = load_split(cfg.shard_path)["train"]
    k = min(cfg.n_train_objects, len(all_names))
    if k < cfg.n_train_objects:
        print(f"WARNING: asked for {cfg.n_train_objects} train objects, "
              f"split only has {len(all_names)}; using all of them")
    rng = np.random.default_rng(cfg.seed)
    names = sorted(rng.choice(all_names, size=k, replace=False).tolist())
    run_dir.mkdir(parents=True, exist_ok=True)
    subset_path.write_text(json.dumps(dict(seed=cfg.seed, n=k, names=names), indent=2))
    print(f"train subset: {k} objects, seed {cfg.seed} -> {subset_path}")
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, default="stage3_full")
    ap.add_argument("--stage", type=int, default=3,
                     help="3 = the current build target (log-scale + opacity + RGB "
                          "diffused, xyz + quaternion frozen). 1 = RGB only, which "
                          "is what the PLAN.md gate runs used.")
    ap.add_argument("--no-learn-sigma", action="store_true",
                     help="drop GaussianCube's learned-variance head and its VB term, "
                          "training on the plain L_simple MSE instead. For an ablation "
                          "against the hybrid loss, not for the headline run.")
    ap.add_argument("--min-snr-gamma", type=float, default=None,
                     help="Min-SNR-gamma loss weighting. GaussianCube ships this OFF "
                          "(min_snr=False); 5.0 is the value in their disabled path.")
    # GaussianCube's image-space loss term (diffusion/render_loss.py). ON by
    # default: it is the other half of what their train.py actually optimises,
    # and at stage 3 it is the only term that can see scale and opacity as
    # SHAPE rather than as numbers.
    ap.add_argument("--no-render-loss", action="store_true",
                     help="drop L_image and train on the diffusion loss alone")
    ap.add_argument("--render-l1-weight", type=float, default=10.0,
                     help="GaussianCube's shipped ShapeNet value")
    ap.add_argument("--render-lpips-weight", type=float, default=10.0,
                     help="GaussianCube's shipped ShapeNet value")
    ap.add_argument("--render-loss-start", type=int, default=10_000,
                     help="step at which L_image starts ramping in. GaussianCube "
                          "hard-switches at 50,000 of a ~1-week 16-GPU run; scaled "
                          "to this run length and softened to a ramp (STAGE_SPEC §5: "
                          "rendering an untrained prediction gives enormous gradients)")
    ap.add_argument("--render-loss-ramp", type=int, default=2_000)
    ap.add_argument("--render-loss-views", type=int, default=1,
                     help="cameras per object per step (their --load_camera 1)")
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
    # Defaults are the configuration decided in reports/ENGINEERING_REPORT.md §8
    # (batch 4 x N=32768, bf16): 380 ms/step, 33.9GB of the 40GB card, and a
    # 10% per-object throughput GAIN over batch 1. Not StageConfig's defaults,
    # which stay at the lighter 8/16384 the checks and benchmarks use.
    ap.add_argument("--n-gaussians", type=int, default=32768)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--n-train-objects", type=int, default=0,
                     help="0 = whole train split. >0 trains on a seeded random "
                          "subset, recorded to runs/<name>/train_subset.json")
    ap.add_argument("--train-names-file", type=str, default=None,
                     help="JSON file with a {'names': [...]} list (or a bare list) "
                          "naming EXACTLY which train objects to use. Overrides "
                          "--n-train-objects. This is how the colour experiments "
                          "select their objects on measured colour "
                          "(experiments/select_chairs.py) and how the loss ablation "
                          "reuses the L2 run's exact 100 names — a seeded random "
                          "draw cannot express either.")
    ap.add_argument("--no-augment", action="store_true",
                     help="disable mirror+rotation augmentation. For the overfit "
                          "experiments, where the claim is that the model reproduces "
                          "THESE objects and augmentation only adds variance to a "
                          "question that isn't being asked.")
    ap.add_argument("--warmup-steps", type=int, default=None,
                     help="linear LR warmup length (GaussianCube's is 1000). Exposed "
                          "because an ablation against an earlier run has to match "
                          "that run's schedule, and stage1_100obj used 500.")
    ap.add_argument("--ema-decay", type=float, default=None,
                     help="EMA decay. The 0.999 default has a ~1000-step horizon, "
                          "which is most of a short overfit run — the EMA weights "
                          "everything is sampled from would still be averaging in "
                          "the untrained start. Lower it for short runs.")
    ap.add_argument("--max-steps", type=int, default=60_000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--amp", choices=["none", "bf16"], default="bf16")
    ap.add_argument("--num-workers", type=int, default=8)
    # Cadences on the CLI so a smoke run can exercise the render / checkpoint
    # branches in a handful of steps instead of thousands.
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--render-every", type=int, default=None)
    ap.add_argument("--ckpt-every", type=int, default=None)
    ap.add_argument("--n-render-objects", type=int, default=None,
                     help="TRAIN objects in the mid-training render grid")
    ap.add_argument("--sample-steps", type=int, default=None,
                     help="DPM-Solver steps for the render grid (50 for real runs; "
                          "drop it for a smoke run, where the render only has to "
                          "execute, not look good)")
    ap.add_argument("--test-t-samples", type=int, default=8,
                     help="timesteps per object in the FINAL test evaluation")
    ap.add_argument("--no-test", action="store_true",
                     help="skip the final held-out evaluation (run it later with "
                          "eval/test_eval.py against the saved snapshot)")
    ap.add_argument("--no-image-metrics", action="store_true",
                     help="skip the final PSNR/SSIM/LPIPS/FID/KID evaluation "
                          "(eval/image_eval.py). It samples and rasterizes every "
                          "test object, so it costs ~15 min at the full settings.")
    ap.add_argument("--metric-objects", type=int, default=0,
                     help="test objects in the image-metric evaluation (0 = all)")
    ap.add_argument("--metric-views", type=int, default=8,
                     help="azimuths per elevation for the image metrics")
    ap.add_argument("--no-keep-snapshots", action="store_true",
                     help="do NOT retain a per-render EMA snapshot. Retaining them "
                          "is what makes a training-progress figure possible at all: "
                          "snapshot_latest.pt is overwritten, so without this the "
                          "run ends holding only final weights and 'show the model "
                          "improving' becomes unbuildable after the fact.")
    ap.add_argument("--resume", action="store_true",
                     help="continue from runs/<run-name>/state_latest.pt, restoring "
                          "weights, optimizer, step count AND RNG state")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    cfg = StageConfig(stage=args.stage, n_gaussians=args.n_gaussians,
                       batch_size=args.batch_size, lr=args.lr, run_name=args.run_name,
                       max_steps=args.max_steps, amp=args.amp, num_workers=args.num_workers,
                       n_train_objects=args.n_train_objects,
                       learn_sigma=not args.no_learn_sigma)
    if args.min_snr_gamma is not None:
        cfg.min_snr_gamma = args.min_snr_gamma
    if args.clip_input_p is not None:
        cfg.clip_input_p = args.clip_input_p
    if args.warmup_steps is not None:
        cfg.warmup_steps = args.warmup_steps
    if args.ema_decay is not None:
        cfg.ema_decay = args.ema_decay
    for field in ("log_every", "render_every", "ckpt_every",
                  "n_render_objects", "sample_steps"):
        override = getattr(args, field)
        if override is not None:
            setattr(cfg, field, override)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"device: {device} — {props.name}, {props.total_memory / 1e9:.1f} GB")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print(f"device: {device}  (no CUDA — a full run here is not practical)")

    if cfg.amp == "bf16":
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise SystemExit("--amp bf16 needs a bf16-capable CUDA device")
        def amp_ctx():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        print("        bf16 autocast ON (master weights fp32, sampling fp32)")
    else:
        amp_ctx = contextlib.nullcontext

    # ------------------------------------------------------------------ data
    run_dir = cfg.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    explicit_names = None
    if args.train_names_file:
        payload = json.loads(Path(args.train_names_file).read_text())
        explicit_names = payload["names"] if isinstance(payload, dict) else payload
        print(f"--train-names-file {args.train_names_file}: {len(explicit_names)} names"
              + (f"\n  rule: {payload['rule']}" if isinstance(payload, dict)
                 and "rule" in payload else ""))
    train_names = resolve_train_subset(cfg, run_dir, explicit=explicit_names)
    augment = not args.no_augment
    train_loader = build_dataloader(cfg, split="train", augment=augment, names=train_names)
    norm_stats = load_norm_stats(cfg.shard_path)
    n_train = len(train_loader.dataset)
    steps_per_epoch = max(1, n_train // cfg.batch_size)   # drop_last=True on train
    print(f"train objects: {n_train}, batch {cfg.batch_size}, N<={cfg.n_gaussians}, "
          f"augment {'ON' if augment else 'OFF'}")
    print(f"  {steps_per_epoch} steps/epoch -> {cfg.max_steps} steps = "
          f"{cfg.max_steps / steps_per_epoch:.0f} epochs")
    # drop_last=True on the train loader: with a small explicit subset that is
    # not a multiple of the batch size, some objects sit out each epoch (a
    # different random few, since the loader shuffles). Harmless in
    # expectation, but worth saying out loud when the whole point of the run is
    # that it fits THESE objects.
    if n_train % cfg.batch_size:
        print(f"  NOTE: {n_train} objects is not a multiple of batch {cfg.batch_size}; "
              f"{n_train % cfg.batch_size} sit out each epoch (drop_last, reshuffled "
              f"every epoch — no object is systematically excluded)")
    if not args.no_keep_snapshots:
        n_snap = max(0, cfg.max_steps // max(1, cfg.render_every))
        print(f"  keeping {n_snap} EMA snapshots (~{n_snap * 0.13:.1f} GB) for the "
              f"training-progress figure — eval/progress_grid.py")
    print("  no validation set — held-out data is untouched until the final "
          "test evaluation")

    # Mid-training render grid: TRAIN objects only. Held-out data is not
    # looked at during the run at all -- watching a test curve tick over is
    # how a test set quietly turns into a validation set, since it starts
    # informing decisions. Fixed indices, so the sequence of grids across
    # the run is comparable (LOGGING.md panel 5).
    render_train_ds = ChairPointDataset(cfg, split="train", augment=False,
                                         names=train_names)
    render_sets = [("TRAIN", render_train_ds,
                    list(range(min(cfg.n_render_objects, len(render_train_ds)))))]

    # ------------------------------------------------------------------ model
    model = Denoiser(cfg).to(device)
    ema = EMA(model, decay=cfg.ema_decay)
    ema.shadow.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = make_train_scheduler(T=cfg.T, learn_sigma=cfg.learn_sigma)
    loss_fn = DiffusionLoss(scheduler, learn_sigma=cfg.learn_sigma,
                             min_snr_gamma=cfg.min_snr_gamma, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.2f}M params, EMA decay {cfg.ema_decay}")
    print(f"stage {cfg.stage} — diffusing {cfg.n_target_ch} channels "
          f"({', '.join(cfg.target_names)}); conditioning on "
          f"{cfg.n_cond_ch} ({', '.join(CHANNEL_NAMES[c] for c in cfg.cond_ch)})")
    render_loss = None
    if not args.no_render_loss:
        render_loss = RenderLoss(
            cfg, norm_stats, device, image_size=args.render_loss_size,
            n_views=args.render_loss_views, l1_weight=args.render_l1_weight,
            lpips_weight=args.render_lpips_weight,
            start_step=args.render_loss_start, ramp_steps=args.render_loss_ramp,
            seed=cfg.seed, pixel_norm=args.render_pixel_norm)

    print(f"loss: GaussianCube hybrid — MSE"
          f"{' * min(snr, %g)' % cfg.min_snr_gamma if cfg.min_snr_gamma else ''}"
          f"{' + VB (learned variance, %d out ch)' % cfg.n_model_out_ch if cfg.learn_sigma else ' only (learn_sigma OFF)'}")
    if render_loss is not None:
        print(f"      + L_image: {args.render_l1_weight:g}*L1 + "
              f"{args.render_lpips_weight:g}*LPIPS on {args.render_loss_views} "
              f"gsplat view(s) @ {args.render_loss_size}px, backend="
              f"{render_loss.backend}, ramping in over steps "
              f"{args.render_loss_start}-{args.render_loss_start + args.render_loss_ramp}")
    else:
        print("      L_image OFF (--no-render-loss)")

    cfg.save()

    start_step = 0
    state_path = run_dir / "state_latest.pt"
    if args.resume:
        if not state_path.exists():
            raise SystemExit(f"--resume given but {state_path} does not exist")
        start_step = load_training_state(state_path, model, ema, opt, device)

    logger = StageLogger(cfg, run_name=cfg.run_name, use_wandb=not args.no_wandb)
    logger.log_summary(n_params=n_params, n_train_objects=n_train, device=str(device),
                        amp=cfg.amp, resolved_N=cfg.n_gaussians, spconv_available=False,
                        steps_per_epoch=steps_per_epoch, validation_set=False,
                        planned_epochs=cfg.max_steps / steps_per_epoch)

    # ---------------------------------------------------- save-on-SIGTERM (§6)
    def _save_and_exit(signum, frame):
        print(f"\nsignal {signum} received — saving training state before exit")
        save_training_state(state_path, model, ema, opt, cfg, _save_and_exit.step)
        logger.close()
        raise SystemExit(0)
    _save_and_exit.step = start_step
    signal.signal(signal.SIGTERM, _save_and_exit)
    signal.signal(signal.SIGINT, _save_and_exit)

    # ------------------------------------------------- L_image feasibility probe
    # L_image only switches on at --render-loss-start (10k by default). Without
    # this, a configuration whose render+LPIPS backward does not fit would train
    # happily for 10,000 steps and THEN start OOMing -- hours in, at the worst
    # possible moment. Run one real render-loss forward+backward now, at full
    # batch and full N, so an unaffordable config fails in the first seconds
    # with something actionable.
    if render_loss is not None:
        probe = next(iter(train_loader))
        p_cond = probe["cond"].to(device)
        p_target = probe["target"].to(device)
        p_lengths = probe["lengths"].tolist()
        p_offset = probe["offset"].to(device)
        p_batch_idx = probe["batch_idx"].to(device)
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            # A REAL training step with L_image forced on, not a synthetic
            # prediction. The peak that matters is the training graph and the
            # render graph LIVE AT THE SAME TIME -- one backward covers both --
            # and probing the render loss alone measures neither. At batch 4 x
            # N=32768 training alone already sits near 34 GB of a 42.6 GB card,
            # so "does the sum fit" is the actual question, and answering it
            # here costs one step instead of discovering it at step 10,000.
            #
            # Forward + backward only; NO optimizer step, so this does not
            # train on a loss the ramp says is not active yet.
            p_t = torch.randint(0, cfg.T, (p_offset.shape[0],), device=device)
            p_yt = add_noise(scheduler, p_target, torch.randn_like(p_target),
                              p_t, p_batch_idx)
            with amp_ctx():
                p_out = model(p_cond, p_yt, p_t, p_offset)
            p_diff_terms = loss_fn(p_out.float(), p_target, p_yt, p_t, p_lengths)
            p_pred = model.split_output(p_out.float())[0]
            # step forced past the ramp so the term is actually evaluated
            probe_terms = render_loss(p_pred, p_cond, p_target, p_lengths,
                                       step=args.render_loss_start + args.render_loss_ramp)
            (p_diff_terms["loss"] + probe_terms["render_total"]).backward()
            # Gradient at the model's own output is what proves L_image reaches
            # the network, not just some leaf tensor.
            g = torch.cat([p.grad.flatten() for p in model.parameters()
                           if p.grad is not None])[None, :]
            opt.zero_grad(set_to_none=True)
            assert g.numel() and torch.isfinite(g).all(), \
                "non-finite gradient on the first full step — see check_losses.py"
            # That L_image specifically reaches every generated channel is
            # check_render_loss.py R3's job, on a leaf it can isolate. Here the
            # question is whether the combined step is finite and fits.
            if device.type == "cuda":
                peak = torch.cuda.max_memory_allocated(device) / 1e9
                total = torch.cuda.get_device_properties(device).total_memory / 1e9
                mem = (f"\n    peak {peak:.1f} GB of {total:.1f} GB "
                       f"({peak / total:.0%}) for a FULL step incl. L_image")
                if peak / total > 0.92:
                    print(f"  WARNING: {peak:.1f}/{total:.1f} GB leaves almost no "
                          f"headroom. The render callback allocates on top of this "
                          f"every {cfg.render_every} steps; expect the OOM guard to "
                          f"fire. Consider --batch-size 3 or --render-loss-size 64.")
            else:
                mem = ""
            print(f"  full-step probe OK (fwd+bwd, L_image forced on, no optimizer "
                  f"step): mse={p_diff_terms['mse'].item():.4f} "
                  f"L1={probe_terms['render_l1'].item():.4f} "
                  f"LPIPS={probe_terms['render_lpips'].item():.4f}{mem}")
        except torch.OutOfMemoryError:
            raise SystemExit(
                f"A full step with L_image does not fit at batch {cfg.batch_size} "
                f"x N={cfg.n_gaussians}. It would have OOMed at step "
                f"{args.render_loss_start}, hours in. Lower --render-loss-size "
                f"(currently {args.render_loss_size}) or --batch-size, or pass "
                f"--no-render-loss.")
        finally:
            p_cond = p_target = p_pred = p_out = p_yt = None
            p_diff_terms = probe_terms = probe = g = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

    # ------------------------------------------------------------------ train
    t_bins = TimestepBinTracker(n_bins=10, T=cfg.T)
    batches = infinite(train_loader)
    model.train()
    t0 = time.time()
    n_skipped = 0
    n_oom, consecutive_oom = 0, 0
    # Bound before the loop so the OOM handler can always release them by
    # rebinding, whether or not the step that failed got as far as assigning.
    pred = loss = terms = terms_render = None
    # Assigned under `step % log_every == 0` but read by the every-100-steps
    # print; on --resume from a step that isn't a multiple of log_every the
    # first print would otherwise hit a NameError.
    metrics = {}

    for step in range(start_step, cfg.max_steps):
        _save_and_exit.step = step
        batch = next(batches)
        cond = batch["cond"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        offset = batch["offset"].to(device)
        batch_idx = batch["batch_idx"].to(device)
        B = offset.shape[0]

        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        t = torch.randint(0, cfg.T, (B,), device=device)
        noise = torch.randn_like(target)
        y_t = add_noise(scheduler, target, noise, t, batch_idx)

        # OOM guard, same spirit as the non-finite-loss guard below: drop the
        # batch and carry on rather than losing hours of training. Peak usage
        # is ~35.5GB of a 39.7GiB card, so a batch of four unusually large
        # objects has little room, and the render's 400 batch-1 sampling
        # passes churn the allocator. A dropped batch costs one step; an
        # uncaught OOM cost the whole run once already (step 2001 of
        # stage1_100obj), which is what this exists to prevent.
        try:
            # Drop the previous step's references before allocating the next
            # graph rather than after.
            pred = loss = terms = terms_render = None
            with amp_ctx():
                pred = model(cond, y_t, t, offset)
            # Loss OUTSIDE the autocast block: `diffusion/losses.py` computes
            # KL terms with `exp(logvar1 - logvar2)`, and bf16's ~3 decimal
            # digits are not enough for that. The network forward is the part
            # that benefits from bf16; the loss is a handful of elementwise ops
            # on its output and costs nothing in fp32.
            lengths_list = batch["lengths"].tolist()
            terms = loss_fn(pred.float(), target, y_t, t, lengths_list)
            loss = terms["loss"]

            # GaussianCube's image-space term, on the MEAN half of the output.
            # A no-op (and no render at all) until the ramp starts.
            if render_loss is not None:
                pred_y0 = model.split_output(pred.float())[0]
                terms_render = render_loss(pred_y0, cond, target, lengths_list, step)
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

            opt.zero_grad(set_to_none=True)
            loss.backward()
        except torch.OutOfMemoryError:
            n_oom += 1
            consecutive_oom += 1
            opt.zero_grad(set_to_none=True)
            # `loss`/`pred` are what matter here: they own the autograd graph,
            # i.e. essentially the whole ~30GB the step was using, and an OOM
            # inside backward() leaves them bound. empty_cache() cannot return
            # memory a live graph still references, so without this the guard
            # would OOM on every following step and trip its own ten-strike
            # bail-out instead of recovering from a one-batch spike.
            # `terms`/`terms_render` hold the same graph via their entries.
            pred = loss = terms = terms_render = None
            oom_lengths = batch["lengths"].tolist()
            del cond, target, y_t, noise, batch_idx, batch
            torch.cuda.empty_cache()
            print(f"step {step}: CUDA OOM (batch sizes {oom_lengths}), "
                  f"skipping ({n_oom} total)")
            # Repeated OOM is not a spike, it is a config that does not fit.
            # Failing loudly beats grinding through 60k skipped steps and
            # reporting a finished run that trained on nothing.
            if consecutive_oom >= 10:
                save_training_state(state_path, model, ema, opt, cfg, step)
                raise SystemExit(
                    f"10 consecutive CUDA OOMs at step {step} — this config does "
                    f"not fit. State saved; lower --batch-size or --n-gaussians "
                    f"and --resume.")
            continue
        consecutive_oom = 0

        # NaN/Inf guard BEFORE the step (TRAINING_SPEC §5) -- one bad batch
        # must not silently poison the EMA weights, which can't be un-poisoned
        # without rolling back to an earlier checkpoint.
        if not torch.isfinite(loss):
            n_skipped += 1
            print(f"step {step}: non-finite loss, skipping ({n_skipped} total)")
            opt.zero_grad(set_to_none=True)
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        ema.update(model)

        loss_val = loss.item()
        # Bin the MSE term, not the total. LOGGING.md panel 1 exists to show
        # denoising difficulty by noise level; the VB term's t=0 decoder-NLL
        # spike lands entirely in bin 0 and would swamp exactly the comparison
        # the panel is for.
        mse_val = terms["mse"].item()
        for t_i in t.tolist():
            t_bins.update(t_i, mse_val)

        if step % cfg.log_every == 0:
            lengths = batch["lengths"]
            # Per-channel error on the MEAN half of the output only — the
            # variance half is not a prediction of anything and squaring its
            # difference from the target is meaningless.
            pred_y0 = model.split_output(pred.detach().float())[0]
            per_ch = ((pred_y0 - target) ** 2).mean(dim=0).tolist()
            gates = [g for g in model.gate_means() if g is not None]
            metrics = dict(
                loss=loss_val, lr=lr, grad_norm=float(grad_norm),
                gate_mean=float(np.mean(gates)) if gates else 0.0,
                n_skipped=n_skipped, n_oom=n_oom,
                # The two halves of the hybrid loss, separately. They must be
                # readable apart: `vb` is the decoder NLL whenever an object
                # draws t=0 (~1 step in 250 at batch 4), which is a real spike
                # of tens of bits in an otherwise ~0.1 loss. Pooled into one
                # curve that looks like an instability; split out, it is
                # obviously the t=0 term doing its job.
                loss_mse=terms["mse"].item(),
                loss_vb=terms["vb"].item() if "vb" in terms else 0.0,
                render_l1=terms_render["render_l1"].item() if terms_render else 0.0,
                render_lpips=terms_render["render_lpips"].item() if terms_render else 0.0,
                render_weight=terms_render["render_weight"] if terms_render else 0.0,
                # epochs, not steps, are the comparable unit between a
                # 100-object run and a 1103-object one at the same batch size
                epoch=step / steps_per_epoch,
                step_time=(time.time() - t0) / max(1, step - start_step + 1),
                # LOGGING.md panel 6: confirm batches genuinely mix sizes
                points_min=int(lengths.min()), points_mean=float(lengths.float().mean()),
                points_max=int(lengths.max()),
            )
            # Real channel names (stage_config.CHANNEL_NAMES), not a hardcoded
            # r/g/b: at stage 3 the target is 7 channels and a curve labelled
            # "r" that is actually log_sx is worse than no label.
            for c, name in enumerate(cfg.target_names):
                metrics[f"loss_{name}"] = per_ch[c]
            metrics.update(grad_terms)
            if device.type == "cuda":
                metrics["gpu_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
            for b, m in enumerate(t_bins.means()):
                if not np.isnan(m):
                    metrics[f"loss_t_bin_{b}"] = float(m)
            logger.log(step, **metrics)

        if step % 100 == 0:
            el = time.time() - t0
            done = step - start_step + 1
            eta = (cfg.max_steps - step) * el / max(1, done) / 60
            print(f"  step {step:6d}/{cfg.max_steps}  ep {step / steps_per_epoch:6.1f}  "
                  f"loss={loss_val:.4f} (mse={mse_val:.4f})  lr={lr:.2e}  "
                  f"gate={metrics.get('gate_mean', 0):.3f}  "
                  f"{el / max(1, done) * 1000:.0f}ms/step  ETA {eta:.0f}min")

        if step > 0 and step % cfg.render_every == 0:
            path, mse_by_tag = sample_and_render(
                ema.module, render_sets, cfg, norm_stats, device,
                out_path=run_dir / "renders" / f"step{step:07d}.png",
                title_prefix=f"{cfg.run_name} — step {step}   ")
            print(f"    [render] {path}  " +
                  "  ".join(f"{t} sample MSE={m:.4f}" for t, m in mse_by_tag.items()))
            logger.log_image(step, "samples", plt.imread(path))
            # the render as a number too, so it lands in the loss curves
            logger.log(step, **{f"sample_mse_{t.lower()}": m for t, m in mse_by_tag.items()})
            # Hand the sampler's blocks back before the next training step,
            # which needs ~35GB contiguous. sample_and_render already does
            # this internally; repeating it after the figure/logging work is
            # what actually precedes the next allocation.
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

        if step > 0 and step % cfg.ckpt_every == 0:
            save_training_state(state_path, model, ema, opt, cfg, step)
            save_snapshot(run_dir / "snapshot_latest.pt", ema, cfg, step)

        # A KEPT, step-named EMA snapshot at every render boundary. 129 MB each
        # (EMA weights only), ~30 of them over a 60k run = ~4 GB, against 12 TB
        # free. The alternative is that the only surviving weights are the final
        # ones, which makes eval/progress_grid.py -- and any "here is the model
        # at step N" figure -- impossible to build after the run rather than
        # merely expensive.
        if (not args.no_keep_snapshots and step > 0
                and step % cfg.render_every == 0):
            save_snapshot(run_dir / "snapshots" / f"ema_step{step:07d}.pt",
                          ema, cfg, step)

    # ------------------------------------------------------------------ done
    save_training_state(state_path, model, ema, opt, cfg, cfg.max_steps)
    save_snapshot(run_dir / f"snapshot_step{cfg.max_steps}.pt", ema, cfg, cfg.max_steps)
    # ALSO into snapshots/, so the progress figure's last column is the final
    # model. Without this the figure silently stops at the last render boundary
    # and omits the one result anybody actually wants to see.
    if not args.no_keep_snapshots:
        save_snapshot(run_dir / "snapshots" / f"ema_step{cfg.max_steps:07d}.pt",
                      ema, cfg, cfg.max_steps)
    train_minutes = (time.time() - t0) / 60
    summary = dict(final_step=cfg.max_steps, n_skipped=n_skipped, n_oom=n_oom,
                    total_minutes=train_minutes)

    # Shut the training workers down before the evaluation starts. Nothing
    # below reads `train_loader`, but `batches` still holds its iterator, so
    # without this its 8 worker processes stay alive -- and their share of the
    # 64MB /dev/shm stays allocated -- while the evaluation brings up loaders
    # of its own. That is what killed `colourful8` AFTER it had finished
    # training and saved every checkpoint: 16 live workers, one 64MB budget.
    # The evaluation loaders now also run at num_workers=0 (eval/test_eval.py),
    # so this is the belt to that suspenders; both are cheap and the failure
    # mode they prevent costs a whole run's evaluation.
    del batches
    del train_loader
    gc.collect()

    # --------------------------------------------- the ONE held-out evaluation
    # First and only time this run touches the test set (STAGE_SPEC §9).
    # Sampling and evaluation use the EMA weights, training used the live
    # ones (STAGE_SPEC §5).
    results = None
    if not args.no_test:
        print("\nrunning the final test evaluation (held-out objects, "
              "first and only look)...")
        results = run_test_eval(ema.module, cfg, norm_stats, device, run_dir,
                                 train_names=train_names,
                                 n_render_objects=max(2, cfg.n_render_objects),
                                 n_t_samples=args.test_t_samples, amp_ctx=amp_ctx)
        (run_dir / "test_results.json").write_text(json.dumps(results, indent=2))
        summary.update({k: v for k, v in results.items()
                        if isinstance(v, (int, float))})

    # -------------------------------------------- the image metrics (STAGE_SPEC §9)
    # Separate from the MSE evaluation above on purpose: this one needs a GPU
    # rasterizer, so it is the part that can fail on a machine where the rest
    # of the evaluation still works. Running it last means a failure here
    # cannot cost the run its test_results.json.
    image_results = None
    if not args.no_test and not args.no_image_metrics:
        print("\nrendering the held-out set and scoring PSNR/SSIM/LPIPS/FID/KID...")
        try:
            image_results = run_image_metrics(
                ema.module, cfg, norm_stats, device,
                n_objects=args.metric_objects, n_views=args.metric_views,
                sample_steps=cfg.sample_steps)
            (run_dir / "image_metrics.json").write_text(json.dumps(image_results, indent=2))
            for tag in ("model", "mean_fill"):
                for k in ("psnr", "ssim", "lpips", "fid", "kid"):
                    v = image_results[tag].get(k)
                    if v is not None:
                        summary[f"{tag}_{k}"] = v
        except Exception as e:
            # A finished 6-hour training run must not be reported as failed
            # because the rasterizer or a metric net misbehaved. Say so loudly
            # and keep everything else.
            print(f"  image metrics FAILED ({type(e).__name__}: {e})")
            print(f"  rerun standalone: python eval/image_eval.py {cfg.run_name}")
            summary["image_metrics_error"] = f"{type(e).__name__}: {e}"

    logger.log_summary(**summary)
    logger.close()

    # Curves last: reads the stats.jsonl this run just finished writing, so
    # the figure exists without a separate command (LOGGING.md — the run
    # should leave its own evidence behind).
    curves = plot_run(cfg.run_name)
    print(f"\ntraining complete: {cfg.max_steps} steps "
          f"({cfg.max_steps / steps_per_epoch:.0f} epochs) in {train_minutes:.0f} min")
    print(f"  curves:       {curves}")
    print(f"  snapshot:     {run_dir / f'snapshot_step{cfg.max_steps}.pt'}")
    if results is not None:
        print_results(results)
        print(f"  wrote {run_dir / 'test_results.json'}")
    if image_results is not None:
        print_image_metrics(image_results)
        print(f"  wrote {run_dir / 'image_metrics.json'}")
    else:
        print(f"  --no-test given; run it later with:\n"
              f"    python eval/test_eval.py {cfg.run_name}")


if __name__ == "__main__":
    main()
