"""
eval/test_eval.py — the held-out evaluation, run ONCE at the end of a run.

**There is no validation set in this project.** No hyperparameter search
and no early stopping happen during training — run length is fixed by
wall clock — so a third split would have nothing to do. The 170 held-out
objects are the TEST set: not tracked as a live curve, not looked at
between checkpoints, evaluated once against a finished model.

(`split.json` still calls that set `"val"` on disk. It is written once and
never regenerated — CLAUDE.md's never-cut list — so `train/dataset.py`
maps `split="test"` onto that key rather than rewriting the file.)

Two numbers, both against ground truth, since this stage is reconstruction
rather than distribution matching (STAGE_SPEC §9 — hence no 1-NNA/COV/MMD):

  test_loss        the training objective, L_simple, on unseen objects.
                   Averaged over a deterministic sweep of timesteps with
                   fixed-seed noise, so it is comparable between runs
                   rather than jittering with whatever `t` got drawn.
  sample MSE       full DPM-Solver sampling from pure noise, compared to
                   the true colours. This is the one that matters — it
                   measures the thing the model is actually asked to do.

PSNR/SSIM/LPIPS/FID/KID (STAGE_SPEC §9) are NOT here — they need rendering,
and they live in `eval/image_eval.py` on top of `eval/metrics.py` and
`eval/render3d.py`. `train/loop.py` runs both at the end of a run. This
file stays the cheap, render-free number: it needs no GPU rasterizer, so
it is the one that still works when gsplat does not.

    python eval/test_eval.py stage1_full                  # latest snapshot
    python eval/test_eval.py stage1_full --n-objects 8
"""

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import container_env  # noqa: F401 -- MUST precede torch

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')  # CLAUDE.md: 64MB /dev/shm

import numpy as np
import torch

from diffusion.schedule import add_noise, l_simple, make_train_scheduler  # noqa: F401
from eval.render_grid import sample_and_render
from model.denoiser import Denoiser
from stage_config import StageConfig, load_norm_stats
from train.dataset import ChairPointDataset, build_dataloader


@torch.no_grad()
def test_loss(model, cfg, device, amp_ctx, n_t_samples: int = 8,
              max_batches: int = 0, seed: int = 1234) -> dict:
    """L_simple over the test set at a FIXED, deterministic sweep of
    timesteps with fixed-seed noise.

    Random `t` per batch — what the training loop does — is fine as a
    training signal but makes a headline number that moves several tens of
    percent between evaluations purely from which timesteps came up. Here
    every object is scored at the same `n_t_samples` timesteps spread
    across [0, T), so two runs are compared on identical work.
    """
    was_training = model.training
    model.eval()
    # num_workers=0 deliberately. This runs at the END of training, while the
    # training loader's workers are still alive, and 8 more workers bursting
    # their prefetch queues took /dev/shm (64MB in the GPU container) over the
    # line -- `colourful8` finished training and then died here. An evaluation
    # is GPU-sampling-bound anyway, so the workers bought nothing.
    loader = build_dataloader(cfg, split="test", augment=False, shuffle=False,
                              num_workers=0)
    scheduler = make_train_scheduler(T=cfg.T, learn_sigma=cfg.learn_sigma)
    t_values = torch.linspace(0, cfg.T - 1, n_t_samples).long().tolist()

    total, per_ch_total, n_scores = 0.0, np.zeros(cfg.n_target_ch), 0
    by_t = {t: [] for t in t_values}
    n_objects = 0

    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        cond = batch["cond"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        offset = batch["offset"].to(device)
        batch_idx = batch["batch_idx"].to(device)
        B = offset.shape[0]
        n_objects += B
        gen = torch.Generator(device=device).manual_seed(seed + i)
        noise = torch.randn(target.shape, generator=gen, device=device)

        for t_val in t_values:
            t = torch.full((B,), t_val, dtype=torch.long, device=device)
            y_t = add_noise(scheduler, target, noise, t, batch_idx)
            with amp_ctx():
                out = model(cond, y_t, t, offset)
            # Mean half only. Under `cfg.learn_sigma` the model emits 2C
            # channels and the second half is the LEARNED_RANGE variance, not
            # a prediction of the target -- feeding it to an MSE would compare
            # a log-variance interpolant against a colour.
            pred = model.split_output(out.float())[0]
            loss = l_simple(pred, target)
            total += loss.item()
            by_t[t_val].append(loss.item())
            per_ch_total += ((pred.detach() - target) ** 2).mean(dim=0).cpu().numpy()
            n_scores += 1

    model.train(was_training)
    out = dict(test_loss=total / max(1, n_scores), n_test_objects=n_objects)
    for c, name in enumerate(cfg.target_names):
        out[f"test_loss_{name}"] = float(per_ch_total[c] / max(1, n_scores))
    out["test_loss_by_t"] = {int(t): float(np.mean(v)) for t, v in by_t.items() if v}
    return out


def run_test_eval(model, cfg, norm_stats, device, out_dir: Path,
                  train_names: list = None, n_render_objects: int = 4,
                  n_t_samples: int = 8, max_batches: int = 0,
                  amp_ctx=contextlib.nullcontext) -> dict:
    """Test loss + a TRAIN-vs-TEST render grid, for a finished model.

    The render keeps a couple of TRAIN rows beside the TEST rows on
    purpose: "did it generalize or memorize" is a comparison, and a TEST
    row alone doesn't answer it — you need to see what the model does on
    an object it HAS seen, in the same figure, at the same sampler
    settings.
    """
    t0 = time.time()
    results = test_loss(model, cfg, device, amp_ctx, n_t_samples=n_t_samples,
                        max_batches=max_batches)

    n_each = max(1, n_render_objects // 2)
    test_ds = ChairPointDataset(cfg, split="test", augment=False)
    sets = [("TEST", test_ds, list(range(min(n_each, len(test_ds)))))]
    if train_names:
        train_ds = ChairPointDataset(cfg, split="train", augment=False, names=train_names)
        sets.insert(0, ("TRAIN", train_ds, list(range(min(n_each, len(train_ds))))))

    path, mse_by_tag = sample_and_render(
        model, sets, cfg, norm_stats, device,
        out_path=out_dir / "test_eval_render.png",
        title_prefix=f"{cfg.run_name} — FINAL TEST EVALUATION\n")
    results.update({f"test_sample_mse_{t.lower()}": m for t, m in mse_by_tag.items()})
    results["test_render"] = str(path)
    results["test_eval_seconds"] = time.time() - t0
    return results


def print_results(r: dict):
    print("\n" + "=" * 66)
    print(f"TEST EVALUATION — {r['n_test_objects']} held-out objects")
    print("=" * 66)
    print(f"  test_loss (L_simple, {len(r['test_loss_by_t'])} fixed timesteps) : "
          f"{r['test_loss']:.5f}")
    ch = [f"{k[len('test_loss_'):]}={v:.5f}" for k, v in r.items()
          if k.startswith("test_loss_") and isinstance(v, float)
          and not k.endswith("_by_t")]
    if ch:
        print(f"  per-channel                                    : {'  '.join(ch)}")
    print("  by timestep (should RISE with t):")
    for t, v in sorted(r["test_loss_by_t"].items()):
        print(f"      t={t:4d}  {v:.5f}")
    for k, v in r.items():
        if k.startswith("test_sample_mse_"):
            print(f"  sampled-from-noise MSE, {k.rsplit('_', 1)[1].upper():<5} rows        : {v:.5f}")
    print(f"  render: {r['test_render']}")
    print("\n  MSE only. PSNR/SSIM/LPIPS/FID/KID are in eval/image_eval.py, which")
    print("  renders through gsplat — run it for the numbers that go in a table:")
    print("    python eval/image_eval.py <run-name>")
    print("=" * 66)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    ap.add_argument("--snapshot", default=None,
                    help="default: the newest snapshot_*.pt in the run dir")
    ap.add_argument("--n-objects", type=int, default=4,
                    help="objects in the render grid, split TRAIN/TEST")
    ap.add_argument("--n-t-samples", type=int, default=8)
    ap.add_argument("--max-batches", type=int, default=0, help="0 = whole test set")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers. 0 by default and you almost "
                         "certainly want to keep it there: /dev/shm is 64MB in "
                         "the GPU container and an evaluation is GPU-bound, so "
                         "workers buy nothing and can blow the shm budget.")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    run_dir = Path("runs") / args.run_name
    cfg = StageConfig.load(run_dir / "config.json")
    cfg.num_workers = args.num_workers
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    snap = Path(args.snapshot) if args.snapshot else max(
        run_dir.glob("snapshot_*.pt"), key=lambda p: p.stat().st_mtime, default=None)
    if snap is None:
        raise SystemExit(f"no snapshot_*.pt in {run_dir} — has the run finished a checkpoint?")
    print(f"evaluating {snap} on {device}")

    model = Denoiser(cfg).to(device)
    # `snapshot_*.pt` holds EMA weights under "ema_state"; `state_latest.pt`
    # (a full training state) holds them under "ema" alongside the live
    # weights. Evaluate the EMA either way — STAGE_SPEC §5: train live,
    # sample and evaluate EMA.
    state = torch.load(snap, map_location=device, weights_only=False)
    key = next((k for k in ("ema_state", "ema") if k in state), None)
    if key is None:
        raise SystemExit(f"{snap} has no EMA weights (keys: {sorted(state)})")
    model.load_state_dict(state[key])
    model.eval()

    amp_ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) \
        if (cfg.amp == "bf16" and device.type == "cuda") else contextlib.nullcontext

    subset_path = run_dir / "train_subset.json"
    train_names = json.loads(subset_path.read_text())["names"] if subset_path.exists() else None

    results = run_test_eval(model, cfg, load_norm_stats(cfg.shard_path), device, run_dir,
                            train_names=train_names, n_render_objects=args.n_objects,
                            n_t_samples=args.n_t_samples, max_batches=args.max_batches,
                            amp_ctx=amp_ctx)
    print_results(results)
    (run_dir / "test_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {run_dir / 'test_results.json'}")


if __name__ == "__main__":
    main()
