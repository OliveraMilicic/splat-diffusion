"""
eval/image_eval.py — the held-out image evaluation: sample every test
object, rasterize the prediction and its ground truth under identical
cameras, and score PSNR / SSIM / LPIPS / FID / KID.

This is the evaluation STAGE_SPEC §9 asks for and that `eval/test_eval.py`
previously said, in its own output, was missing.

**Everything is scored against a baseline in the same table.** `mean_fill`
— every target channel set to the dataset mean, i.e. the model's INPUT
panel — is run through the identical sampler-free path. Without it a PSNR
has no scale: chairs are mostly grey-brown, so "predict the dataset mean"
is a genuinely strong predictor (STAGE_SPEC §9 check 9 is the warning that
it can beat a real model on exactly these metrics), and a diffusion model
that does not clear it has not earned the machinery.

`oracle` is the third row: ground truth scored against itself. It is not a
result, it is the ceiling — PSNR caps at 50 dB by construction
(`eval/metrics.py` clamps the +inf), SSIM at 1.0, FID at 0. A run where
`oracle` is not at the ceiling means the evaluation is broken, not the
model.

    python eval/image_eval.py stage3_full
    python eval/image_eval.py stage3_full --n-objects 8 --n-views 4
"""

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import container_env  # noqa: F401 -- MUST precede torch

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import numpy as np
import torch

from diffusion.sampling import sample as diffusion_sample
from diffusion.sampling import sample_ddpm
from eval.metrics import (MetricSuite, format_gaussiancube_comparison,
                           format_results)
from eval.load_run import load_run
from eval.render3d import gsplat_available, orbit_cameras, render_views
from eval.render_grid import mean_fill_target, to_tokens
from stage_config import load_norm_stats
from train.dataset import ChairPointDataset


@torch.no_grad()
def run_image_metrics(model, cfg, norm_stats, device, n_objects: int = 0,
                      n_views: int = 8, image_size: int = 256,
                      sample_steps: int = None, seed: int = 0,
                      kid_subset_size: int = 100, split: str = "test",
                      progress_every: int = 10, sampler: str = "dpm") -> dict:
    """Sample, render, score. Returns {tag: metrics dict} for `model`,
    `mean_fill` and `oracle`."""
    viewmats, Ks = orbit_cameras(n_views=n_views, image_size=image_size)
    backend = "gsplat" if gsplat_available() else "points"
    ds = ChairPointDataset(cfg, split=split, augment=False)
    n = len(ds) if not n_objects else min(n_objects, len(ds))
    sample_steps = sample_steps or cfg.sample_steps

    suites = {tag: MetricSuite(device, kid_subset_size=kid_subset_size,
                               backend=backend)
              for tag in ("model", "mean_fill", "oracle")}

    was_training = model.training
    model.eval()
    t0 = time.time()
    for i in range(n):
        item = ds[i]
        cond = item["cond"].to(device)
        target = item["target"].to(device)
        offset = torch.tensor([item["n"]], dtype=torch.long, device=device)

        # A FIXED seed per object index, not one global stream: the sample for
        # object 17 is then the same whether the run covered 20 objects or 170,
        # so a quick --n-objects 8 pass is a prefix of the full evaluation
        # rather than a different experiment.
        # DPM-Solver is a deterministic ODE: all randomness is in the initial
        # noise. Measured on this model, that collapses object colour to the
        # conditional mean (grey), because geometry does not determine hue and
        # the mean is what minimises MSE. DDPM ancestral sampling injects noise
        # at every step and commits to a hue instead. The two optimise
        # different things -- PSNR rewards the mean, FID/KID reward the
        # distribution -- so the evaluation can report either, and reporting
        # only one hides the trade-off.
        sample_fn = diffusion_sample if sampler == "dpm" else sample_ddpm
        pred = sample_fn(model, cond, offset, T=cfg.T,
                         num_steps=sample_steps, seed=seed + i, device=device)

        cond_np = cond.cpu().numpy()
        tok_gt = to_tokens(cond_np, target, cfg, norm_stats)
        tok_pred = to_tokens(cond_np, pred, cfg, norm_stats)
        tok_mean = to_tokens(cond_np, mean_fill_target(cond_np, cfg), cfg, norm_stats)

        def rendered(tok):
            return render_views(tok, viewmats, Ks, image_size=image_size,
                                device=device, backend=backend)[0]

        imgs_gt = rendered(tok_gt)
        suites["model"].update(rendered(tok_pred), imgs_gt, name=item["name"])
        suites["mean_fill"].update(rendered(tok_mean), imgs_gt, name=item["name"])
        # The oracle renders ground truth twice through the same path. It is
        # bit-identical by construction, so it verifies the pipeline (identical
        # cameras, deterministic rasterization) rather than measuring anything.
        suites["oracle"].update(imgs_gt, imgs_gt, name=item["name"])

        if device.type == "cuda" and (i + 1) % 20 == 0:
            torch.cuda.empty_cache()
        if progress_every and (i + 1) % progress_every == 0:
            el = time.time() - t0
            print(f"    {i + 1}/{n} objects  ({el / (i + 1):.1f}s/object, "
                  f"ETA {(n - i - 1) * el / (i + 1) / 60:.1f} min)")

    model.train(was_training)
    out = {tag: s.compute() for tag, s in suites.items()}
    out["model"]["per_object"] = suites["model"].per_object_table()
    out["_meta"] = dict(n_objects=n, n_views=n_views, image_size=image_size,
                        sample_steps=sample_steps, seed=seed, split=split,
                        render_backend=backend, stage=cfg.stage,
                        target_channels=cfg.target_names,
                        learn_sigma=cfg.learn_sigma, sampler=sampler,
                        seconds=time.time() - t0)
    return out


def print_image_metrics(results: dict):
    m = results["_meta"]
    print("\n" + "=" * 78)
    print(f"IMAGE METRICS — {m['n_objects']} {m['split'].upper()} objects x {m['n_views']} views "
          f"@ {m['image_size']}px, {m['sample_steps']} sampler steps")
    print(f"stage {m['stage']} ({', '.join(m['target_channels'])}), "
          f"learn_sigma={m['learn_sigma']}, sampler={m.get('sampler', 'dpm')}, "
          f"renderer={m['render_backend']}")
    print("=" * 78)
    for tag in ("oracle", "mean_fill", "model"):
        if tag in results:
            print(format_results(tag, results[tag]))
    if m["render_backend"] != "gsplat":
        print("\n  WARNING: rendered with the POINT-CLOUD fallback, not gsplat.")
        print("  It draws no covariance and no real opacity compositing, so it is")
        print("  blind to two of the three channel groups stage 3 generates.")
        print("  These numbers check the pipeline; they are not results.")
    worst = results["model"].get("per_object", [])[:3]
    if worst:
        print("\n  worst 3 objects by PSNR (where to look first):")
        for o in worst:
            print(f"    {o['name'][:38]:<38} PSNR {o['psnr']:6.2f}  SSIM {o['ssim']:.4f}  "
                  f"LPIPS {o['lpips']:.4f}")
    print()
    print(format_gaussiancube_comparison())
    print("=" * 78)


def load_model(run_dir: Path, snapshot: str, device):
    """Delegates to `eval/load_run.py`, which rebuilds the config from the
    CHECKPOINT rather than from config.json.

    This used to read config.json directly, which silently mis-builds any
    run older than a StageConfig default — `runs/stage1_100obj` predates
    `learn_sigma`, so its config would construct a 6-channel output head for
    3-channel weights. The ablation in `experiments/` evaluates exactly that
    run, so the shared loader is not optional here.
    """
    run_dir = Path(run_dir)
    model, cfg, meta = load_run(run_dir.name, device, snapshot=snapshot,
                                 run_root=str(run_dir.parent))
    print(f"loaded {Path(meta['snapshot']).name} [{meta['weights_key']}] — "
          f"stage {cfg.stage}, learn_sigma={cfg.learn_sigma}")
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--n-objects", type=int, default=0, help="0 = the whole split")
    ap.add_argument("--n-views", type=int, default=8,
                    help="azimuths; the rig renders this many at each of 2 elevations")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--sample-steps", type=int, default=None)
    ap.add_argument("--kid-subset-size", type=int, default=100)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--sampler", choices=["dpm", "ddpm"], default="dpm",
                    help="dpm = DPMSolverMultistep, deterministic, PSNR-optimal but "
                         "collapses colour to the conditional mean. ddpm = ancestral, "
                         "stochastic, produces saturated plausible colour (better "
                         "FID/KID, worse PSNR). Report BOTH -- they answer different "
                         "questions about an under-determined task.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="default: runs/<run>/image_metrics.json")
    args = ap.parse_args()

    run_dir = Path("runs") / args.run_name
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(run_dir, args.snapshot, device)

    results = run_image_metrics(
        model, cfg, load_norm_stats(cfg.shard_path), device,
        n_objects=args.n_objects, n_views=args.n_views, image_size=args.image_size,
        sample_steps=args.sample_steps, seed=args.seed,
        kid_subset_size=args.kid_subset_size, split=args.split,
        sampler=args.sampler)
    print_image_metrics(results)

    out = (Path(args.out) if args.out else
           run_dir / f"image_metrics_{args.sampler}.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
