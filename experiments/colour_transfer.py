"""
experiments/colour_transfer.py — experiment B's figure: what colour does a
model paint geometry it has never seen?

Experiment B trains a stage-1 (RGB-only) model on twelve BROWN chairs. The
claim it is meant to support is not "the model memorised twelve chairs" —
that is experiment A — but "the model learned a mapping from geometry to
appearance, and applies it to new geometry". The evidence for that is a
held-out chair, whose true colour is blue or green or magenta, coming out
of the model brown.

So this script:

  1. loads the run's EMA weights (`eval/load_run.py`),
  2. picks HELD-OUT objects whose ground-truth colour is as far from brown
     as the test split offers — the strongest possible counter-examples,
  3. samples RGB from pure noise conditioned only on their geometry,
  4. renders INPUT / MODEL / GROUND TRUTH side by side, and
  5. **measures** it: the same opacity-weighted `frac_brown` that chose the
     training set, applied to the model's own output.

Point 5 is the part that makes this an experiment rather than a picture.
"Looks brown to me" is not a result; `frac_brown = 0.91` on objects whose
ground truth is `0.02` is.

    python experiments/colour_transfer.py brown12
    python experiments/colour_transfer.py brown12 --n-objects 6 --split test

The figure is honest about the *negative* reading too: a model that paints
everything brown regardless of geometry has learned a constant, not a
mapping. The per-object spread panel is what separates those two, so it is
in the figure whichever way it comes out.
"""

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import container_env  # noqa: F401 -- MUST precede torch

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')  # CLAUDE.md: 64MB /dev/shm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from diffusion.sampling import sample as diffusion_sample
from eval.load_run import load_run
from eval.render3d import gsplat_available
from eval.render_grid import geometry_only_tokens, render_tokens, to_tokens
from experiments.select_chairs import (BROWN_HUE_DEG, colour_stats,
                                        colour_stats_rgb)
from splat_utils import find_ply_files, read_splat
from stage_config import load_norm_stats, load_split
from train.dataset import ChairPointDataset

REPORT_DIR = _REPO_ROOT / "reports"
EXP_DIR = _REPO_ROOT / "experiments"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def stats_of_tokens(tok_raw: np.ndarray, name: str = "") -> dict:
    """Colour stats of a RAW (N, 14) token array — the same measure applied
    to ground truth and to samples.

    Channel 10 is the opacity LOGIT (STAGE_SPEC §2), so it is activated
    here. Getting that wrong would weight the average by a number in
    roughly [-8, 6] instead of a probability, which silently produces a
    plausible-looking but meaningless mean colour.
    """
    return colour_stats_rgb(tok_raw[:, 11:14], sigmoid(tok_raw[:, 10]), name=name)


def pick_test_objects(cfg, k: int, mode: str) -> list:
    """Indices into the test split, and why.

    `far-from-brown` scans the held-out .ply files for colour and returns
    the least-brown, most-colourful ones. Those are the objects on which a
    brown prediction is unambiguous: if the model returns brown for a
    magenta chair, no amount of "well, chairs are mostly brown anyway" can
    explain it away. 170 objects at ~50 ms each, so the scan costs ~9 s and
    needs no cached table.
    """
    names = load_split(cfg.shard_path)["val"]
    if mode == "first":
        return list(range(min(k, len(names)))), None
    name_to_file = {f.stem: f for f in find_ply_files(cfg.data_dir)}
    print(f"scanning {len(names)} held-out objects for colour (~"
          f"{len(names) * 0.05:.0f}s)...")
    t0 = time.time()
    rows = []
    for i, n in enumerate(names):
        try:
            rows.append((i, colour_stats(read_splat(name_to_file[n]))))
        except Exception as e:
            print(f"  SKIP {n}: {type(e).__name__}: {e}")
    print(f"  done in {time.time() - t0:.0f}s")
    # Qualify, then spread — the same two-stage rule as
    # `experiments/select_chairs.pick_colourful`, and for the same reasons.
    #
    #   not brown      the point is that the model's output is brown and the
    #                  truth is not; a brown held-out chair proves nothing.
    #   colourful      a white chair turning brown is a weaker picture than a
    #                  magenta one turning brown.
    #   hue-spread     straight top-k returned three MAGENTA chairs, because
    #                  magenta is simply the most saturated thing in the
    #                  held-out split. Three different colours all coming back
    #                  brown is a strictly stronger result than one colour
    #                  three times.
    #
    # Qualifying on chroma has to be a real threshold and not just a sort key:
    # ranking by (frac_brown, -chroma) put a chroma=0.001 grey chair in the
    # pool, because its frac_brown happened to sort ahead of a vivid one's.
    chroma_cut = float(np.quantile([r[1]["chroma"] for r in rows], 0.75))
    pool = sorted([r for r in rows if r[1]["frac_brown"] < 0.10
                   and r[1]["chroma"] >= chroma_cut],
                  key=lambda r: -r[1]["chroma"])
    print(f"  pool: frac_brown < 0.10 and chroma >= {chroma_cut:.3f} (q0.75) "
          f"-> {len(pool)} of {len(rows)} held-out objects")
    if len(pool) < k:
        raise SystemExit(
            f"only {len(pool)} held-out objects qualify but {k} were asked for. "
            f"The held-out split may be browner than this experiment assumes — "
            f"check reports/chair_colour_stats.csv rather than loosening this.")
    picked = [pool[0]]
    while len(picked) < min(k, len(pool)):
        chosen = [p[1]["hue_mean"] for p in picked]

        def hue_gap(r):
            d = [abs(r[1]["hue_mean"] - h) for h in chosen]
            return min(min(x, 360.0 - x) for x in d)
        rest = [r for r in pool if r[0] not in {p[0] for p in picked}]
        picked.append(max(rest, key=hue_gap))
    print(f"picked {len(picked)} held-out objects (least brown, most colourful):")
    for i, s in picked:
        print(f"  [{i:3d}] {s['name'][-12:]}  frac_brown {s['frac_brown']:.3f}  "
              f"chroma {s['chroma']:.3f}  hue {s['hue_mean']:6.1f}°")
    return [i for i, _ in picked], {i: s for i, s in picked}


@torch.no_grad()
def build(run_name: str, n_objects: int = 5, split: str = "test",
          select: str = "far-from-brown", sample_steps: int = None,
          seed: int = 0, image_size: int = 256, device=None,
          snapshot: str = None, out: Path = None,
          n_gaussians: int = None) -> Path:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, meta = load_run(run_name, device, snapshot=snapshot)
    if n_gaussians:
        print(f"  N override: {cfg.n_gaussians} -> {n_gaussians} (smoke run; the "
              f"figure for the writeup should use the run's own N)")
        cfg.n_gaussians = n_gaussians
    norm_stats = load_norm_stats(cfg.shard_path)
    sample_steps = sample_steps or cfg.sample_steps
    run_dir = Path(meta["run_dir"])

    if split == "test":
        indices, _pre = pick_test_objects(cfg, n_objects, select)
        train_names = None
    else:                       # sanity contrast: the objects it was fit on
        train_names = json.loads((run_dir / "train_subset.json").read_text())["names"]
        indices = list(range(min(n_objects, len(train_names))))
    ds = ChairPointDataset(cfg, split=split, augment=False, names=train_names)

    # The colour prior the model was TAUGHT — measured from the run's own
    # training objects, not assumed. If this run's subset were not actually
    # brown, the figure would say so instead of quietly claiming otherwise.
    taught = json.loads((run_dir / "train_subset.json").read_text())["names"]
    name_to_file = {f.stem: f for f in find_ply_files(cfg.data_dir)}
    taught_stats = [colour_stats(read_splat(name_to_file[n])) for n in taught]
    taught_brown = float(np.mean([s["frac_brown"] for s in taught_stats]))
    taught_hue = np.array([s["hue_mean"] for s in taught_stats])
    taught_rgb = np.array([[s["r"], s["g"], s["b"]] for s in taught_stats])
    print(f"\ntraining prior: {len(taught)} objects, mean frac_brown "
          f"{taught_brown:.3f}, hue {taught_hue.mean():.1f}° ± {taught_hue.std():.1f}°")

    rows = []
    print(f"\nsampling {len(indices)} objects ({sample_steps} DPM-Solver steps)...")
    for idx in indices:
        item = ds[idx]
        cond = item["cond"].to(device)
        offset = torch.tensor([item["n"]], dtype=torch.long, device=device)
        pred = diffusion_sample(model, cond, offset, T=cfg.T,
                                num_steps=sample_steps, seed=seed, device=device)
        cond_np = cond.cpu().numpy()
        tok_gt = to_tokens(cond_np, item["target"], cfg, norm_stats)
        tok_pred = to_tokens(cond_np, pred, cfg, norm_stats)
        rows.append(dict(
            name=item["name"], n=int(item["n"]),
            tok_input=geometry_only_tokens(cond_np, cfg, norm_stats),
            tok_pred=tok_pred, tok_gt=tok_gt,
            s_pred=stats_of_tokens(tok_pred, item["name"]),
            s_gt=stats_of_tokens(tok_gt, item["name"]),
            mse=float(torch.mean((pred - item["target"].to(device)) ** 2)),
        ))
        print(f"  {item['name'][-12:]}  GT brown {rows[-1]['s_gt']['frac_brown']:.3f} "
              f"-> MODEL brown {rows[-1]['s_pred']['frac_brown']:.3f}")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return draw(rows, run_name, cfg, taught, taught_brown, taught_hue, taught_rgb,
                split, sample_steps, image_size, device, meta, out)


def draw(rows, run_name, cfg, taught, taught_brown, taught_hue, taught_rgb,
         split, sample_steps, image_size, device, meta, out) -> Path:
    n = len(rows)
    # Renders on top (n rows x 3), the measurement underneath (1 row x 3).
    # One figure, because the picture and the number have to be read
    # together — split across two files, the picture travels alone.
    #
    # Two SUBFIGURES rather than one tall gridspec: the render rows want to
    # be packed tight, the stats row wants room for a polar axis and its
    # legend, and a single gridspec cannot give them different spacing —
    # the first attempt had the bottom render row's caption sitting on top
    # of the polar plot's title.
    fig = plt.figure(figsize=(10.5, 3.1 * n + 4.4))
    sub_img, sub_stat = fig.subfigures(
        2, 1, height_ratios=[3.1 * n, 4.4])
    # Reserve a CONSTANT strip in inches for the suptitle and the column
    # headings, not a constant fraction: the figure grows with `n`, so a
    # fixed `top=0.94` reserved three times as much space at n=6 as at n=2
    # and overlapped the heading at small n.
    img_h = 3.1 * n
    gs = sub_img.add_gridspec(n, 3, hspace=0.16, wspace=0.02,
                              top=1.0 - 1.05 / img_h, bottom=0.02)

    for r, row in enumerate(rows):
        for c, (key, title) in enumerate((
                ("tok_input", "INPUT\ngeometry only (colour withheld)"),
                ("tok_pred", "MODEL\nsampled from noise"),
                ("tok_gt", "GROUND TRUTH\nthe real colour"))):
            ax = sub_img.add_subplot(gs[r, c])
            ax.imshow(render_tokens(row[key], size=image_size, device=device))
            ax.axis("off")
            if r == 0:
                ax.set_title(title, fontsize=10)
            if c == 0:
                ax.text(0.0, -0.03, f"{row['name'][-12:]}   N={row['n']:,}",
                        transform=ax.transAxes, fontsize=8, va="top")
            else:
                # Mean-colour swatch plus the number, under each colour panel,
                # so the comparison does not depend on the reader's monitor.
                s = row["s_pred"] if c == 1 else row["s_gt"]
                ax.add_patch(plt.Rectangle(
                    (0.30, -0.075), 0.16, 0.05, transform=ax.transAxes,
                    facecolor=(s["r"], s["g"], s["b"]), edgecolor="black",
                    lw=0.6, clip_on=False))
                ax.text(0.50, -0.03, f"frac_brown {s['frac_brown']:.2f}",
                        transform=ax.transAxes, fontsize=8, va="top")

    pred_brown = np.array([r["s_pred"]["frac_brown"] for r in rows])
    gt_brown = np.array([r["s_gt"]["frac_brown"] for r in rows])
    pred_hue = np.array([r["s_pred"]["hue_mean"] for r in rows])
    gt_hue = np.array([r["s_gt"]["hue_mean"] for r in rows])
    labels = [r["name"][-6:] for r in rows]

    gs2 = sub_stat.add_gridspec(1, 3, wspace=0.34, top=0.86, bottom=0.26)

    # --- panel 1: the headline number, per object ---------------------------
    ax = sub_stat.add_subplot(gs2[0, 0])
    x = np.arange(len(rows))
    ax.bar(x - 0.2, gt_brown, 0.4, label="ground truth", color="tab:blue")
    ax.bar(x + 0.2, pred_brown, 0.4, label="model output", color="saddlebrown")
    ax.axhline(taught_brown, color="k", ls="--", lw=1.2,
               label=f"training prior ({taught_brown:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, fontsize=7, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("frac_brown (opacity-weighted)")
    ax.set_title("brownness: truth vs model", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    # --- panel 2: hue, on the circle it actually lives on --------------------
    ax = sub_stat.add_subplot(gs2[0, 1], projection="polar")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    band = np.deg2rad(np.linspace(*BROWN_HUE_DEG, 40))
    ax.fill_between(band, 0, 1, color="saddlebrown", alpha=0.25,
                    label=f"brown band {BROWN_HUE_DEG[0]:.0f}–{BROWN_HUE_DEG[1]:.0f}°")
    ax.plot(np.deg2rad(taught_hue), np.full(len(taught_hue), 0.55), "k^", ms=6,
            label="training objects")
    ax.plot(np.deg2rad(gt_hue), np.full(len(gt_hue), 0.78), "o", color="tab:blue",
            ms=7, label="held-out truth")
    ax.plot(np.deg2rad(pred_hue), np.full(len(pred_hue), 0.95), "s",
            color="saddlebrown", ms=7, label="model output")
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.05)
    ax.set_title("mean hue (circular)", fontsize=10, pad=16)
    ax.legend(fontsize=6, loc="lower center", bbox_to_anchor=(0.5, -0.28))

    # --- panel 3: the verdict, in words --------------------------------------
    ax = sub_stat.add_subplot(gs2[0, 2])
    ax.axis("off")
    # Spread across objects is what distinguishes "learned a geometry ->
    # appearance mapping" from "learned one constant colour". Both would put
    # the bars high; only the first varies with the input.
    spread = float(np.std(np.stack([[r["s_pred"]["r"], r["s_pred"]["g"],
                                     r["s_pred"]["b"]] for r in rows]), axis=0).mean())
    gt_spread = float(np.std(np.stack([[r["s_gt"]["r"], r["s_gt"]["g"],
                                        r["s_gt"]["b"]] for r in rows]), axis=0).mean())
    # The verdict is COMPUTED, not written in advance. This same script is
    # run against a control model trained on mixed-colour chairs, where the
    # honest answer is "no transfer" — a figure whose caption asserts the
    # conclusion regardless of the numbers is not evidence of anything.
    transferred = pred_brown.mean() > 0.5 > gt_brown.mean()
    verdict = ([
        "=> the model imposes the colour",
        "   prior it was trained on, onto",
        "   geometry it has never seen.",
    ] if transferred else [
        "=> NO brown transfer here. That is",
        "   the expected result when the",
        "   training set is not itself brown",
        f"   (prior frac_brown {taught_brown:.2f}).",
    ])
    if transferred:
        verdict.append("   Per-object spread is "
                       + ("PRESERVED:" if spread > 0.2 * gt_spread else "COLLAPSED:"))
        verdict.append("   a mapping, not one constant."
                       if spread > 0.2 * gt_spread else
                       "   one near-constant colour.")

    lines = [
        f"TRAINED ON {len(taught)} chairs",
        f"    mean frac_brown  {taught_brown:.3f}",
        f"    mean hue         {taught_hue.mean():.0f}° +/- {taught_hue.std():.0f}°",
        "",
        f"{len(rows)} HELD-OUT chairs, geometry only",
        f"    ground truth brown   {gt_brown.mean():.3f}",
        f"    model output brown   {pred_brown.mean():.3f}",
        "",
        "per-object colour spread",
        f"    ground truth   {gt_spread:.4f}",
        f"    model output   {spread:.4f}",
        "",
    ] + verdict
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=9,
            family="monospace", transform=ax.transAxes)
    ax.set_title("what this shows", fontsize=10)

    backend = "gsplat 3DGS rasterization" if gsplat_available() else \
        "POINT-CLOUD fallback (no GPU) — colour exact, shape indicative"
    fig.suptitle(
        f"{run_name} — colour transfer onto {split.upper()} geometry the model "
        f"has never seen\nstage {cfg.stage} ({', '.join(cfg.target_names)}), EMA "
        f"weights at step {meta['step']:,}, {sample_steps} DPM-Solver steps, "
        f"seed 0   ·   renderer: {backend}", fontsize=12, y=0.995)

    out = Path(out) if out else REPORT_DIR / f"colour_transfer_{run_name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # The numbers, machine-readable, next to the picture. A figure is not a
    # result until the number behind it can be quoted without re-reading it
    # off an axis.
    json_out = out.with_suffix(".json")
    json_out.write_text(json.dumps(dict(
        run=run_name, split=split, snapshot=meta["snapshot"], step=meta["step"],
        sample_steps=sample_steps,
        training_prior=dict(n=len(taught), frac_brown=taught_brown,
                            hue_mean=float(taught_hue.mean()),
                            hue_std=float(taught_hue.std())),
        held_out=dict(n=len(rows),
                      gt_frac_brown=float(gt_brown.mean()),
                      model_frac_brown=float(pred_brown.mean()),
                      gt_colour_spread=gt_spread, model_colour_spread=spread),
        objects=[dict(name=r["name"], mse=r["mse"],
                      gt=r["s_gt"], model=r["s_pred"]) for r in rows],
    ), indent=2))
    print(f"\nwrote {out}\nwrote {json_out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    ap.add_argument("--n-objects", type=int, default=5)
    ap.add_argument("--split", default="test", choices=["test", "train"],
                    help="'train' is the sanity contrast — the objects it WAS fit on")
    ap.add_argument("--select", default="far-from-brown",
                    choices=["far-from-brown", "first"])
    ap.add_argument("--sample-steps", type=int, default=None)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-gaussians", type=int, default=None,
                    help="subsample fewer points than the run trained at. For CPU "
                         "smoke-testing this script's plumbing only — a figure "
                         "made at reduced N under-represents the model")
    args = ap.parse_args()
    build(args.run_name, n_objects=args.n_objects, split=args.split,
          select=args.select, sample_steps=args.sample_steps,
          image_size=args.image_size, seed=args.seed, snapshot=args.snapshot,
          device=torch.device(args.device) if args.device else None,
          out=args.out, n_gaussians=args.n_gaussians)


if __name__ == "__main__":
    main()
