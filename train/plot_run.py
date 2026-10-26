"""
train/plot_run.py — the figure for a training run.

Reads `runs/<name>/stats.jsonl` (what `train/logging_utils.StageLogger`
writes unconditionally, W&B or not) plus, if they exist,
`config.json` and `image_metrics.json`, and produces a 6-panel PNG. Works
on a finished run or a live one — it only ever reads, so it is safe to
call against an in-flight training process, same as `watch_run.py`.

`watch_run.py` is the live *text* view for watching a run tick over; this
is the figure you keep and put in the talk. Both read the same file.

    python train/plot_run.py stage3_full
    python train/plot_run.py stage3_full --out reports/curves_full.png

Panels:
  1. total loss, its MSE and VB halves, sampled-from-noise MSE, ★ test
  2. loss by timestep bin, as curves       — LOGGING.md panel 1
  3. loss by timestep bin, as a heatmap    — the same data as a shape:
                                             how the difficulty profile
                                             across noise levels MOVES
  4. adaLN gate_mean + grad_norm           — LOGGING.md panel 2, the
                                             "looks like slow training"
                                             failure mode
  5. per-channel loss by real channel name, coloured by family
  6. the final image metrics as a table    — PSNR/SSIM/LPIPS/FID/KID
                                             against the mean-fill
                                             baseline and the oracle
"""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")  # CLAUDE.md: no display on this node
import matplotlib.pyplot as plt
import numpy as np

from eval.load_run import image_metrics_of

REPORT_DIR = _REPO_ROOT / "reports"

# Colour by channel FAMILY, not by index: the question a per-channel panel
# answers at stage 3 is "is one attribute group lagging?", and three
# arbitrary colours per group make that harder to see, not easier.
FAMILY_COLOURS = {
    "log_sx": "tab:orange", "log_sy": "tab:orange", "log_sz": "tab:orange",
    "opacity": "tab:purple",
    "r": "tab:red", "g": "tab:green", "b": "tab:blue",
    "x": "tab:grey", "y": "tab:grey", "z": "tab:grey",
    "qw": "tab:brown", "qx": "tab:brown", "qy": "tab:brown", "qz": "tab:brown",
}
FAMILY_STYLE = {"log_sx": "-", "log_sy": "--", "log_sz": ":",
                "qw": "-", "qx": "--", "qy": ":", "qz": "-."}


def read_stats(run_name: str, run_root: str = "runs") -> tuple:
    """-> (step rows, summary rows). Tolerates a torn final line: the
    trainer may be mid-write when we read (same reasoning as
    `watch_run.py:read_stats`)."""
    path = _REPO_ROOT / run_root / run_name / "stats.jsonl"
    if not path.exists():
        raise SystemExit(f"no stats at {path} — has the run started?")
    rows, summaries = [], []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn final line, or a partial flush
        (summaries if row.get("summary") else rows).append(row)
    return rows, summaries


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _series(rows: list, key: str) -> tuple:
    """(steps, values) for every row that has `key`. Metrics are logged at
    different cadences (loss every log_every, sample MSE every
    render_every), so each series carries its own x-axis rather than
    sharing one."""
    pts = [(r["step"], r[key]) for r in rows if key in r and r[key] is not None]
    if not pts:
        return np.array([]), np.array([])
    s, v = zip(*pts)
    return np.asarray(s), np.asarray(v, dtype=float)


def running_median(v: np.ndarray, frac: int = 50) -> np.ndarray:
    """Per-step loss is dominated by which `t` each step happened to draw —
    high-t steps are intrinsically harder — so the raw trace is noisy for
    reasons that are not instability. The median overlay is the trend."""
    win = max(1, len(v) // frac)
    return np.array([np.median(v[max(0, i - win):i + 1]) for i in range(len(v))])


def _bin_keys(rows: list) -> list:
    keys = {k for r in rows for k in r if k.startswith("loss_t_bin_")}
    return sorted(keys, key=lambda k: int(k.rsplit("_", 1)[1]))


# ----------------------------------------------------------------- panels
def _panel_loss(ax, rows, summaries):
    steps, loss = _series(rows, "loss")
    ax.plot(steps, loss, lw=0.5, alpha=0.25, color="C0",
            label="total (per-step, noisy: random t)")
    ax.plot(steps, running_median(loss), lw=1.8, color="C0", label="total (median)")

    # The two halves of GaussianCube's hybrid loss, drawn apart. They are in
    # DIFFERENT UNITS -- the MSE is squared standardized units, the VB term is
    # bits -- so their sum is a training signal, not a quantity, and the split
    # is what makes the curve readable. The VB trace is also where the t=0
    # decoder-NLL spikes live; seeing them here instead of in the total is the
    # difference between "expected" and "the run is unstable".
    ms, mv = _series(rows, "loss_mse")
    if len(ms):
        ax.plot(ms, running_median(mv), lw=1.6, color="C2",
                label="MSE term (median) — L_simple")
    vs, vv = _series(rows, "loss_vb")
    if len(vs) and np.any(vv > 0):
        ax.plot(vs, running_median(vv), lw=1.2, color="C1", alpha=0.85,
                label="VB term (median, bits)")

    # GaussianCube's image-space term. Drawn as its two raw halves, NOT as the
    # weighted sum that enters the loss: the weights are 10 and 10, so the sum
    # would sit an order of magnitude above the MSE trace and squash it, while
    # the raw L1 and LPIPS are the numbers that mean something (an L1 in
    # rendered-pixel units, an LPIPS in perceptual ones).
    for key, colour, label in (("render_l1", "C6", "render L1 (unweighted)"),
                               ("render_lpips", "C9", "render LPIPS (unweighted)")):
        rs, rv = _series(rows, key)
        if len(rs) and np.any(rv > 0):
            live = rv > 0        # before the ramp the term is exactly 0
            ax.plot(rs[live], running_median(rv[live]), lw=1.3, color=colour,
                    ls="-.", label=label)
    ws, wv = _series(rows, "render_weight")
    if len(ws) and np.any(wv > 0):
        first_on = ws[np.argmax(wv > 0)]
        ax.axvline(first_on, color="C6", ls=":", lw=1.2, alpha=0.7)
        ax.annotate("L_image on", xy=(first_on, ax.get_ylim()[1]), fontsize=7,
                    color="C6", rotation=90, va="top", ha="right")

    ss, sv = _series(rows, "sample_mse_train")
    if len(ss):
        ax.plot(ss, sv, "s--", ms=5, lw=1.2, color="C5",
                label="sampled-from-noise MSE (train)")
    # There is deliberately NO val curve: no validation set exists (nothing
    # during training selects a model). The single held-out number is the
    # final test evaluation, drawn as a marker at the end if it has run.
    final = {k: v for s in summaries for k, v in s.items()}
    if "test_loss" in final and len(steps):
        ax.plot([steps[-1]], [final["test_loss"]], "*", ms=16, color="C3",
                label=f"FINAL test L_simple = {final['test_loss']:.4f}")
    ax.set(yscale="log", xlabel="step", ylabel="loss")
    ax.set_title("training loss, split into its two terms\n"
                 "(no validation set — held-out data is touched once: the ★)",
                 fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def _panel_bins(ax, rows):
    keys = _bin_keys(rows)
    if keys:
        cmap = plt.get_cmap("viridis")
        for i, k in enumerate(keys):
            bs, bv = _series(rows, k)
            ax.plot(bs, bv, lw=1.3, color=cmap(i / max(1, len(keys) - 1)),
                    label=f"t bin {i}")
        ax.set_yscale("log")
        ax.legend(fontsize=6, ncol=2)
    ax.set(xlabel="step", ylabel="mean MSE in bin (cumulative)")
    ax.set_title("MSE by timestep bucket (dark = low t, light = high t)\n"
                 "should RISE with t; flat-near-zero at high t = the collapse\n"
                 "NORMALIZATION.md §4 watches for", fontsize=10)
    ax.grid(alpha=0.3)


def _panel_bin_heatmap(ax, rows, T=1000):
    """The same t-bin data as an image. The curves panel answers "is bin 7
    falling?"; this one answers "is the SHAPE across noise levels right, and
    is it changing?" — which is the actual diagnostic, and is very hard to
    read off ten overlapping log-scale lines."""
    keys = _bin_keys(rows)
    if not keys:
        ax.axis("off")
        return
    steps = sorted({r["step"] for r in rows if any(k in r for k in keys)})
    if len(steps) < 2:
        ax.axis("off")
        return
    grid = np.full((len(keys), len(steps)), np.nan)
    step_pos = {s: i for i, s in enumerate(steps)}
    for r in rows:
        if r["step"] not in step_pos:
            continue
        for b, k in enumerate(keys):
            if k in r and r[k] is not None:
                grid[b, step_pos[r["step"]]] = r[k]
    with np.errstate(invalid="ignore", divide="ignore"):
        img = np.log10(grid)
    im = ax.imshow(img, aspect="auto", origin="lower", cmap="magma",
                   extent=[steps[0], steps[-1], 0, T])
    plt.colorbar(im, ax=ax, label="log10(mean MSE)")
    ax.set(xlabel="step", ylabel="diffusion timestep t")
    ax.set_title("MSE by (step, noise level)\nvertical stripes = a noise regime "
                 "training differently", fontsize=10)


def _panel_gate(ax, rows):
    gs, gv = _series(rows, "gate_mean")
    if len(gs):
        ax.plot(gs, gv, lw=1.4, color="C4", label="adaLN gate_mean")
    ax.set(xlabel="step", ylabel="gate_mean")
    ax.yaxis.label.set_color("C4")
    ax.set_title("adaLN gate_mean (must grow from 0) + the GRADIENT BUDGET\n"
                 "dashed = each loss term's own gradient norm; whichever is\n"
                 "highest is the term actually driving training", fontsize=10)
    ax.grid(alpha=0.3)
    ns, nv = _series(rows, "grad_norm")
    if len(ns):
        ax2 = ax.twinx()
        ax2.plot(ns, running_median(nv), lw=1.0, color="C1", alpha=0.7,
                 label="total (pre-clip)")
        ax2.set_ylabel("grad norm", color="C1")
        ax2.set_yscale("log")
        # Everything above this line is clipped away. GaussianCube trains with
        # exactly this too (`clip_grad_norm_(params, 1.0)`).
        ax2.axhline(1.0, color="C1", ls=":", lw=1.0, alpha=0.6)

        # THE GRADIENT BUDGET: how much of the gradient each loss term
        # contributes, measured separately (train/logging_utils.
        # per_term_grad_norms). This is the only thing that answers "which term
        # is driving training" -- loss values cannot, because a small term with
        # a steep slope beats a large flat one, and clip_grad_norm_ rescales
        # the SUM so the bigger-gradient term keeps proportionally more say.
        for key, colour, label in (("grad_norm_mse", "C2", "MSE (L2) term"),
                                   ("grad_norm_vb", "C0", "VB term"),
                                   ("grad_norm_render_total", "C6", "L_image term")):
            gs2, gv2 = _series(rows, key)
            live = gv2 > 0 if len(gs2) else np.array([], bool)
            if live.any():
                ax2.plot(gs2[live], gv2[live], lw=1.3, ls="--", color=colour,
                         label=label)
        if ax2.get_legend_handles_labels()[0]:
            ax2.legend(fontsize=6, loc="upper right")


def _panel_channels(ax, rows, target_names):
    if not target_names:
        target_names = [k[len("loss_"):] for k in rows[-1]
                        if k.startswith("loss_") and k[len("loss_"):] in FAMILY_COLOURS]
    plotted = 0
    for name in target_names:
        cs, cv = _series(rows, f"loss_{name}")
        if not len(cs):
            continue
        ax.plot(cs, running_median(cv), lw=1.4,
                color=FAMILY_COLOURS.get(name, "black"),
                ls=FAMILY_STYLE.get(name, "-"), label=name)
        plotted += 1
    if plotted:
        ax.set_yscale("log")
        ax.legend(fontsize=7, ncol=2)
    ax.set(xlabel="step", ylabel="per-channel MSE (running median)")
    ax.set_title("per-channel loss, by real channel name\n(one channel lagging = "
                 "a normalization problem, not a capacity one)", fontsize=10)
    ax.grid(alpha=0.3)


def _panel_metrics(ax, metrics):
    """The final image metrics, drawn as a table so the figure is
    self-contained — a curves PNG that needs a second file open beside it to
    mean anything is not a figure you can put in a talk."""
    ax.axis("off")
    if not metrics:
        ax.text(0.5, 0.5, "image metrics not run yet\n\n"
                          "python eval/image_eval.py <run-name>",
                ha="center", va="center", fontsize=10, color="grey",
                transform=ax.transAxes)
        ax.set_title("PSNR / SSIM / LPIPS / FID / KID", fontsize=10)
        return

    meta = metrics.get("_meta", {})
    cols = ["PSNR↑", "SSIM↑", "LPIPS↓", "FID↓", "KID↓"]
    order = [("oracle", "oracle (ceiling)"), ("mean_fill", "mean-fill baseline"),
             ("model", "MODEL")]
    cells, labels = [], []
    for tag, label in order:
        r = metrics.get(tag)
        if not r:
            continue
        kid = r.get("kid")
        cells.append([f"{r['psnr']:.2f}", f"{r['ssim']:.4f}", f"{r['lpips']:.4f}",
                      f"{r.get('fid', float('nan')):.2f}",
                      "n/a" if kid is None else f"{kid:.5f}"])
        labels.append(label)

    tbl = ax.table(cellText=cells, rowLabels=labels, colLabels=cols,
                   cellLoc="center", rowLoc="right", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.7)
    for j in range(len(cols)):                       # highlight the model row
        tbl[(len(labels), j)].set_facecolor("#e8f0fe")
    note = (f"{meta.get('n_objects', '?')} test objects x "
            f"{meta.get('n_views', '?')} azimuths, {meta.get('image_size', '?')}px, "
            f"renderer={meta.get('render_backend', '?')}")
    ax.set_title(f"held-out image metrics\n{note}", fontsize=10)
    if meta.get("render_backend") != "gsplat":
        ax.text(0.5, 0.02, "POINT-CLOUD fallback renderer — pipeline check, not results",
                ha="center", fontsize=8, color="tab:red", transform=ax.transAxes)


# ----------------------------------------------------------------- figure
def plot_run(run_name: str, out_path: Path = None, run_root: str = "runs") -> Path:
    rows, summaries = read_stats(run_name, run_root)
    if not rows:
        raise SystemExit(f"{run_name}: no step rows logged yet")
    run_dir = _REPO_ROOT / run_root / run_name
    cfg = _read_json(run_dir / "config.json") or {}
    # Resolves image_metrics.json OR image_metrics_<sampler>.json — the
    # in-loop and standalone evaluations write different names, and looking
    # for only one silently blanks this panel (eval/load_run.py).
    metrics = image_metrics_of(run_dir)
    target_names = (metrics or {}).get("_meta", {}).get("target_channels")

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    _panel_loss(axes[0, 0], rows, summaries)
    _panel_bins(axes[0, 1], rows)
    _panel_bin_heatmap(axes[0, 2], rows, T=cfg.get("T", 1000))
    _panel_gate(axes[1, 0], rows)
    _panel_channels(axes[1, 1], rows, target_names)
    _panel_metrics(axes[1, 2], metrics)

    # ------------------------------------------------------------- caption
    last = rows[-1]
    merged = {k: v for s in summaries for k, v in s.items()}
    bits = [run_name, f"step {last['step']}"]
    if cfg:
        bits.append(f"stage {cfg.get('stage')}")
        if cfg.get("learn_sigma"):
            bits.append("hybrid loss (MSE+VB)")
        else:
            bits.append("L_simple only")
    if "n_train_objects" in merged:
        bits.append(f"{merged['n_train_objects']} train objects")
    if cfg.get("n_gaussians"):
        bits.append(f"N={cfg['n_gaussians']}")
    if "epoch" in last:
        bits.append(f"epoch {last['epoch']:.1f}")
    if "gpu_mem_gb" in last:
        bits.append(f"{last['gpu_mem_gb']:.1f} GB")
    if "step_time" in last:
        bits.append(f"{last['step_time'] * 1000:.0f} ms/step")
    plt.suptitle("   ·   ".join(str(b) for b in bits), fontsize=13)
    plt.tight_layout()

    out_path = Path(out_path) if out_path else REPORT_DIR / f"curves_{run_name}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    ap.add_argument("--out", default=None)
    ap.add_argument("--run-root", default="runs")
    args = ap.parse_args()
    print(f"wrote {plot_run(args.run_name, args.out, args.run_root)}")


if __name__ == "__main__":
    main()
