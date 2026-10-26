"""
experiments/ablation_compare.py — experiment C's figure: plain L2 against
GaussianCube's full loss, on the SAME 100 objects.

    python experiments/ablation_compare.py stage1_100obj stage1_100obj_full
    python experiments/ablation_compare.py a b --labels "L2 only" "full loss"

**The one thing this figure must not do is plot both runs' total loss on
one axis.** They optimise different objectives:

    L2 run     loss = L_simple                       (MSE, squared std units)
    full run   loss = L_simple + L_vb + 10*L1 + 10*LPIPS
                            (bits)   (rendered-pixel units)

so the full run's total is larger by construction and always will be. Two
curves like that on one axis look like a result and are an artefact of the
axis. Only quantities that mean the same thing in both runs are drawn
together, and each panel says which kind it is:

  COMPARABLE       sampled-from-noise MSE — full DPM-Solver sampling against
                   ground truth. Objective-independent: it measures the task,
                   not the loss. **This is the ablation's answer.**
  COMPARABLE       L_simple (`loss_mse`) alone — present and identically
                   defined in both runs, though the full run is not solely
                   optimising it, so a tie here is not a null result.
  COMPARABLE       per-channel r/g/b error, adaLN gate, final held-out test
                   loss, and the rendered-image metrics.
  NOT COMPARABLE   total loss, VB, L_image — drawn per-run, never overlaid.

A one-seed-each comparison of two training runs is weak evidence in
general. It is stated on the figure rather than left for the reader to
remember.
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
from train.plot_run import _series, read_stats, running_median

REPORT_DIR = _REPO_ROOT / "reports"
COLOURS = ["tab:blue", "tab:red", "tab:green", "tab:purple"]


def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def load(run_name: str, run_root: str = "runs") -> dict:
    rows, summaries = read_stats(run_name, run_root)
    if not rows:
        raise SystemExit(f"{run_name}: no step rows logged yet")
    run_dir = _REPO_ROOT / run_root / run_name
    subset = _read_json(run_dir / "train_subset.json")
    return dict(
        name=run_name, rows=rows, summaries=summaries,
        cfg=_read_json(run_dir / "config.json") or {},
        test=_read_json(run_dir / "test_results.json"),
        image=image_metrics_of(run_dir),
        names=set(subset["names"]) if subset else None,
        merged={k: v for s in summaries for k, v in s.items()},
    )


def check_comparable(runs: list):
    """Refuse to draw a comparison of two runs that differ in more than the
    thing being ablated.

    An ablation is only an ablation if ONE thing changed. These runs are
    launched hours apart from a shell script; a mistyped `--n-gaussians` or
    a different object subset produces a figure that looks exactly like a
    real result and means nothing. Checked here, loudly, rather than trusted.
    """
    print("\ncomparability check:")
    base = runs[0]
    problems = []
    for other in runs[1:]:
        if base["names"] is not None and other["names"] is not None:
            if base["names"] != other["names"]:
                shared = len(base["names"] & other["names"])
                problems.append(
                    f"{base['name']} and {other['name']} trained on DIFFERENT "
                    f"objects ({shared} shared of {len(base['names'])}/"
                    f"{len(other['names'])})")
            else:
                print(f"  OK    same {len(base['names'])} training objects")
        elif (base["names"] is None) != (other["names"] is None):
            problems.append(f"one run used a subset and the other the full split")

        for field, label in (("n_gaussians", "N"), ("batch_size", "batch"),
                             ("lr", "lr"), ("stage", "stage"),
                             ("warmup_steps", "warmup"), ("seed", "seed")):
            a, b = base["cfg"].get(field), other["cfg"].get(field)
            if a is not None and b is not None and a != b:
                problems.append(f"{label} differs: {base['name']}={a} vs "
                                f"{other['name']}={b}")
    for field, label in (("n_gaussians", "N"), ("batch_size", "batch"),
                         ("lr", "lr"), ("stage", "stage")):
        vals = {r["cfg"].get(field) for r in runs}
        if len(vals) == 1 and None not in vals:
            print(f"  OK    same {label} = {vals.pop()}")

    steps = [r["rows"][-1]["step"] for r in runs]
    if max(steps) - min(steps) > 0.02 * max(steps):
        problems.append(f"runs reached different steps: {steps} — the shorter "
                         f"one is not finished, or they were configured "
                         f"differently")
    else:
        print(f"  OK    both reached ~step {max(steps):,}")

    for p in problems:
        print(f"  WARN  {p}")
    if problems:
        print("  -> the figure will carry these warnings; an ablation where "
              "more than one thing changed does not isolate anything")
    return problems


# --------------------------------------------------------------- panels
def _panel_sample_mse(ax, runs, labels):
    """The headline: sampling from pure noise, scored against ground truth.

    Objective-independent, so it is the only loss-like curve on which the
    two runs can be compared directly."""
    for r, lab, col in zip(runs, labels, COLOURS):
        s, v = _series(r["rows"], "sample_mse_train")
        if len(s):
            ax.plot(s, v, "o-", ms=5, lw=1.6, color=col, label=lab)
            ax.annotate(f"{v[-1]:.3f}", (s[-1], v[-1]), fontsize=8, color=col,
                        xytext=(4, 0), textcoords="offset points", va="center")
    ax.set(xlabel="step", ylabel="sampled-from-noise MSE (train objects)")
    ax.set_title("THE ABLATION ANSWER — sampled-from-noise MSE\n"
                 "measures the task, not the objective, so it is comparable\n"
                 "across two runs optimising different losses", fontsize=10)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def lsimple_series(run: dict) -> tuple:
    """(steps, values, note) for L_simple, whatever the run called it.

    `loss_mse` was added to the logger alongside the hybrid loss. A run
    predating it — `stage1_100obj` is exactly that — logs only `loss`, which
    for a pure-L2 run IS L_simple. Without this fallback the older arm of
    the ablation simply vanishes from the panel, leaving one curve that
    looks like a run with nothing to compare against rather than like a
    missing key.

    The fallback is only safe when the run has no other loss terms, so that
    is verified against the run's own logs rather than assumed.
    """
    s, v = _series(run["rows"], "loss_mse")
    if len(s):
        return s, v, ""
    s, v = _series(run["rows"], "loss")
    if not len(s):
        return s, v, ""
    extras = [k for k in ("loss_vb", "render_l1", "render_lpips")
              if len(_series(run["rows"], k)[0])
              and np.any(_series(run["rows"], k)[1] > 0)]
    if extras:
        # `loss` here is a SUM including terms in other units — not L_simple.
        # Drawing it on this axis would be the exact mistake this module's
        # docstring says not to make.
        print(f"  {run['name']}: no loss_mse logged and `loss` includes "
              f"{extras} — cannot recover L_simple, omitting from that panel")
        return np.array([]), np.array([]), ""
    return s, v, " (from `loss`: pre-dates loss_mse, and this run's loss IS L_simple)"


def _panel_lsimple(ax, runs, labels):
    notes = []
    for r, lab, col in zip(runs, labels, COLOURS):
        s, v, note = lsimple_series(r)
        if len(s):
            ax.plot(s, v, lw=0.5, alpha=0.18, color=col)
            ax.plot(s, running_median(v), lw=1.8, color=col, label=lab + note)
            if note:
                notes.append(lab)
    ax.set(yscale="log", xlabel="step", ylabel="L_simple (MSE term)")
    ax.set_title("L_simple, the term BOTH runs contain\n"
                 "the full-loss run is not solely optimising it, so a tie\n"
                 "here is not a null result", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def _panel_extra_terms(ax, runs, labels):
    """The full run's own terms, drawn alone. Never overlaid on the L2 run's
    curve — the L2 run does not have them, and an empty series next to a
    populated one reads as zero rather than as absent."""
    drew = False
    for r, lab, col in zip(runs, labels, COLOURS):
        for key, style, tag in (("loss_vb", "--", "VB (bits)"),
                                ("render_l1", "-.", "render L1"),
                                ("render_lpips", ":", "render LPIPS")):
            s, v = _series(r["rows"], key)
            live = v > 0 if len(s) else np.array([], bool)
            if live.any():
                ax.plot(s[live], running_median(v[live]), style, lw=1.4,
                        color=col, label=f"{lab}: {tag}")
                drew = True
    if drew:
        ax.set_yscale("log")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "neither run has extra loss terms", ha="center",
                va="center", color="grey", transform=ax.transAxes)
    ax.set(xlabel="step", ylabel="term value")
    ax.set_title("the terms only the FULL loss has\n"
                 "NOT comparable to anything in the L2 run — shown so the\n"
                 "difference between the two objectives is visible", fontsize=10)
    ax.grid(alpha=0.3)


def _panel_channels(ax, runs, labels):
    for r, lab, col in zip(runs, labels, COLOURS):
        for ch, style in (("r", "-"), ("g", "--"), ("b", ":")):
            s, v = _series(r["rows"], f"loss_{ch}")
            if len(s):
                ax.plot(s, running_median(v), style, lw=1.3, color=col,
                        label=f"{lab}: {ch}")
    if ax.get_legend_handles_labels()[0]:
        ax.set_yscale("log")
        ax.legend(fontsize=6, ncol=2)
    ax.set(xlabel="step", ylabel="per-channel MSE (running median)")
    ax.set_title("per-channel colour error\n"
                 "does the image-space term buy anything on RGB specifically?",
                 fontsize=10)
    ax.grid(alpha=0.3)


def _panel_gate(ax, runs, labels):
    for r, lab, col in zip(runs, labels, COLOURS):
        s, v = _series(r["rows"], "gate_mean")
        if len(s):
            ax.plot(s, v, lw=1.5, color=col, label=f"{lab}: gate")
        gs, gv = _series(r["rows"], "grad_norm")
        if len(gs):
            ax.plot(gs, running_median(gv) / 10.0, ":", lw=1.0, color=col,
                    alpha=0.7, label=f"{lab}: grad/10")
    ax.set(xlabel="step", ylabel="adaLN gate_mean   ·   grad norm / 10")
    ax.set_title("timestep conditioning and gradient scale\n"
                 "gates must grow from 0; a run whose gates stall is not\n"
                 "training on t at all (LOGGING.md panel 2)", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def _panel_table(ax, runs, labels, problems):
    """The final numbers, side by side. A curve panel shows a trend; the
    thing anyone quotes from an ablation is this table."""
    ax.axis("off")
    metrics = [
        ("test L_simple ↓", lambda r: (r["test"] or {}).get("test_loss")),
        ("test sample MSE ↓", lambda r: (r["test"] or {}).get("test_sample_mse_test")),
        ("train sample MSE ↓", lambda r: (r["test"] or {}).get("test_sample_mse_train")),
        ("PSNR ↑", lambda r: ((r["image"] or {}).get("model") or {}).get("psnr")),
        ("SSIM ↑", lambda r: ((r["image"] or {}).get("model") or {}).get("ssim")),
        ("LPIPS ↓", lambda r: ((r["image"] or {}).get("model") or {}).get("lpips")),
        ("FID ↓", lambda r: ((r["image"] or {}).get("model") or {}).get("fid")),
        ("KID ↓", lambda r: ((r["image"] or {}).get("model") or {}).get("kid")),
    ]
    cells, row_labels = [], []
    for label, fn in metrics:
        vals = [fn(r) for r in runs]
        if all(v is None for v in vals):
            continue
        row_labels.append(label)
        cells.append(["—" if v is None else
                      (f"{v:.5f}" if abs(v) < 0.01 else f"{v:.4f}")
                      for v in vals])

    if not cells:
        ax.text(0.5, 0.6, "no test_results.json / image_metrics.json yet\n\n"
                          "python eval/test_eval.py <run>\n"
                          "python eval/image_eval.py <run>",
                ha="center", va="center", color="grey", fontsize=10,
                transform=ax.transAxes)
    else:
        tbl = ax.table(cellText=cells, rowLabels=row_labels, colLabels=labels,
                       cellLoc="center", rowLoc="right", loc="upper center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.5)
        # Highlight the better value in each row. Direction comes from the
        # arrow in the label, so adding a metric cannot silently get it wrong.
        for i, (label, _fn) in enumerate([m for m in metrics
                                          if m[0] in row_labels]):
            vals = [None if c == "—" else float(c) for c in cells[i]]
            good = [v for v in vals if v is not None]
            if len(good) < 2:
                continue
            best = min(good) if "↓" in label else max(good)
            for j, v in enumerate(vals):
                if v == best:
                    tbl[(i + 1, j)].set_facecolor("#e6f4ea")

    note = ("n=1 run per arm, one seed each — a difference smaller than the\n"
            "seed-to-seed spread is not a finding. Green = better value only.")
    if problems:
        note = "NOT A CLEAN ABLATION:\n" + "\n".join(
            f"  {p}" for p in problems) + "\n\n" + note
    ax.text(0.5, 0.02, note, ha="center", va="bottom", fontsize=8,
            color="tab:red" if problems else "grey", transform=ax.transAxes)
    ax.set_title("final numbers (held-out test set)", fontsize=10)


def build(run_names: list, labels: list = None, out: Path = None,
          run_root: str = "runs") -> Path:
    runs = [load(n, run_root) for n in run_names]
    labels = labels or [r["name"] for r in runs]
    if len(labels) != len(runs):
        raise SystemExit(f"{len(labels)} labels for {len(runs)} runs")
    problems = check_comparable(runs)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10.5))
    _panel_sample_mse(axes[0, 0], runs, labels)
    _panel_lsimple(axes[0, 1], runs, labels)
    _panel_extra_terms(axes[0, 2], runs, labels)
    _panel_channels(axes[1, 0], runs, labels)
    _panel_gate(axes[1, 1], runs, labels)
    _panel_table(axes[1, 2], runs, labels, problems)

    cfg = runs[0]["cfg"]
    n_obj = len(runs[0]["names"]) if runs[0]["names"] else "all"
    # Only claim "the SAME objects" when the check actually confirmed it.
    # A title that asserts the experimental control while the panel below
    # reports it was violated is worse than no title.
    same = all("objects" not in p and "subset" not in p for p in problems)
    scope = (f"the SAME {n_obj} objects" if same
             else f"DIFFERENT object sets — see the warnings")
    fig.suptitle(
        f"loss ablation on {scope}   ·   "
        f"{'  vs  '.join(labels)}   ·   stage {cfg.get('stage')}, "
        f"N={cfg.get('n_gaussians')}, batch {cfg.get('batch_size')}, "
        f"lr {cfg.get('lr')}, {runs[0]['rows'][-1]['step']:,} steps",
        fontsize=13)
    plt.tight_layout()

    out = Path(out) if out else REPORT_DIR / f"ablation_{'_vs_'.join(run_names)}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="two or more run names")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="display names, in the same order as `runs`")
    ap.add_argument("--out", default=None)
    ap.add_argument("--run-root", default="runs")
    args = ap.parse_args()
    if len(args.runs) < 2:
        raise SystemExit("an ablation needs at least two runs")
    build(args.runs, args.labels, args.out, args.run_root)


if __name__ == "__main__":
    main()
