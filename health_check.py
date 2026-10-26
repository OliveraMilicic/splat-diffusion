"""
health_check.py — is this run going well? One screen, read-only, safe on a
live run.

`watch_run.py` is the live tail; `train/plot_run.py` is the figure. This is
the thing to run when the question is specifically "should I be worried",
because it prints the handful of numbers whose SHAPE answers that, each
next to what it is supposed to look like. A loss going down is not by
itself evidence that a diffusion model is training correctly.

    python health_check.py stage3_full
"""

import argparse
import json
from pathlib import Path

import numpy as np

TARGET_CHANNELS = ("log_sx", "log_sy", "log_sz", "opacity", "r", "g", "b",
                   "x", "y", "z", "qw", "qx", "qy", "qz")


def load(run_name: str, run_root: str = "runs"):
    path = Path(run_root) / run_name / "stats.jsonl"
    if not path.exists():
        raise SystemExit(f"no stats at {path} — has the run started?")
    rows, summaries = [], []
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue          # torn final line: the trainer may be mid-write
        (summaries if r.get("summary") else rows).append(r)
    if not rows:
        raise SystemExit(f"{run_name}: no step rows logged yet")
    return rows, summaries


def med(rows, key, lo, hi=10 ** 12):
    v = [r[key] for r in rows if lo <= r["step"] < hi and r.get(key) is not None]
    return float(np.median(v)) if v else float("nan")


def verdict(ok: bool, good: str, bad: str) -> str:
    return f"  OK   {good}" if ok else f"  WATCH {bad}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    ap.add_argument("--run-root", default="runs")
    args = ap.parse_args()
    rows, summaries = load(args.run_name, args.run_root)
    last = rows[-1]
    step = last["step"]

    print(f"\n{'=' * 72}\n{args.run_name} — step {step:,}  epoch {last.get('epoch', 0):.1f}"
          f"   {last.get('step_time', 0) * 1000:.0f} ms/step"
          f"   {last.get('gpu_mem_gb', 0):.1f} GB peak\n{'=' * 72}")

    # --- 1. did anything break -------------------------------------------------
    n_skip, n_oom = last.get("n_skipped", 0), last.get("n_oom", 0)
    print(f"\n[1] failures      non-finite={n_skip}  OOM={n_oom}")
    print(verdict(n_skip == 0 and n_oom == 0, "nothing dropped",
                  "batches are being dropped — see the log for which"))

    # --- 2. is the loss falling ------------------------------------------------
    print("\n[2] loss trend (median per window)")
    edges = [0, 500, 1000, 2000, 4000, 8000, 12000, 20000, 30000, 45000, 10 ** 12]
    print(f"    {'steps':>16} {'mse':>9} {'total':>9} {'gate':>7} {'grad':>7}")
    for lo, hi in zip(edges, edges[1:]):
        if not any(lo <= r["step"] < hi for r in rows):
            continue
        print(f"    {lo:7,}-{min(hi - 1, step):<8,} {med(rows, 'loss_mse', lo, hi):9.4f} "
              f"{med(rows, 'loss', lo, hi):9.4f} {med(rows, 'gate_mean', lo, hi):7.3f} "
              f"{med(rows, 'grad_norm', lo, hi):7.3f}")
    early = med(rows, "loss_mse", 0, max(500, step // 10))
    recent = med(rows, "loss_mse", max(0, step - 1000))
    print(verdict(recent < early, f"mse {early:.4f} -> {recent:.4f}",
                  f"mse NOT falling ({early:.4f} -> {recent:.4f})"))

    # --- 3. adaLN gates --------------------------------------------------------
    # LOGGING.md panel 2: gates stuck near 0 means the model is fitting through
    # the conditioning path and ignoring t entirely. It looks like slow
    # training, not like a bug.
    gate = med(rows, "gate_mean", max(0, step - 1000))
    print(f"\n[3] adaLN gate_mean = {gate:.3f}   (starts at 0, must GROW)")
    print(verdict(gate > 0.01, "timestep conditioning is alive",
                  "gates near zero — the model may be ignoring t"))

    # --- 4. loss by noise level ------------------------------------------------
    # The single most important panel: difficulty must rise with t. A flat or
    # near-zero high-t tail is the collapse NORMALIZATION.md §4 watches for.
    bins = sorted([k for k in last if k.startswith("loss_t_bin_")],
                  key=lambda k: int(k.split("_")[-1]))
    if bins:
        vals = [last[k] for k in bins]
        print(f"\n[4] MSE by timestep bin (low t -> high t), must RISE")
        print("    " + "  ".join(f"{v:.3f}" for v in vals))
        rising = vals[-1] > vals[0]
        print(verdict(rising, f"rises {vals[0]:.3f} -> {vals[-1]:.3f}",
                      f"FLAT/falling {vals[0]:.3f} -> {vals[-1]:.3f} — high-t regime "
                      f"may be collapsing to the mean"))

    # --- 5. per-channel balance ------------------------------------------------
    chans = [c for c in TARGET_CHANNELS if f"loss_{c}" in last]
    if chans:
        v = {c: med(rows, f"loss_{c}", max(0, step - 1000)) for c in chans}
        print("\n[5] per-channel MSE (one lagging = normalization, not capacity)")
        print("    " + "  ".join(f"{c}={x:.3f}" for c, x in v.items()))
        worst, best = max(v.values()), min(v.values())
        print(verdict(worst / max(best, 1e-9) < 5,
                      f"spread {best:.3f}-{worst:.3f}, balanced",
                      f"spread {best:.3f}-{worst:.3f} — one channel is far behind"))

    # --- 6. the strict number --------------------------------------------------
    # Full sampling from pure noise, compared to ground truth. This measures the
    # task rather than the denoising objective, and it is the number that can
    # stay flat while the training loss looks fine.
    sm = [(r["step"], r["sample_mse_train"]) for r in rows if "sample_mse_train" in r]
    print("\n[6] sampled-from-noise MSE at each render (THE strict number)")
    if sm:
        print("    " + "  ".join(f"{s:,}:{v:.3f}" for s, v in sm[-8:]))
        print(verdict(len(sm) < 2 or sm[-1][1] < sm[0][1],
                      "falling",
                      "NOT falling — training loss can look fine while this stalls"))
    else:
        print("    (no render yet — first one lands at the render_every boundary)")

    # --- 7. gradient budget ----------------------------------------------------
    g = [r for r in rows if r.get("grad_norm_mse") is not None]
    if g:
        print("\n[7] gradient budget — which term is driving training")
        print(f"    {'step':>9} {'mse':>9} {'vb':>9} {'L_image':>9}")
        for r in g[-4:]:
            rt = r.get("grad_norm_render_total")
            rt_s = "  (off)" if rt in (None, 0.0) else f"{rt:9.3f}"
            print(f"    {r['step']:>9,} {r['grad_norm_mse']:9.3f} "
                  f"{r.get('grad_norm_vb', 0.0):9.4f} {rt_s:>9}")
        print("    (L_image is 0 until its ramp opens — that is the schedule, not a fault)")

    print(f"\n{'=' * 72}\n")


if __name__ == "__main__":
    main()
