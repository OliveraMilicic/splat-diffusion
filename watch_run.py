"""
watch_run.py — live view of a running train/overfit.py (or any run that
writes `runs/<name>/stats.jsonl` via train/logging_utils.StageLogger).

Reads the jsonl the training process is appending to; never touches the
training process itself, so it's safe to start/stop/rerun at will, and
safe to run several at once.

Two modes:

    python watch_run.py overfit1_stage1_16k             # live text, refreshes
    python watch_run.py overfit1_stage1_16k --plot      # write a PNG and exit

`--plot` is the one to use from a Jupyter cell (there's no display on
these nodes, so it saves to reports/ rather than showing a window):

    !python watch_run.py overfit1_stage1_16k --plot
    from IPython.display import Image; Image("reports/live_overfit1_stage1_16k.png")
"""

import argparse
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
REPORT_DIR = REPO_ROOT / "reports"


def read_stats(run_name: str) -> list:
    path = REPO_ROOT / "runs" / run_name / "stats.jsonl"
    if not path.exists():
        return []
    rows = []
    # tolerate a torn final line: the trainer may be mid-write when we read
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not row.get("summary"):
            rows.append(row)
    return rows


def fmt_text(rows: list, total_steps: int = None) -> str:
    if not rows:
        return "no steps logged yet"
    last = rows[-1]
    step = last["step"] + 1
    recent = [r["loss"] for r in rows[-50:]]
    med = sorted(recent)[len(recent) // 2]
    first_losses = [r["loss"] for r in rows[:50]]
    med0 = sorted(first_losses)[len(first_losses) // 2]

    elapsed = rows[-1]["time"] - rows[0]["time"]
    per_step = elapsed / max(1, len(rows) - 1)
    eta = ""
    if total_steps:
        remaining = (total_steps - step) * per_step
        eta = f"  ETA {remaining / 60:5.1f} min"

    lines = [
        f"step {step}" + (f"/{total_steps}" if total_steps else "")
        + f"   elapsed {elapsed / 60:5.1f} min   {per_step * 1000:.0f} ms/step{eta}",
        f"  loss (median of last 50) : {med:.5f}     (first 50: {med0:.5f})",
        f"  loss (this step)         : {last['loss']:.5f}",
        f"  per-channel r/g/b        : {last.get('loss_r', 0):.5f} / "
        f"{last.get('loss_g', 0):.5f} / {last.get('loss_b', 0):.5f}",
        f"  adaLN gate_mean          : {last.get('gate_mean', 0):.4e}   "
        f"(must grow from 0 — LOGGING.md panel 2)",
        f"  grad_norm (pre-clip)     : {last.get('grad_norm', 0):.4f}",
    ]
    if "gpu_mem_gb" in last:
        lines.append(f"  gpu_mem                  : {last['gpu_mem_gb']:.1f} GB")

    bins = sorted(k for k in last if k.startswith("loss_t_bin_"))
    if bins:
        lines.append("  loss by timestep bucket (t=0 .. t=T, cumulative mean):")
        vals = "  ".join(f"{last[b]:.4f}" for b in bins)
        lines.append(f"    {vals}")
    return "\n".join(lines)


def make_plot(rows: list, run_name: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")  # CLAUDE.md: no display on these nodes
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in rows]
    losses = [r["loss"] for r in rows]
    gates = [r.get("gate_mean", 0.0) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    win = max(1, len(losses) // 50)
    smoothed = [sorted(losses[max(0, i - win):i + 1])[(min(i, win)) // 2]
                for i in range(len(losses))]
    axes[0].plot(steps, losses, lw=0.5, alpha=0.3, label="per-step (random t)")
    axes[0].plot(steps, smoothed, lw=1.8, color="C3", label=f"running median ({win})")
    axes[0].set(xlabel="step", ylabel="L_simple", yscale="log", title="loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, gates, lw=1.4, color="C2")
    axes[1].set(xlabel="step", ylabel="mean |gate|",
                title="adaLN gate magnitude\n(must grow from 0)")
    axes[1].grid(alpha=0.3)

    bins = sorted((k for k in rows[-1] if k.startswith("loss_t_bin_")),
                   key=lambda k: int(k.rsplit("_", 1)[1]))
    if bins:
        n_bins = len(bins)
        centres = [(int(b.rsplit("_", 1)[1]) + 0.5) * 1000 / n_bins for b in bins]
        axes[2].plot(centres, [rows[-1][b] for b in bins], "o-")
        axes[2].set(xlabel="diffusion timestep t", ylabel="mean loss in bucket",
                    yscale="log", title="loss by timestep bucket\n(LOGGING.md panel 1)")
        axes[2].grid(alpha=0.3)

    plt.suptitle(f"{run_name} — step {steps[-1] + 1}", fontsize=11)
    plt.tight_layout()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"live_{run_name}.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out


def live(run_name: str, refresh: float = 15.0, total_steps: int = 2000,
         max_minutes: float = 60.0):
    """Auto-refreshing view for a Jupyter cell. Redraws in place until the
    run stops advancing (finished/died) or `max_minutes` elapses; interrupt
    the cell at any time — nothing here touches the training process.

        from watch_run import live
        live("overfit_32k_bf16", total_steps=1000)
    """
    from IPython.display import clear_output, display, Image

    deadline = time.time() + max_minutes * 60
    last_step, stalled = -1, 0
    while time.time() < deadline:
        rows = read_stats(run_name)
        clear_output(wait=True)
        if rows:
            print(fmt_text(rows, total_steps))
            display(Image(str(make_plot(rows, run_name))))
            step = rows[-1]["step"]
            stalled = stalled + 1 if step == last_step else 0
            last_step = step
            if step + 1 >= total_steps:
                print("\nrun reached the requested step count.")
                return
            if stalled >= 3:
                print("\nno new steps in 3 refreshes — run finished or stopped.")
                return
        else:
            print(f"waiting for {run_name} to log its first step...")
        time.sleep(refresh)
    print("\nstopped watching (max_minutes reached; training continues).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    ap.add_argument("--plot", action="store_true", help="write a PNG and exit")
    ap.add_argument("--total-steps", type=int, default=2000, help="for the ETA")
    ap.add_argument("--every", type=float, default=10.0, help="text refresh seconds")
    args = ap.parse_args()

    if args.plot:
        rows = read_stats(args.run_name)
        if not rows:
            print("no steps logged yet")
            return
        print(f"wrote {make_plot(rows, args.run_name)}")
        return

    try:
        while True:
            rows = read_stats(args.run_name)
            print("\033[2J\033[H", end="")  # clear screen, cursor home
            print(f"watching {args.run_name}   (ctrl-C to stop; this does not "
                  f"affect the training process)\n")
            print(fmt_text(rows, args.total_steps))
            time.sleep(args.every)
    except KeyboardInterrupt:
        print("\nstopped watching (training continues)")


if __name__ == "__main__":
    main()
