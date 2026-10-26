"""
train/logging_utils.py — STAGE_SPEC §11 build-order step 5.5.

LOGGING.md supersedes STAGE_SPEC §7 on timing: built here, BEFORE the
overfit gate (step 6), not after — "the overfit test is the hardest
debugging moment in the project; running it blind wastes the gate."

`stats.jsonl` (TRAINING_SPEC §7) is written unconditionally, in
parallel with W&B, not as a fallback bolted on later — LOGGING.md: "the
one thing that can't get locked out."
"""

import dataclasses
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np


def check_wandb_reachable(timeout: float = 5.0) -> bool:
    """`curl -sI https://api.wandb.ai` equivalent (LOGGING.md / STAGE_SPEC
    §7). An HTTP error response (e.g. 404 for a bare HEAD on `/`) still
    proves the host is reachable — only a connection-level failure
    (DNS/timeout/refused, the Cloudflare-block signature LOGGING.md
    warns about) means "unreachable"."""
    try:
        urllib.request.urlopen("https://api.wandb.ai", timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


class TimestepBinTracker:
    """Running per-bin mean loss over the whole run so far (LOGGING.md
    panel 1: "loss bucketed by timestep — the single most important
    panel"). Cumulative rather than windowed — simpler, and the overfit
    run is short enough (~2000 steps / 10 bins = ~200 samples/bin by the
    end) that a cumulative mean is still informative throughout."""

    def __init__(self, n_bins: int = 10, T: int = 1000):
        self.n_bins = n_bins
        self.T = T
        self.sums = np.zeros(n_bins)
        self.counts = np.zeros(n_bins)

    def update(self, t: int, loss: float):
        b = min(int(t / self.T * self.n_bins), self.n_bins - 1)
        self.sums[b] += loss
        self.counts[b] += 1

    def means(self) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            return self.sums / self.counts  # nan where count==0, by design


def per_term_grad_norms(terms: dict, params, keys=("mse", "vb", "render_total")) -> dict:
    """Gradient norm contributed by each loss term SEPARATELY.

    The loss values alone cannot answer "which term is driving training".
    A term can be numerically tiny and still dominate the gradient (a small
    quantity with a steep slope), or large and nearly flat. Only
    `||d term / d params||` settles it, and that is what the optimizer
    actually sees — after `clip_grad_norm_` rescales the SUM, a term with
    10x the gradient of another is getting 10x the say regardless of how the
    printed losses compare.

    Costs one extra backward per term, so call it every few hundred steps,
    not every step. `retain_graph=True` keeps the graph alive for the real
    backward that follows; `allow_unused` covers the variance-head
    parameters, which the MSE term genuinely does not touch.

    Returns `{f"grad_norm_{key}": float}` for the terms present.
    """
    import torch

    params = [p for p in params if p.requires_grad]
    out = {}
    for key in keys:
        term = terms.get(key)
        if term is None or not torch.is_tensor(term) or not term.requires_grad:
            continue
        grads = torch.autograd.grad(term, params, retain_graph=True,
                                     allow_unused=True)
        total = torch.zeros((), device=grads[0].device if grads[0] is not None
                             else term.device)
        for g in grads:
            if g is not None:
                total = total + g.pow(2).sum()
        out[f"grad_norm_{key}"] = float(total.sqrt())
    return out


class StageLogger:
    """W&B (LOGGING.md) + `stats.jsonl` (TRAINING_SPEC §7), always both —
    W&B is best-effort (falls back to offline mode, or to nothing if the
    package/init itself fails), `stats.jsonl` never is."""

    def __init__(self, cfg, run_name: str, use_wandb: bool = True, run_root: str = "runs"):
        self.run_dir = Path(run_root) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._stats_file = open(self.run_dir / "stats.jsonl", "a")

        self.wandb_run = None
        if use_wandb:
            self._init_wandb(cfg, run_name)

    def _init_wandb(self, cfg, run_name: str):
        try:
            import wandb
        except ImportError:
            print("W&B: package not installed, logging to stats.jsonl only")
            return

        reachable = check_wandb_reachable()
        authenticated = bool(wandb.api.api_key)
        # "online" needs BOTH: reachability alone isn't enough -- without
        # an API key, wandb.init(mode="online") can hang waiting on
        # interactive `wandb login` input, which would stall an
        # unattended/background training run.
        online = reachable and authenticated
        mode = "online" if online else "offline"
        try:
            self.wandb_run = wandb.init(
                project="splat-diffusion", name=run_name,
                config=dataclasses.asdict(cfg), mode=mode,
            )
            if online:
                print(f"W&B: {self.wandb_run.url}")
            elif not authenticated:
                print("W&B: not logged in (`wandb login`), running offline "
                      "(`wandb sync` later once authenticated)")
            else:
                print("W&B: api.wandb.ai unreachable, running offline "
                      "(`wandb sync` later) — LOGGING.md fallback")
        except Exception as e:
            print(f"W&B init failed ({e}); continuing with stats.jsonl only")
            self.wandb_run = None

    def log(self, step: int, **metrics):
        record = dict(step=step, time=time.time(), **metrics)
        self._stats_file.write(json.dumps(record) + "\n")
        self._stats_file.flush()
        if self.wandb_run is not None:
            import wandb
            wandb.log(metrics, step=step)

    def log_image(self, step: int, key: str, image_hwc_float01: np.ndarray):
        if self.wandb_run is not None:
            import wandb
            wandb.log({key: wandb.Image(image_hwc_float01)}, step=step)

    def log_summary(self, **kv):
        """Summary metrics (LOGGING.md "Config and summary"): logged
        once, not as a time series."""
        self._stats_file.write(json.dumps(dict(summary=True, **kv)) + "\n")
        self._stats_file.flush()
        if self.wandb_run is not None:
            for k, v in kv.items():
                self.wandb_run.summary[k] = v

    def close(self):
        self._stats_file.close()
        if self.wandb_run is not None:
            import wandb
            wandb.finish()
