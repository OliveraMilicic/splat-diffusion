"""
train/checkpoint.py — save/load, TRAINING_SPEC §6.

Two kinds of saved state, deliberately separate:

  **snapshot** (`snapshot_latest.pt`, plus rotating `snapshot_step{N}.pt`)
      EMA weights + config only. What the render callback and final
      sampling load. Small, safe to keep many of.

  **training state** (`state_latest.pt`, overwritten, never accumulated)
      live weights, EMA weights, optimizer, step, and RNG state. What
      `--resume` loads.

**Resume restores RNG state**, not just weights (TRAINING_SPEC §6):
otherwise the timestep/noise sequence after a resume diverges from what
it would have been, which matters here because resumes are expected to
be frequent (dropped SSH, shared-GPU preemption).
"""

import dataclasses
import random
from pathlib import Path

import numpy as np
import torch


def save_snapshot(path, ema, cfg, step: int, extra: dict = None):
    """EMA weights only — for sampling/eval/renders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(ema_state=ema.state_dict(), cfg=dataclasses.asdict(cfg), step=step)
    if extra:
        payload.update(extra)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)       # write-then-rename: a crash mid-save
    tmp.replace(path)              # can't leave a truncated checkpoint behind
    return path


def save_training_state(path, model, ema, optimizer, cfg, step: int, extra: dict = None):
    """Everything needed to resume bit-for-bit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        model_state=model.state_dict(),
        ema_state=ema.state_dict(),
        optimizer_state=optimizer.state_dict(),
        cfg=dataclasses.asdict(cfg),
        step=step,
        rng=dict(
            torch=torch.get_rng_state(),
            torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            numpy=np.random.get_state(),
            python=random.getstate(),
        ),
    )
    if extra:
        payload.update(extra)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_training_state(path, model, ema, optimizer, device) -> int:
    """Restores weights, optimizer and RNG. Returns the step to resume at."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    ema.load_state_dict(ckpt["ema_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])

    rng = ckpt.get("rng")
    if rng:
        torch.set_rng_state(rng["torch"].cpu() if torch.is_tensor(rng["torch"]) else rng["torch"])
        if rng.get("torch_cuda") is not None and torch.cuda.is_available():
            try:
                # `map_location=device` above put these ON the GPU, and
                # set_rng_state_all only accepts CPU ByteTensors -- a
                # cuda.ByteTensor fails its type check with TypeError. Same
                # .cpu() the line above does for the CPU generator.
                torch.cuda.set_rng_state_all(
                    [s.cpu() if torch.is_tensor(s) else s for s in rng["torch_cuda"]])
            except (RuntimeError, ValueError, TypeError) as e:
                # e.g. resumed on a different GPU count -- worth saying out
                # loud rather than silently continuing with a different
                # noise stream than the run had before. Catching broadly is
                # deliberate: a diverged noise stream costs continuity, an
                # uncaught raise here costs the whole resume.
                print(f"WARNING: could not restore CUDA RNG state ({e}); "
                      f"noise sequence will diverge from the original run")
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
    else:
        print("WARNING: checkpoint has no RNG state; noise sequence will not "
              "continue the original run")

    step = int(ckpt.get("step", 0))
    print(f"resumed from {path} at step {step}")
    return step
