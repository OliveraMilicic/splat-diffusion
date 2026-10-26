"""
eval/load_run.py — load a finished run's model without trusting its
config.json.

Every eval script so far does the same three lines:

    cfg = StageConfig.load(run_dir / "config.json")
    model = Denoiser(cfg)
    model.load_state_dict(torch.load(snap)["ema_state"])

which is correct only as long as `StageConfig`'s defaults never change.
They have. `runs/stage1_100obj` was trained before `learn_sigma` existed,
so its config.json has no such key, the dataclass fills in the *current*
default (`True`), and `Denoiser` builds a 6-channel output head for a
checkpoint holding 3-channel weights. `load_state_dict` then fails with a
shape mismatch — which is the good case. The bad case is a field whose
default changed but whose shape did not, where nothing fails and the run
is evaluated under a configuration it was never trained with.

So: **the checkpoint is the authority, not the config file.** This module
reconstructs the config from the payload the checkpoint itself carries,
drops keys the current dataclass no longer has, and then *infers*
`learn_sigma` from the width of the saved output projection — a property
of the weights, which cannot be stale.

    from eval.load_run import load_run
    model, cfg, meta = load_run("stage1_100obj", device)
"""

import dataclasses
import json
from pathlib import Path

from stage_config import DATA_DIR, SHARD_PATH, StageConfig

# `torch` and `Denoiser` are imported lazily inside `load_run()` rather than
# here. `image_metrics_of()` is pure JSON reading and is called by
# `train/plot_run.py` and `experiments/ablation_compare.py`, both of which are
# deliberately matplotlib+numpy only — a figure script should not need a deep
# learning framework loaded to draw a table. Everything that genuinely needs
# torch already has it.

# The tensor whose first dimension IS the output width. Named once here so a
# rename in model/ptv3.py surfaces as a clear failure in this file rather
# than as a silently skipped inference.
OUT_PROJ_KEY = "backbone.out_proj.weight"


def find_snapshot(run_dir: Path, prefer: str = None) -> Path:
    """The snapshot to evaluate. `snapshot_step<N>.pt` (written once at the
    end) beats `snapshot_latest.pt` (overwritten every ckpt_every), because
    on an interrupted run the two can differ and the step-named one says
    which step it is."""
    if prefer:
        p = Path(prefer)
        if not p.exists():
            raise SystemExit(f"--snapshot {p} does not exist")
        return p
    named = sorted(run_dir.glob("snapshot_step*.pt"))
    if named:
        return max(named, key=lambda p: int("".join(c for c in p.stem if c.isdigit())))
    latest = run_dir / "snapshot_latest.pt"
    if latest.exists():
        return latest
    snaps = sorted(run_dir.glob("snapshots/ema_step*.pt"))
    if snaps:
        return snaps[-1]
    # An overfit run (train/overfit.py) keeps no EMA by design and writes
    # live weights here instead.
    overfit = run_dir / "overfit_final.pt"
    if overfit.exists():
        return overfit
    raise SystemExit(f"no snapshot found under {run_dir}")


def config_from_payload(payload: dict, state: dict,
                        run_dir: Path = None) -> StageConfig:
    """Rebuild the StageConfig a checkpoint was trained under.

    Three corrections applied to the recorded dict, in order:

    1. **Drop unknown keys.** A config written by a newer version of the
       code than is running would otherwise raise `TypeError` on an
       unexpected keyword. Warn rather than crash — the eval is still
       meaningful without a field this code has no use for.
    2. **Infer `learn_sigma` from the weights.** `out_proj.weight` has
       `2 * n_target_ch` rows under learn_sigma and `n_target_ch` without.
       That is a fact about the file; the config field is a claim about it.
    3. **Re-resolve data paths** if the recorded ones do not exist. Runs
       launched on the slurm head node record `/lustre/...`; the GPU
       container sees only `/home/...` (stage_config's `_resolve`).
    """
    cfg_dict = dict(payload.get("cfg") or {})
    if not cfg_dict and run_dir is not None:
        cfg_path = run_dir / "config.json"
        if cfg_path.exists():
            cfg_dict = json.loads(cfg_path.read_text())
            print(f"  checkpoint carries no cfg; fell back to {cfg_path.name}")

    known = {f.name for f in dataclasses.fields(StageConfig)}
    unknown = sorted(set(cfg_dict) - known)
    if unknown:
        print(f"  ignoring {len(unknown)} unknown config key(s) from a different "
              f"code version: {unknown}")
    cfg = StageConfig(**{k: v for k, v in cfg_dict.items() if k in known})

    missing = sorted(known - set(cfg_dict))
    if missing:
        print(f"  config predates {len(missing)} field(s), using current defaults: "
              f"{missing}")

    if OUT_PROJ_KEY not in state:
        raise SystemExit(
            f"checkpoint has no {OUT_PROJ_KEY!r} — cannot verify the output width "
            f"against the config. Keys look like: {sorted(state)[:4]}")
    out_ch = int(state[OUT_PROJ_KEY].shape[0])
    n_target = cfg.n_target_ch
    if out_ch == 2 * n_target:
        inferred = True
    elif out_ch == n_target:
        inferred = False
    else:
        raise SystemExit(
            f"checkpoint's output width is {out_ch}, which is neither "
            f"{n_target} nor {2 * n_target} — the recorded stage ({cfg.stage}, "
            f"{n_target} target channels) cannot be the stage it was trained at. "
            f"Refusing to guess.")
    if inferred != cfg.learn_sigma:
        print(f"  learn_sigma: config says {cfg.learn_sigma}, the weights say "
              f"{inferred} ({out_ch} output channels for {n_target} target "
              f"channels) — trusting the WEIGHTS")
        cfg.learn_sigma = inferred

    if not Path(cfg.shard_path).is_dir():
        print(f"  shard_path {cfg.shard_path} not visible here -> {SHARD_PATH}")
        cfg.shard_path = SHARD_PATH
    if not Path(cfg.data_dir).is_dir():
        print(f"  data_dir {cfg.data_dir} not visible here -> {DATA_DIR}")
        cfg.data_dir = DATA_DIR
    return cfg


def load_run(run_name: str, device, snapshot: str = None,
             run_root: str = "runs") -> tuple:
    """-> (model in eval mode with EMA weights, cfg, meta dict).

    EMA weights, always: STAGE_SPEC §5 is train-live, sample-EMA, and a
    figure made from the live weights is not the model the run produced.
    """
    import torch                                    # see module header
    from model.denoiser import Denoiser

    run_dir = Path(run_root) / run_name
    if not run_dir.is_dir():
        raise SystemExit(f"no run at {run_dir}")
    snap = find_snapshot(run_dir, snapshot)
    print(f"loading {snap}")
    payload = torch.load(snap, map_location=device, weights_only=False)

    key = next((k for k in ("ema_state", "ema", "model_state") if k in payload), None)
    if key is None:
        raise SystemExit(f"{snap} has no weights (keys: {sorted(payload)})")
    if key != "ema_state":
        print(f"  WARNING: no ema_state in this checkpoint, falling back to "
              f"{key!r} — these are the LIVE training weights, not the EMA ones")
    state = payload[key]

    cfg = config_from_payload(payload, state, run_dir)
    cfg.run_name = run_name
    model = Denoiser(cfg).to(device)
    model.load_state_dict(state)          # strict: any remaining mismatch is a bug
    model.eval()

    step = int(payload.get("step", 0))
    print(f"  stage {cfg.stage} ({', '.join(cfg.target_names)}), "
          f"learn_sigma={cfg.learn_sigma}, step {step:,}")
    return model, cfg, dict(snapshot=str(snap), step=step, run_dir=run_dir,
                            weights_key=key)


def train_names_of(run_dir: Path) -> list:
    """The objects a run actually trained on, or None for the whole split."""
    p = Path(run_dir) / "train_subset.json"
    return json.loads(p.read_text())["names"] if p.exists() else None


def image_metrics_of(run_dir: Path) -> dict:
    """A run's PSNR/SSIM/LPIPS/FID/KID, whichever file they landed in.

    Two writers disagree on the name, which silently cost the loss ablation
    half its results table:

        train/loop.py         -> runs/<run>/image_metrics.json
        eval/image_eval.py    -> runs/<run>/image_metrics_<sampler>.json

    and every reader looked only for the first. A run whose metrics came
    from the standalone script was therefore invisible to `plot_run.py` and
    `ablation_compare.py` — no error, just an empty panel and em-dashes in
    the table, which reads as "not run yet" rather than "looked in the wrong
    place".

    Resolution order is deliberate: the run's own `image_metrics.json`
    first, then the DPM-sampler file (the default both writers use, so it is
    the like-for-like one), then any remaining variant. The chosen path is
    recorded under `_meta.source_file` so a figure can say which sampler it
    is showing rather than implying they are interchangeable — they are not
    (see eval/image_eval.py's --sampler help).
    """
    run_dir = Path(run_dir)
    candidates = [run_dir / "image_metrics.json",
                  run_dir / "image_metrics_dpm.json"]
    candidates += sorted(p for p in run_dir.glob("image_metrics_*.json")
                         if p not in candidates)
    for p in candidates:
        if not p.exists():
            continue
        try:
            m = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        m.setdefault("_meta", {})["source_file"] = p.name
        return m
    return None
