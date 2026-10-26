"""
eval/render_grid.py — the one renderer both `train/overfit.py` and
`train/loop.py` use, so the gate figure and the training figures are
visually comparable. A render that differs between the two makes "does
the full run look as good as the gate?" unanswerable by eye, which is the
question these figures exist to answer.

Columns, left to right:

    INPUT   exactly what the model is conditioned on, and nothing else.
            Every TARGET channel is replaced by its dataset mean, so at
            stage 1 the colour is withheld, and at stage 3 the size,
            opacity AND colour are all withheld — a panel that only greys
            out the colour would show the viewer generated geometry and
            call it input. This panel doubles as the **mean-fill
            baseline** (STAGE_SPEC §9): it is literally "predict the
            dataset mean appearance", so the samples to its right have to
            beat something visible.
    GT      ground-truth appearance, the training target.
    seed 0  sampled from pure noise.
    seed 1  sampled again, different seed. The pair shows cross-seed
            diversity, the mode-collapse signal at dataset scale
            (STAGE_SPEC §9 check 9).

**Renderer**: real 3DGS rasterization via `eval/render3d.py` when a GPU is
present, falling back to `splat_utils.render_points` otherwise. This
matters more from stage 3 on than it did at stage 1: the point renderer
draws neither covariance nor real opacity compositing, so a model that got
every Gaussian's SHAPE wrong would look correct under it. The panel titles
name the backend so a figure can never be misread as showing more than it
does.
"""

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from diffusion.sampling import sample as diffusion_sample
from eval.render3d import UP_AXIS, gsplat_available, orbit_cameras, render_views
from preprocessing import tokens_to_splat_dict, unstandardize
from stage_config import recombine_channels

# One shared rig for every figure, so panels across a run are comparable.
FIGURE_VIEWS = 1          # azimuths per elevation in a grid panel
FIGURE_ELEV = (20.0,)
FIGURE_SIZE = 260


def to_tokens(cond_np, target_arr, cfg, norm_stats) -> np.ndarray:
    """(standardized cond, standardized target) -> RAW (N, 14) tokens, still
    in the per-object NORMALIZED frame. The recombine -> unstandardize chain
    `eval/decode.py` checks, stopping one step short of world units because
    `eval/render3d.py` renders in the normalized frame on purpose (see its
    docstring: one fixed camera rig has to frame all 1,273 objects the same
    way, or PSNR averaged over objects is not a number)."""
    target_np = target_arr.detach().cpu().numpy() if torch.is_tensor(target_arr) else target_arr
    return unstandardize(recombine_channels(cond_np, target_np, cfg), norm_stats)


def to_splat(cond_np, target_arr, cfg, norm_stats, centre, radius, name):
    """Same chain, taken all the way back to WORLD units — what
    `splat_utils.render_points` expects.

    `centre`/`radius` must be THIS object's pair: per-object normalization
    (DESIGN_SPEC §3.4) is only invertible with them, and the wrong pair
    silently yields a mis-scaled splat rather than an error.
    """
    return tokens_to_splat_dict(to_tokens(cond_np, target_arr, cfg, norm_stats),
                                centre, radius, name=name)


def mean_fill_target(cond_np, cfg) -> np.ndarray:
    """The mean-fill baseline: every target channel set to the dataset mean.

    Zero in STANDARDIZED units is exactly the per-channel training mean, so
    this needs no separate statistics — it is the same `norm_stats` the rest
    of the pipeline already asserts against (STAGE_SPEC §2's table).
    """
    return np.zeros((cond_np.shape[0], cfg.n_target_ch), dtype=np.float32)


def geometry_only_tokens(cond_np, cfg, norm_stats, grey: float = 0.62) -> np.ndarray:
    """The INPUT panel's tokens: conditioning as given, target channels at
    the dataset mean, and colour additionally forced to flat grey.

    The grey override is deliberate on top of mean-fill: the dataset mean
    colour is a real prediction (a muddy grey-brown, and a strong one —
    STAGE_SPEC §9 check 9 warns it scores well on PSNR), so rendering it
    unlabelled next to the samples invites reading it as a result. Flat grey
    cannot be mistaken for one.
    """
    tok = to_tokens(cond_np, mean_fill_target(cond_np, cfg), cfg, norm_stats)
    tok = tok.copy()
    tok[:, 11:14] = grey
    return tok


def geometry_only(gt_splat: dict, grey: float = 0.62) -> dict:
    """Legacy world-space form, kept for callers holding a splat dict.
    Greys the COLOUR only, so it is honest at stage 1 and NOT at stage 3+
    (where scale and opacity are generated too) — prefer
    `geometry_only_tokens`."""
    geom = dict(gt_splat)
    geom["rgb"] = np.full_like(gt_splat["rgb"], grey)
    return geom


# ----------------------------------------------------------------- rendering
def render_tokens(tok_raw, size: int = FIGURE_SIZE,
                  backend: str = "auto", device=None) -> np.ndarray:
    """One image of a normalized-frame token array, through the best
    available renderer. The up axis is `eval.render3d.UP_AXIS`, not a
    parameter — the rig is fixed so every figure frames every object
    identically (see that module's docstring)."""
    viewmats, Ks = orbit_cameras(n_views=FIGURE_VIEWS, elevations=FIGURE_ELEV,
                                 image_size=size)
    imgs, _used = render_views(tok_raw, viewmats, Ks, image_size=size,
                               device=device, backend=backend)
    return imgs[0]


def active_backend() -> str:
    return "gsplat" if gsplat_available() else "points"


def render_rows(rows: list, title: str, out_path: Path, up_axis: int,
                size: int = FIGURE_SIZE, dpi: int = 110,
                backend: str = "auto", device=None) -> Path:
    """`rows` = list of dicts with keys:
        name, tag ("TRAIN"/"TEST"/None), n, tok_input, tok_gt, tok_s0,
        tok_s1, mse, div
    `tok_s1`/`div` may be None (single-sample render) — that column is then
    dropped for every row.
    """
    # The rig hardcodes the up axis (eval/render3d.UP_AXIS). Catching a
    # disagreement here beats rendering 1,273 chairs on their backs, which is
    # exactly the failure PLAN.md records from day 1.
    assert up_axis == UP_AXIS, (
        f"norm_stats up_axis={up_axis} but eval/render3d.py's rig uses {UP_AXIS}")

    two_samples = all(r.get("tok_s1") is not None for r in rows)
    n_cols = 4 if two_samples else 3
    fig, axes = plt.subplots(len(rows), n_cols,
                             figsize=(3.6 * n_cols, 4.0 * len(rows)))
    axes = np.atleast_2d(axes)

    for r, row in enumerate(rows):
        tag = f"[{row['tag']}] " if row.get("tag") else ""
        panels = [
            (row["tok_input"],
             f"INPUT: conditioning only\n(target ch = dataset mean)\n"
             f"{tag}{row['name'][:22]}  N={row['n']}"),
            (row["tok_gt"], "GROUND TRUTH\n(the training target)"),
            (row["tok_s0"], f"SAMPLED, seed 0\nMSE={row['mse']:.4f}"),
        ]
        if two_samples:
            panels.append((row["tok_s1"],
                           f"SAMPLED, seed 1\n|s0-s1|={row['div']:.4f}"))
        for ax, (tok, panel_title) in zip(axes[r], panels):
            ax.imshow(render_tokens(tok, size=size, backend=backend, device=device))
            ax.set_title(panel_title, fontsize=9)
            ax.axis("off")

    used = active_backend() if backend == "auto" else backend
    renderer_note = ("3DGS rasterization (gsplat)" if used == "gsplat" else
                     "POINT-CLOUD fallback — no covariance, no real opacity "
                     "compositing (no GPU here)")
    plt.suptitle(f"{title}\nrenderer: {renderer_note}", fontsize=11,
                 y=1.02 if len(rows) > 1 else 1.10)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


@torch.no_grad()
def sample_and_render(ema_model, datasets: list, cfg, norm_stats, device,
                      out_path: Path, title_prefix: str = "") -> tuple:
    """Sample with the EMA weights (STAGE_SPEC §5: train live, sample EMA)
    and write one figure.

    `datasets` = [(tag, dataset, [indices]), ...] — one group of rows per
    tag, e.g. [("TRAIN", ds, [0, 1])] mid-training, or
    [("TRAIN", ...), ("TEST", ...)] for the final evaluation. Fixed
    indices, so the sequence of grids across a run is comparable
    (LOGGING.md panel 5).

    Returns (path, {tag: mean sample MSE}).
    """
    # Training holds ~34GB of a 42.6GB card at batch 4 x N=32768; the
    # sampler allocates on top, so hand back the cached blocks first.
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rows, mse_by_tag = [], {}
    for tag, dataset, indices in datasets:
        mses = []
        for idx in indices:
            item = dataset[idx]
            cond = item["cond"].to(device)
            target = item["target"].to(device)
            offset = torch.tensor([item["n"]], dtype=torch.long, device=device)
            s0 = diffusion_sample(ema_model, cond, offset, T=cfg.T,
                                  num_steps=cfg.sample_steps, seed=0, device=device)
            s1 = diffusion_sample(ema_model, cond, offset, T=cfg.T,
                                  num_steps=cfg.sample_steps, seed=1, device=device)
            mses.append(torch.mean((s0 - target) ** 2).item())

            cond_np = cond.cpu().numpy()
            rows.append(dict(
                name=item["name"], tag=tag, n=item["n"], mse=mses[-1],
                div=(s0 - s1).abs().mean().item(),
                tok_input=geometry_only_tokens(cond_np, cfg, norm_stats),
                tok_gt=to_tokens(cond_np, target, cfg, norm_stats),
                tok_s0=to_tokens(cond_np, s0, cfg, norm_stats),
                tok_s1=to_tokens(cond_np, s1, cfg, norm_stats),
            ))
        if mses:
            mse_by_tag[tag] = float(np.mean(mses))

    summary = "  ".join(f"{t}={m:.4f}" for t, m in mse_by_tag.items())
    path = render_rows(rows, title=f"{title_prefix}(sampled with EMA weights)  "
                                   f"stage {cfg.stage}: {', '.join(cfg.target_names)}\n"
                                   f"mean sample MSE: {summary}",
                       out_path=out_path, up_axis=norm_stats["up_axis"], device=device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return path, mse_by_tag
