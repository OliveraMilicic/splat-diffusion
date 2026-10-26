"""
eval/render3d.py — real 3D Gaussian Splatting rasterization, via `gsplat`.

`splat_utils.render_points` is a matplotlib scatter plot: it has no
covariance, no orientation, and no alpha compositing beyond a painter's
depth sort. Its own docstring says so. That is fine for "is this a
chair?", and it is *actively wrong* for this build target: stage 3
diffuses **log-scale and opacity**, and a point renderer draws neither
(point size comes from mean scale, alpha is clamped to >= 0.05). A model
that got every Gaussian's shape wrong would look identical under it.

Every image metric (`eval/metrics.py`: PSNR / SSIM / LPIPS / FID / KID)
and every presentation figure therefore renders through here.

**Everything is rendered in the per-object NORMALIZED frame**, not world
units. `preprocessing.normalize_object` centres each object and divides
by its own radius, so in that frame every chair is inscribed in the unit
sphere and ONE fixed camera rig frames all 1,273 of them identically.
That identical framing is a requirement, not a convenience: PSNR
averaged over objects framed differently is not a meaningful number, and
FID between two sets of images that differ in camera placement measures
the cameras.

**Backend.** `gsplat` is CUDA-only. On a CPU-only box (login01 has no
`/dev/nvidia*` at all) `render_views` falls back to the point renderer so
the pipeline stays runnable end to end — but it returns the backend name
alongside the images and `eval/metrics.py` records it in its output, so a
CPU smoke number can never be quietly compared against a GPU one.
"""

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch

# ----------------------------------------------------------------- camera rig
# Elevation 20 deg looking slightly down, which is how a chair reads: it shows
# the seat plane and the leg structure at once. Azimuths evenly spaced so the
# views are decorrelated; the count is the caller's choice, and the SAME rig is
# used for prediction and ground truth (the metrics compare them pairwise).
DEFAULT_ELEVATIONS = (20.0, -5.0)
UP_AXIS = 2                # STAGE_SPEC §2, established empirically, do not re-derive
FOV_DEG = 40.0
# `preprocessing.normalize_object` divides by `radius = max |xyz - centre|`
# taken over ALL COORDINATES, so objects are inscribed in the unit CUBE, not
# the unit sphere — a corner point sits at up to sqrt(3) = 1.732 from the
# origin, not 1.0. Framing for radius 1.0 clipped real chairs (measured: 1.5%
# of points off-frame, 6px past the border, on the very first val object);
# check_render3d.py check C2 caught it. Measured max ||xyz|| over 215 objects
# = 1.695, right up against the cube bound, so the bound is the constant to
# use rather than a sample maximum that a 216th object could exceed.
FIT_RADIUS = float(np.sqrt(3.0) * 1.04)      # cube bound + 4% margin
CAM_DISTANCE = FIT_RADIUS / np.tan(np.radians(FOV_DEG) / 2)
WHITE_BG = (1.0, 1.0, 1.0)


def orbit_cameras(n_views: int = 8, elevations=DEFAULT_ELEVATIONS,
                  distance: float = CAM_DISTANCE, image_size: int = 256,
                  fov_deg: float = FOV_DEG):
    """Fixed orbit rig -> (viewmats (C,4,4) world->cam, Ks (C,3,3)).

    Deterministic: no RNG anywhere. Two views of the same object rendered
    in different processes must be pixel-identical, or PSNR/SSIM between a
    prediction and its ground truth measures camera jitter.
    """
    up = np.zeros(3, np.float64)
    up[UP_AXIS] = 1.0
    horiz = [i for i in range(3) if i != UP_AXIS]

    viewmats = []
    for elev in elevations:
        e = np.radians(elev)
        for k in range(n_views):
            a = 2 * np.pi * k / n_views
            eye = np.zeros(3, np.float64)
            eye[horiz[0]] = np.cos(a) * np.cos(e)
            eye[horiz[1]] = np.sin(a) * np.cos(e)
            eye[UP_AXIS] = np.sin(e)
            eye *= distance

            forward = -eye / np.linalg.norm(eye)           # camera -> origin
            right = np.cross(forward, up)
            nr = np.linalg.norm(right)
            if nr < 1e-6:                                   # looking straight down `up`
                right = np.cross(forward, np.array([1.0, 0.0, 0.0]))
                nr = np.linalg.norm(right)
            right /= nr
            true_up = np.cross(right, forward)

            # World->camera, OpenCV convention (x right, y DOWN, z forward),
            # which is what gsplat's projection expects. Getting the y sign
            # wrong renders a vertically mirrored chair and throws no error.
            R = np.stack([right, -true_up, forward], axis=0)
            view = np.eye(4)
            view[:3, :3] = R
            view[:3, 3] = -R @ eye
            viewmats.append(view)

    f = image_size / (2 * np.tan(np.radians(fov_deg) / 2))
    K = np.array([[f, 0.0, image_size / 2],
                  [0.0, f, image_size / 2],
                  [0.0, 0.0, 1.0]])
    Ks = np.repeat(K[None], len(viewmats), axis=0)
    return np.stack(viewmats).astype(np.float32), Ks.astype(np.float32)


# ----------------------------------------------------------------- token -> gaussians
def tokens_to_gaussians(tok_raw, device=None):
    """RAW (unstandardized, per-object normalized) (N, 14) tokens -> the five
    arrays `gsplat.rasterization` wants, in the normalized frame.

    Applies the activations `splat_utils.read_splat`'s docstring calls the
    three silent corrupters, in the same direction
    `preprocessing.tokens_to_splat_dict` does: scale is a LOG (exp it),
    opacity is a LOGIT (sigmoid it), the quaternion is w-first (which is
    also `gsplat`'s "wxyz convension", and it does not require unit norm).

    Clamping (STAGE_SPEC §5 "Output clamping"): opacity to [0, 1) comes free
    from the sigmoid; RGB is clamped to [0, 1]; scale is clamped BELOW at a
    positive floor and above, because at stage 3 log-scale is generated and
    an unlucky sample can emit a value whose exp is either 0 (a degenerate
    covariance the rasterizer divides by) or huge (one Gaussian covering the
    whole frame).
    """
    # Graph-preserving when handed a tensor: the render LOSS calls this on a
    # tensor that requires grad, and `np.asarray` on such a tensor raises
    # rather than silently detaching — but going through numpy unconditionally
    # would detach for any caller that got past it, which is the failure mode
    # where a render loss trains nothing and reports no error.
    if torch.is_tensor(tok_raw):
        tok = tok_raw.to(device=device or tok_raw.device, dtype=torch.float32)
    else:
        tok = torch.as_tensor(np.asarray(tok_raw), dtype=torch.float32, device=device)
    means = tok[:, 0:3]
    # exp(log_scale) in the normalized frame. The object is inscribed in the
    # unit sphere, so a Gaussian wider than the object itself is nonsense;
    # 1.0 is a generous ceiling and 1e-6 keeps the covariance non-singular.
    scales = torch.exp(tok[:, 3:6]).clamp(1e-6, 1.0)
    quats = tok[:, 6:10]
    opacities = torch.sigmoid(tok[:, 10])
    colors = tok[:, 11:14].clamp(0.0, 1.0)
    return means, quats, scales, opacities, colors


# ----------------------------------------------------------------- rendering
def _gsplat_rgb(means, quats, scales, opacities, colors, viewmats, Ks,
                image_size: int, background):
    """`gsplat.rasterization` -> (C, H, W, 3) composited over `background`.

    **The background is composited here, not by gsplat**, and that is not a
    stylistic choice. In gsplat 1.5.3's `packed=True` path,
    `rasterize_to_pixels` derives `image_dims` from `means2d`, which is
    `[nnz, 2]` when packed — so `image_dims` is empty and it asserts
    `backgrounds.shape == (channels,)`, while `rasterization`'s own docstring
    documents `backgrounds` as `[..., C, D]`. Passing the documented `(C, 3)`
    raises `AssertionError: torch.Size([1, 3])`. Passing `(3,)` instead would
    work only in packed mode and break in the other.

    Compositing manually is exactly equivalent — gsplat's kernel does
    `pix_out += T * background` with `T` the final transmittance, and it
    returns that transmittance as `1 - alpha` — and it is correct in both
    modes, so nothing here depends on which one is selected.

    One shared helper for the evaluation renderer and the render loss: they
    must not drift, or the loss would be trained against one compositing
    convention and scored against another.
    """
    from gsplat import rasterization

    imgs, alphas, _meta = rasterization(
        means=means, quats=quats, scales=scales, opacities=opacities,
        colors=colors, viewmats=viewmats, Ks=Ks,
        width=image_size, height=image_size,
        render_mode="RGB", packed=True,
        # "antialiased" applies the Mip-Splatting 2D-filter compensation. The
        # ShapeSplats objects were optimised with the CLASSIC rasterizer
        # (standard Kerbl 3DGS), so classic is the mode that reproduces them.
        rasterize_mode="classic",
    )
    bg = torch.as_tensor(background, dtype=imgs.dtype, device=imgs.device)
    return (imgs + (1.0 - alphas) * bg).clamp(0.0, 1.0)


def gsplat_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import gsplat  # noqa: F401
        return True
    except ImportError:
        return False


@torch.no_grad()
def render_views(tok_raw, viewmats, Ks, image_size: int = 256, device=None,
                 backend: str = "auto", background=WHITE_BG):
    """Render one object from every camera in the rig.

    Returns `(images, backend_used)` where `images` is a float32 numpy
    array `(C, H, W, 3)` in [0, 1].
    """
    if backend == "auto":
        backend = "gsplat" if gsplat_available() else "points"
    if backend == "gsplat":
        return _render_gsplat(tok_raw, viewmats, Ks, image_size, device, background), "gsplat"
    return _render_points_fallback(tok_raw, viewmats, Ks, image_size), "points"


def _render_gsplat(tok_raw, viewmats, Ks, image_size, device, background):
    device = device or torch.device("cuda")
    means, quats, scales, opacities, colors = tokens_to_gaussians(tok_raw, device=device)
    vm = torch.as_tensor(viewmats, dtype=torch.float32, device=device)
    ks = torch.as_tensor(Ks, dtype=torch.float32, device=device)
    imgs = _gsplat_rgb(means, quats, scales, opacities, colors, vm, ks,
                       image_size, background)
    return imgs.float().cpu().numpy()


def _render_points_fallback(tok_raw, viewmats, Ks, image_size):
    """CPU path: a depth-sorted point scatter through **the rig's actual view
    matrices**.

    Deliberately not `splat_utils.render_points`, which orbits with its own
    `azim`/`elev` and auto-scales the frame to each object
    (`r = |p|.max()*1.05`). That makes every object fill the frame regardless
    of its true size, and it cannot reproduce a given `viewmats`/`Ks` pair at
    all — so a CPU figure and a GPU figure of the same object were framed
    differently, which is exactly the confusion this path should not add.
    Projecting through the real matrices means the two backends differ ONLY
    in what a splat looks like (ellipse vs circle, true compositing vs
    clipped alpha), not in where anything sits.

    Still not a rasterizer: no covariance, no orientation, no real alpha
    compositing. Metrics computed on these carry `backend="points"` and are
    a pipeline check, not results (see module docstring).
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    tok = np.asarray(tok_raw, dtype=np.float64)
    xyz = tok[:, 0:3]
    scale = np.exp(tok[:, 3:6])
    opacity = 1.0 / (1.0 + np.exp(-tok[:, 10]))
    rgb = np.clip(tok[:, 11:14], 0.0, 1.0)

    out = []
    for view, K in zip(np.asarray(viewmats, np.float64), np.asarray(Ks, np.float64)):
        cam = xyz @ view[:3, :3].T + view[:3, 3]
        z = np.maximum(cam[:, 2], 1e-6)
        uv = np.stack([K[0, 0] * cam[:, 0] / z + K[0, 2],
                       K[1, 1] * cam[:, 1] / z + K[1, 2]], axis=-1)

        order = np.argsort(-z)                      # far first, near painted on top
        # Screen-space radius from the true world scale and the focal length,
        # so a bigger Gaussian really is bigger on screen (matplotlib `s` is
        # area in points^2, hence the square).
        r_px = K[0, 0] * scale.mean(1)[order] / z[order]
        sizes = np.clip((2 * r_px) ** 2, 1.0, 4000.0)
        colors = np.concatenate(
            [rgb[order], np.clip(opacity[order], 0.02, 1.0)[:, None]], axis=-1)

        dpi = 100
        fig = Figure(figsize=(image_size / dpi, image_size / dpi), dpi=dpi)
        FigureCanvasAgg(fig)          # no global matplotlib state, same as render_points
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor("white")
        ax.scatter(uv[order, 0], uv[order, 1], s=sizes, c=colors,
                   linewidths=0, marker="o")
        ax.set_xlim(0, image_size)
        ax.set_ylim(image_size, 0)    # image convention: row 0 at the top
        ax.set_aspect("equal")
        ax.axis("off")
        fig.canvas.draw()
        out.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.float32) / 255.0)
    return np.stack(out)


# ----------------------------------------------------------------- differentiable path
# `render_views` above is `@torch.no_grad()` and takes numpy tokens — it is for
# evaluation and figures. The render LOSS needs the opposite: gradients must
# flow from the image back into the predicted Gaussian parameters. That is a
# different entry point, not a flag on the same one, because the no_grad
# decorator would silently make a render loss a no-op that trains nothing and
# throws no error.
def rasterize_differentiable(means, quats, scales, opacities, colors,
                             viewmats, Ks, image_size: int,
                             backend: str = "auto", background=WHITE_BG):
    """(N,3),(N,4),(N,3),(N,),(N,3) Gaussians -> (C, H, W, 3) images, with the
    graph intact. `backend="gsplat"` uses the CUDA rasterizer; `"torch"` uses
    the reference splatter below.
    """
    if backend == "auto":
        backend = "gsplat" if (torch.cuda.is_available() and means.is_cuda) else "torch"
    if backend == "gsplat":
        return _gsplat_rgb(means, quats, scales, opacities, colors,
                           viewmats, Ks, image_size, background)
    return _rasterize_torch(means, quats, scales, opacities, colors,
                            viewmats, Ks, image_size, background)


def _quat_to_rotmat(quats: torch.Tensor) -> torch.Tensor:
    """(N,4) w-first, unnormalized -> (N,3,3). Same convention as gsplat's
    "wxyz convension" and as `splat_utils.read_splat`."""
    q = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)


def _rasterize_torch(means, quats, scales, opacities, colors,
                     viewmats, Ks, image_size, background=WHITE_BG):
    """A compact, pure-PyTorch EWA splatter. Differentiable in every input.

    **This exists to be checkable, not to be fast.** gsplat is CUDA-only and
    login01 has no GPU, so without it the render loss's central claim —
    "gradients reach the predicted scale, opacity and colour" — could not be
    verified anywhere before committing GPU hours to a run that might be
    training on a disconnected graph. It computes every pixel against every
    Gaussian (no tiling), so cost is O(N x H x W): fine at the N~512, 32px of
    `check_render_loss.py`, hopeless at N=32768.

    gsplat ships its own reference implementation (`cuda/_torch_impl.py`) but
    `_rasterize_to_pixels` there requires `nerfacc`, which is not installed and
    is a large dependency to add for a test path.

    The maths is standard EWA splatting: project the centre, push the 3D
    covariance through the projection Jacobian to get a 2D covariance,
    evaluate each Gaussian at each pixel, alpha-composite front to back.
    """
    device, dtype = means.device, means.dtype
    C = viewmats.shape[0]
    H = W = image_size
    bg = torch.as_tensor(background, dtype=dtype, device=device)

    R_q = _quat_to_rotmat(quats)                                   # (N,3,3)
    S = torch.diag_embed(scales)
    cov3d = R_q @ S @ S @ R_q.transpose(1, 2)                      # (N,3,3) = R S^2 R^T

    ys, xs = torch.meshgrid(torch.arange(H, device=device, dtype=dtype),
                            torch.arange(W, device=device, dtype=dtype),
                            indexing="ij")
    pix = torch.stack([xs, ys], dim=-1).reshape(-1, 2)             # (P,2)

    out = []
    for c in range(C):
        Rc, tc = viewmats[c, :3, :3], viewmats[c, :3, 3]
        fx, fy = Ks[c, 0, 0], Ks[c, 1, 1]
        cx, cy = Ks[c, 0, 2], Ks[c, 1, 2]

        p_cam = means @ Rc.T + tc                                  # (N,3)
        z = p_cam[:, 2].clamp(min=1e-4)
        u = fx * p_cam[:, 0] / z + cx
        v = fy * p_cam[:, 1] / z + cy
        mean2d = torch.stack([u, v], dim=-1)                       # (N,2)

        # Projection Jacobian, then the covariance through world->cam->screen.
        zero = torch.zeros_like(z)
        J = torch.stack([
            torch.stack([fx / z, zero, -fx * p_cam[:, 0] / z ** 2], dim=-1),
            torch.stack([zero, fy / z, -fy * p_cam[:, 1] / z ** 2], dim=-1),
        ], dim=1)                                                   # (N,2,3)
        T = J @ Rc                                                  # (N,2,3)
        cov2d = T @ cov3d @ T.transpose(1, 2)
        # gsplat's `eps2d=0.3` default: a floor on the screen-space size, so a
        # Gaussian smaller than a pixel still covers one and still gets
        # gradient. Without it sub-pixel Gaussians vanish from the loss.
        cov2d = cov2d + 0.3 * torch.eye(2, device=device, dtype=dtype)

        det = (cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0])
        det = det.clamp(min=1e-8)
        conic = torch.stack([cov2d[:, 1, 1] / det, -cov2d[:, 0, 1] / det,
                             cov2d[:, 0, 0] / det], dim=-1)         # (N,3) [a,b,c]

        d = pix[:, None, :] - mean2d[None, :, :]                    # (P,N,2)
        power = -0.5 * (conic[None, :, 0] * d[..., 0] ** 2
                        + 2 * conic[None, :, 1] * d[..., 0] * d[..., 1]
                        + conic[None, :, 2] * d[..., 1] ** 2)
        alpha = (opacities[None, :] * torch.exp(power.clamp(max=0.0))).clamp(0.0, 0.999)
        # Behind the camera contributes nothing, and must not contribute a
        # gradient either.
        alpha = alpha * (p_cam[:, 2] > 1e-4).to(dtype)[None, :]

        order = torch.argsort(z)                                    # front to back
        a = alpha[:, order]
        col = colors[order]
        # Front-to-back compositing: weight_i = alpha_i * prod_{j<i}(1-alpha_j).
        # cumprod is exclusive here (shift by one) so a Gaussian is not
        # occluded by itself.
        trans = torch.cumprod(1.0 - a + 1e-10, dim=1)
        trans = torch.cat([torch.ones_like(trans[:, :1]), trans[:, :-1]], dim=1)
        weight = a * trans                                          # (P,N)
        img = weight @ col + (1.0 - weight.sum(1, keepdim=True)).clamp(min=0.0) * bg
        out.append(img.reshape(H, W, 3))
    return torch.stack(out).clamp(0.0, 1.0)


def render_grid_image(images: np.ndarray, max_views: int = 4) -> np.ndarray:
    """(C,H,W,3) -> one (H, W*k, 3) strip, for dropping into a figure panel."""
    k = min(max_views, images.shape[0])
    step = max(1, images.shape[0] // k)
    return np.concatenate([images[i * step] for i in range(k)], axis=1)
