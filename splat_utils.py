"""
splat_utils.py — read and inspect 3D Gaussian Splat .ply files.

No classes, no config files, no installation. Just functions.
Put this next to your script/notebook and `import splat_utils`.

    splat = read_splat("chair_0001.ply")
    splat["xyz"], splat["scale"], splat["opacity"], splat["rgb"]   # all activated
"""

from pathlib import Path

import numpy as np

# SH degree-0 basis constant: rgb = 0.5 + SH_C0 * f_dc
SH_C0 = 0.28209479177387814


def _numbered(names, prefix):
    """Property names like scale_0, scale_1, ... sorted NUMERICALLY.

    Sorting as plain strings puts f_rest_10 before f_rest_2 and scrambles
    anything with more than 10 numbered properties.
    """
    sel = [n for n in names if n.startswith(prefix) and n[len(prefix):].isdigit()]
    return sorted(sel, key=lambda n: int(n[len(prefix):]))


def read_splat(path):
    """Load one .ply into a dict of numpy arrays, activations already applied.

    Returns keys: name, xyz, scale, log_scale, quat, opacity, opacity_logit,
    rgb, f_dc, n. Everything is float32.

    The three things a splat loader gets wrong silently:
      - opacity is stored as a LOGIT  -> apply sigmoid
      - scale is stored as a LOG      -> apply exp
      - the quaternion is UNNORMALIZED and w-first (rot_0 == w, not x)
    This function applies all three.
    """
    from plyfile import PlyData

    path = Path(path)
    v = PlyData.read(str(path))["vertex"].data
    names = list(v.dtype.names)

    def col(*keys):
        return np.stack([np.asarray(v[k], np.float32) for k in keys], axis=-1)

    xyz = col("x", "y", "z")
    log_scale = col(*_numbered(names, "scale_"))
    quat = col(*_numbered(names, "rot_"))
    quat = quat / np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-12)
    f_dc = col(*_numbered(names, "f_dc_"))
    opacity_logit = np.asarray(v["opacity"], np.float32)

    return dict(
        name=path.stem,
        n=len(xyz),
        xyz=xyz,
        log_scale=log_scale,
        scale=np.exp(log_scale),
        quat=quat,
        opacity_logit=opacity_logit,
        opacity=1.0 / (1.0 + np.exp(-opacity_logit)),
        f_dc=f_dc,
        rgb=np.clip(0.5 + SH_C0 * f_dc, 0.0, 1.0),
    )


def sh_rest_energy_ratio(path):
    """Per-Gaussian ratio of higher-order SH energy to DC energy.

    `f_rest` is stored **channel-major** in the PLY: the 45 properties
    reshape as `(N, 3, 15)` (3 colour channels x 15 higher-order
    coefficients for SH degree <=3), NOT `(N, 15, 3)` then transposed —
    see CLAUDE.md's loader pitfalls. Getting this wrong doesn't crash,
    it just silently scrambles which energy belongs to which channel.

    ratio = sum(f_rest**2) / sum(f_dc**2), per Gaussian. Not loaded by
    read_splat() since most of the pipeline drops f_rest entirely
    (DESIGN_SPEC §3.2) — this is only for the EDA decision of whether to.
    """
    from plyfile import PlyData

    path = Path(path)
    v = PlyData.read(str(path))["vertex"].data
    names = list(v.dtype.names)

    def col(*keys):
        return np.stack([np.asarray(v[k], np.float32) for k in keys], axis=-1)

    f_dc = col(*_numbered(names, "f_dc_"))                      # (N, 3)
    f_rest = col(*_numbered(names, "f_rest_")).reshape(-1, 3, 15)  # (N, 3, 15) channel-major

    dc_energy = (f_dc ** 2).sum(-1)
    rest_energy = (f_rest ** 2).sum((1, 2))
    return rest_energy / np.maximum(dc_energy, 1e-12)


def find_ply_files(root, prefix=None):
    """All .ply files under `root` (recursive), sorted, optionally filtered
    to filenames starting with `prefix`."""
    root = Path(root)
    files = sorted(root.rglob("*.ply"))
    if prefix:
        files = [f for f in files if f.stem.startswith(prefix)]
    return files


def object_stats(splat):
    """Reduce one loaded splat to a flat dict of scalars — one CSV row."""
    xyz, scale, opacity, rgb = splat["xyz"], splat["scale"], splat["opacity"], splat["rgb"]
    lo, hi = xyz.min(0), xyz.max(0)
    center = 0.5 * (lo + hi)
    extent = hi - lo
    s_sorted = np.sort(scale, axis=1)
    aniso = s_sorted[:, 2] / np.maximum(s_sorted[:, 0], 1e-12)
    sat = rgb.max(1) - rgb.min(1)

    return dict(
        name=splat["name"],
        n_gaussians=splat["n"],
        center_x=float(center[0]), center_y=float(center[1]), center_z=float(center[2]),
        extent_x=float(extent[0]), extent_y=float(extent[1]), extent_z=float(extent[2]),
        radius=float(np.abs(xyz - center).max()),
        dominant_axis="xyz"[int(np.argmax(extent))],
        log_scale_mean=float(splat["log_scale"].mean()),
        log_scale_std=float(splat["log_scale"].std()),
        scale_median=float(np.median(scale)),
        aniso_median=float(np.median(aniso)),
        opacity_mean=float(opacity.mean()),
        frac_opacity_lt01=float((opacity < 0.01).mean()),
        rgb_mean=float(rgb.mean()),
        saturation_median=float(np.median(sat)),
        frac_quat_w_negative=float((splat["quat"][:, 0] < 0).mean()),
    )


def render_points(splat, azim=35.0, elev=18.0, size=400, up_axis=1,
                   max_points=60_000, seed=0):
    """A quick point-cloud render — good enough to confirm 'is this a chair'.

    Not a real splat rasterizer (no covariance, no occlusion beyond
    painter's-algorithm depth sort). Returns an (size, size, 3) float image
    in [0, 1]. Point size is driven by mean scale, alpha by opacity.

    Camera: orthographic, orbiting the object's bbox centre. `azim` rotates
    around `up_axis`, `elev` tilts up from the horizontal plane.

    Builds its own Figure via the Agg canvas directly rather than through
    pyplot, so it never touches the global matplotlib backend. Calling this
    from inside a notebook using `%matplotlib inline` is safe; a stray
    `matplotlib.use("Agg")` in here would silently kill inline plotting for
    every cell that runs afterwards.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    xyz, scale, opacity, rgb = splat["xyz"], splat["scale"], splat["opacity"], splat["rgb"]
    n = len(xyz)
    if n > max_points:
        idx = np.random.default_rng(seed).choice(n, max_points, replace=False)
        xyz, scale, opacity, rgb = xyz[idx], scale[idx], opacity[idx], rgb[idx]

    center = 0.5 * (splat["xyz"].min(0) + splat["xyz"].max(0))
    p = xyz - center

    up = np.zeros(3, np.float32)
    up[up_axis] = 1.0
    other = [i for i in range(3) if i != up_axis]
    a, e = np.radians(azim), np.radians(elev)

    horiz = np.zeros(3, np.float32)
    horiz[other[0]] = np.cos(a)
    horiz[other[1]] = np.sin(a)
    eye_dir = horiz * np.cos(e) + up * np.sin(e)          # centre -> camera, unit
    eye_dir /= np.linalg.norm(eye_dir)
    forward = -eye_dir                                     # camera -> centre

    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:                       # looking straight along `up`
        right = np.cross(forward, np.array([1.0, 0.0, 0.0], np.float32))
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)

    screen_x = p @ right
    screen_y = p @ true_up
    depth = p @ forward                                    # larger = farther from camera

    order = np.argsort(depth)[::-1]                        # far first, near last (on top)
    sizes = np.clip(scale.mean(1)[order] * 4000, 1.0, 60.0)
    colors = np.concatenate([rgb[order], np.clip(opacity[order], 0.05, 1.0)[:, None]], -1)

    dpi = 100
    fig = Figure(figsize=(size / dpi, size / dpi), dpi=dpi)
    FigureCanvasAgg(fig)  # attaches an Agg canvas to `fig`, no global state
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("white")
    ax.scatter(screen_x[order], screen_y[order], s=sizes, c=colors,
               linewidths=0, marker="o")
    r = np.abs(p).max() * 1.05
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.float32) / 255.0
    return img
