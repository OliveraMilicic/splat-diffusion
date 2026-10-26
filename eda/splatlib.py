"""
splatlib -- minimal 3D Gaussian Splatting toolkit for the ShapeSplats dataset.

Pure numpy + plyfile. No CUDA, no compilation, no gsplat.
Everything here runs on CPU in seconds so you can iterate inside a notebook.

Contents
--------
  Loading      find_objects, load_ply, Splats
  Geometry     quat_to_rotmat, covariance_3d
  Camera       look_at, orbit_camera
  Rendering    project_gaussians, rasterize, render
  Analysis     object_stats, gaussian_sample

Conventions (standard 3DGS / INRIA format)
------------------------------------------
  Stored in the PLY (raw, pre-activation):
      x,y,z             mean position
      nx,ny,nz          normals -- usually all zero, ignored
      f_dc_0..2         SH degree-0 coefficients (base colour)
      f_rest_0..N       higher-order SH coefficients (view dependent)
      opacity           LOGIT of opacity   -> sigmoid()
      scale_0..2        LOG of scale       -> exp()
      rot_0..3          quaternion (w,x,y,z), unnormalised -> normalise

  Applying those activations is not optional. Reading `scale_0` directly and
  calling it a size gives you a number around -5, not a length.
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# SH degree-0 basis function, 0.5 / sqrt(pi). Converts the DC coefficient to RGB.
SH_C0 = 0.28209479177387814


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def find_objects(root, category="chair", split=None, pattern="*.ply"):
    """Recursively find PLY files for one category, sorted by index.

    Works whether the layout is  train/chair_0001.ply  or
    train/chair/chair_0001.ply.
    """
    root = Path(root)
    if split:
        root = root / split
    files = [p for p in root.rglob(pattern) if p.stem.startswith(category)]

    def key(p):
        m = re.search(r"(\d+)", p.stem)
        return (int(m.group(1)) if m else 0, p.stem)

    return sorted(files, key=key)


def _numbered(names, prefix):
    """Property names with a numeric suffix, sorted NUMERICALLY.

    Lexical sort puts f_rest_10 before f_rest_2 and silently scrambles the
    spherical harmonics. This is the single easiest way to corrupt a loader.
    """
    sel = [n for n in names if n.startswith(prefix) and n[len(prefix):].isdigit()]
    return sorted(sel, key=lambda n: int(n[len(prefix):]))


@dataclass
class Splats:
    """One object's Gaussians, with activations already applied."""
    xyz: np.ndarray          # (N,3)  world position
    scale: np.ndarray        # (N,3)  world-space std devs (exp applied)
    quat: np.ndarray         # (N,4)  unit quaternion, w first
    opacity: np.ndarray      # (N,)   in [0,1] (sigmoid applied)
    rgb: np.ndarray          # (N,3)  in [0,1], from the SH DC term
    f_dc: np.ndarray         # (N,3)  raw DC coefficients
    f_rest: np.ndarray       # (N,K)  raw higher-order SH (K may be 0)
    raw_scale: np.ndarray    # (N,3)  log-scale, as stored
    raw_opacity: np.ndarray  # (N,)   logit-opacity, as stored
    name: str = ""

    def __len__(self):
        return len(self.xyz)

    @property
    def sh_degree(self):
        n_coef = 1 + self.f_rest.shape[1] // 3
        return int(round(np.sqrt(n_coef))) - 1

    def summary(self):
        s = self.scale
        return {
            "name": self.name,
            "n_gaussians": len(self),
            "sh_degree": self.sh_degree,
            "n_sh_rest": self.f_rest.shape[1],
            "bbox_min": self.xyz.min(0).round(4).tolist(),
            "bbox_max": self.xyz.max(0).round(4).tolist(),
            "extent": (self.xyz.max(0) - self.xyz.min(0)).round(4).tolist(),
            "centroid": self.xyz.mean(0).round(4).tolist(),
            "scale_median": np.median(s, 0).round(5).tolist(),
            "opacity_mean": float(self.opacity.mean().round(4)),
            "rgb_mean": self.rgb.mean(0).round(3).tolist(),
        }


def load_ply(path, apply_activations=True):
    """Read a 3DGS PLY into a `Splats` object."""
    from plyfile import PlyData

    path = Path(path)
    v = PlyData.read(str(path))["vertex"].data
    names = list(v.dtype.names)

    def col(*keys):
        return np.stack([np.asarray(v[k], np.float32) for k in keys], -1)

    xyz = col("x", "y", "z")

    dc_names = _numbered(names, "f_dc_")
    f_dc = col(*dc_names) if dc_names else np.zeros((len(v), 3), np.float32)

    rest_names = _numbered(names, "f_rest_")
    f_rest = col(*rest_names) if rest_names else np.zeros((len(v), 0), np.float32)

    raw_scale = col(*_numbered(names, "scale_"))
    raw_quat = col(*_numbered(names, "rot_"))
    raw_opacity = np.asarray(v["opacity"], np.float32)

    if apply_activations:
        scale = np.exp(raw_scale)
        opacity = 1.0 / (1.0 + np.exp(-raw_opacity))
        norm = np.linalg.norm(raw_quat, axis=1, keepdims=True)
        quat = raw_quat / np.maximum(norm, 1e-12)
    else:
        scale, opacity, quat = raw_scale, raw_opacity, raw_quat

    rgb = np.clip(0.5 + SH_C0 * f_dc, 0.0, 1.0)

    return Splats(
        xyz=xyz, scale=scale, quat=quat, opacity=opacity, rgb=rgb,
        f_dc=f_dc, f_rest=f_rest, raw_scale=raw_scale, raw_opacity=raw_opacity,
        name=path.stem,
    )


def describe_ply(path, n_preview=6):
    """Print the raw PLY structure. Run this once before trusting any loader."""
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    print(f"file    : {Path(path).name}")
    print(f"format  : {ply.header.splitlines()[1] if len(ply.header.splitlines()) > 1 else '?'}")
    for el in ply.elements:
        print(f"element : {el.name}  count={el.count}  properties={len(el.data.dtype.names)}")
        names = list(el.data.dtype.names)
        groups = {}
        for n in names:
            g = re.sub(r"\d+$", "*", n)
            groups.setdefault(g, []).append(n)
        for g, members in groups.items():
            print(f"    {g:<12} x{len(members):<4} e.g. {members[:3]}")
        print(f"\n  first {n_preview} rows, first 8 columns:")
        sub = el.data[:n_preview]
        print("   ", "  ".join(f"{n:>10}" for n in names[:8]))
        for row in sub:
            print("   ", "  ".join(f"{row[n]:>10.4f}" for n in names[:8]))
    return ply


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def quat_to_rotmat(q):
    """(N,4) unit quaternions (w,x,y,z) -> (N,3,3) rotation matrices."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(-1, 3, 3).astype(np.float32)


def covariance_3d(scale, quat):
    """Sigma = R S S^T R^T, built as M M^T with M = R @ diag(s).

    This is the whole geometric content of a Gaussian: an ellipsoid described
    by an orientation (quat) and three axis lengths (scale).
    """
    R = quat_to_rotmat(quat)
    M = R * scale[:, None, :]          # scale the COLUMNS of R
    return M @ M.transpose(0, 2, 1)


def canonicalize_quat(q):
    """Force w >= 0.

    q and -q are the same rotation (the double cover of SO(3)). If you train a
    network to regress raw quaternions without fixing this, identical geometry
    appears under two different targets and the loss is inconsistent.
    """
    return np.where(q[:, :1] < 0, -q, q)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    """World -> camera. Returns (R, t) with camera looking down +z, image y down."""
    eye = np.asarray(eye, np.float32)
    target = np.asarray(target, np.float32)
    up = np.asarray(up, np.float32)

    f = target - eye
    f /= np.linalg.norm(f) + 1e-12
    r = np.cross(up, f)
    if np.linalg.norm(r) < 1e-6:                  # camera on the up axis
        r = np.cross(np.array([1.0, 0.0, 0.0], np.float32), f)
    r /= np.linalg.norm(r) + 1e-12
    d = np.cross(r, f)                            # image-down direction

    R = np.stack([r, d, f], 0).astype(np.float32)
    t = -R @ eye
    return R, t


def orbit_camera(center, radius, azim_deg, elev_deg, up_axis=1):
    """Camera position on a sphere around `center`.

    up_axis: 0=x, 1=y, 2=z. ModelNet is often z-up while renderers assume
    y-up -- if your chairs come out lying on their side, change this.
    """
    a, e = np.radians(azim_deg), np.radians(elev_deg)
    horiz, vert = np.cos(e) * radius, np.sin(e) * radius
    off = np.zeros(3, np.float32)
    axes = [i for i in range(3) if i != up_axis]
    off[axes[0]] = horiz * np.cos(a)
    off[axes[1]] = horiz * np.sin(a)
    off[up_axis] = vert
    up = np.zeros(3, np.float32)
    up[up_axis] = 1.0
    return np.asarray(center, np.float32) + off, up


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def project_gaussians(splats, R, t, fx, fy, cx, cy, near=0.01, dilation=0.3):
    """Project 3D Gaussians to 2D screen-space Gaussians (EWA splatting).

    The 3x3 world covariance becomes a 2x2 screen covariance via the local
    affine approximation of the perspective projection:

        Sigma_2D  =  J W Sigma W^T J^T

    where W is the world->camera rotation and J the Jacobian of the projection
    evaluated at the Gaussian's centre.
    """
    xyz_cam = splats.xyz @ R.T + t
    Z = xyz_cam[:, 2]
    keep = Z > near
    if not keep.any():
        return None

    xyz_cam, Z = xyz_cam[keep], Z[keep]
    X, Y = xyz_cam[:, 0], xyz_cam[:, 1]

    u = fx * X / Z + cx
    v = fy * Y / Z + cy

    n = len(Z)
    J = np.zeros((n, 2, 3), np.float32)
    J[:, 0, 0] = fx / Z
    J[:, 0, 2] = -fx * X / (Z * Z)
    J[:, 1, 1] = fy / Z
    J[:, 1, 2] = -fy * Y / (Z * Z)

    cov3d = covariance_3d(splats.scale[keep], splats.quat[keep])
    T = J @ R[None]                                   # (n,2,3)
    cov2d = T @ cov3d @ T.transpose(0, 2, 1)          # (n,2,2)

    # Low-pass filter: keeps sub-pixel Gaussians from vanishing or aliasing.
    cov2d[:, 0, 0] += dilation
    cov2d[:, 1, 1] += dilation

    a, b, c = cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]
    det = a * c - b * b
    ok = det > 1e-10
    inv_det = np.where(ok, 1.0 / np.maximum(det, 1e-10), 0.0)
    inv_cov = np.stack([c * inv_det, -b * inv_det, a * inv_det], -1)   # (A,B,C)

    mid = 0.5 * (a + c)
    delta = np.sqrt(np.maximum(mid * mid - det, 0.0))
    radius = 3.0 * np.sqrt(np.maximum(mid + delta, 1e-8))

    return dict(u=u[ok], v=v[ok], inv_cov=inv_cov[ok], radius=radius[ok],
                depth=Z[ok], rgb=splats.rgb[keep][ok],
                opacity=splats.opacity[keep][ok])


def rasterize(proj, H, W, n_slabs=64, bg=1.0, max_patch=41, alpha_cutoff=1.0 / 255):
    """Alpha-composite projected Gaussians into an image.

    Exact front-to-back compositing is inherently sequential, which is
    unusably slow in Python. Instead we slice depth into `n_slabs` thin
    slabs; within a slab, order barely matters, so we use order-independent
    blending (transmittance multiplies, so sum of log(1-alpha) is exact),
    then composite the slabs sequentially. n_slabs=64 is visually
    indistinguishable from a real rasteriser for objects like these.
    """
    u, v = proj["u"], proj["v"]
    A, B, C = proj["inv_cov"][:, 0], proj["inv_cov"][:, 1], proj["inv_cov"][:, 2]
    alpha_g, rgb_g, depth = proj["opacity"], proj["rgb"], proj["depth"]

    P = int(2 * np.ceil(np.percentile(proj["radius"], 99)) + 1)
    P = int(np.clip(P, 3, max_patch))
    r = (P - 1) // 2
    oy, ox = np.mgrid[-r:r + 1, -r:r + 1]
    ox, oy = ox.ravel(), oy.ravel()

    order = np.argsort(depth)
    slabs = np.array_split(order, n_slabs)

    img = np.zeros((H * W, 3), np.float32)
    trans = np.ones(H * W, np.float32)
    npix = H * W

    for sl in slabs:
        if len(sl) == 0:
            continue
        cu, cv = np.round(u[sl]).astype(np.int64), np.round(v[sl]).astype(np.int64)
        px = cu[:, None] + ox[None, :]
        py = cv[:, None] + oy[None, :]
        dx = px - u[sl][:, None]
        dy = py - v[sl][:, None]

        power = -0.5 * (A[sl][:, None] * dx * dx
                        + 2.0 * B[sl][:, None] * dx * dy
                        + C[sl][:, None] * dy * dy)
        w = alpha_g[sl][:, None] * np.exp(np.minimum(power, 0.0))

        good = ((px >= 0) & (px < W) & (py >= 0) & (py < H) & (w > alpha_cutoff))
        if not good.any():
            continue

        gi = np.broadcast_to(np.arange(len(sl))[:, None], w.shape)[good]
        idx = (py * W + px)[good]
        ww = np.minimum(w[good], 0.999)
        cols = rgb_g[sl][gi]

        acc_w = np.bincount(idx, ww, minlength=npix)
        acc_wc = np.stack(
            [np.bincount(idx, ww * cols[:, k], minlength=npix) for k in range(3)], -1)
        acc_logt = np.bincount(idx, np.log1p(-ww), minlength=npix)

        slab_T = np.exp(acc_logt)
        slab_a = 1.0 - slab_T
        hit = acc_w > 1e-8
        slab_rgb = np.zeros_like(acc_wc)
        slab_rgb[hit] = acc_wc[hit] / acc_w[hit, None]

        img += (trans * slab_a)[:, None] * slab_rgb
        trans *= slab_T

    img += trans[:, None] * bg
    return np.clip(img.reshape(H, W, 3), 0.0, 1.0)


def render(splats, azim=30.0, elev=20.0, H=384, W=384, fov_deg=40.0,
           zoom=1.0, up_axis=1, bg=1.0, center=None, radius=None, **kw):
    """Render one object from an orbit camera. Returns an (H,W,3) float image."""
    if center is None:
        center = 0.5 * (splats.xyz.min(0) + splats.xyz.max(0))
    if radius is None:
        radius = float(np.linalg.norm(splats.xyz - center, axis=1).max())

    dist = radius / np.tan(np.radians(fov_deg) / 2) / max(zoom, 1e-6)
    eye, up = orbit_camera(center, dist, azim, elev, up_axis=up_axis)
    R, t = look_at(eye, center, up)

    f = 0.5 * H / np.tan(np.radians(fov_deg) / 2)
    proj = project_gaussians(splats, R, t, f, f, W / 2, H / 2)
    if proj is None or len(proj["u"]) == 0:
        return np.full((H, W, 3), bg, np.float32)
    return rasterize(proj, H, W, bg=bg, **kw)


def render_grid(splats, azims=(0, 90, 180, 270), elev=20.0, **kw):
    return [render(splats, azim=a, elev=elev, **kw) for a in azims]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def object_stats(splats):
    """Per-object summary row. Cheap enough to run over the whole category."""
    s, xyz, op = splats.scale, splats.xyz, splats.opacity
    center = 0.5 * (xyz.min(0) + xyz.max(0))
    rad = np.linalg.norm(xyz - center, axis=1)
    ssort = np.sort(s, axis=1)                        # ascending per Gaussian
    aniso = ssort[:, 2] / np.maximum(ssort[:, 0], 1e-12)
    rest = splats.f_rest

    row = dict(
        name=splats.name,
        n=len(splats),
        sh_degree=splats.sh_degree,
        # extent / placement
        cx=float(center[0]), cy=float(center[1]), cz=float(center[2]),
        ext_x=float(np.ptp(xyz[:, 0])), ext_y=float(np.ptp(xyz[:, 1])),
        ext_z=float(np.ptp(xyz[:, 2])),
        radius_max=float(rad.max()), radius_p99=float(np.percentile(rad, 99)),
        centroid_off=float(np.linalg.norm(xyz.mean(0) - center)),
        # scale (log space is the natural parameterisation)
        logs_mean=float(splats.raw_scale.mean()),
        logs_std=float(splats.raw_scale.std()),
        s_med=float(np.median(s)),
        s_p99=float(np.percentile(s, 99)),
        aniso_med=float(np.median(aniso)),
        aniso_p99=float(np.percentile(aniso, 99)),
        # opacity
        op_mean=float(op.mean()),
        op_med=float(np.median(op)),
        frac_op_lt01=float((op < 0.01).mean()),
        frac_op_lt05=float((op < 0.05).mean()),
        frac_op_gt50=float((op > 0.50).mean()),
        # appearance
        rgb_mean=float(splats.rgb.mean()),
        rgb_std_within=float(splats.rgb.std(0).mean()),
        sat_med=float(np.median(splats.rgb.max(1) - splats.rgb.min(1))),
        rest_absmean=float(np.abs(rest).mean()) if rest.size else 0.0,
        rest_energy_ratio=(float((rest ** 2).sum() / max((splats.f_dc ** 2).sum(), 1e-12))
                           if rest.size else 0.0),
        # rotation
        frac_qw_neg=float((splats.quat[:, 0] < 0).mean()),
    )
    return row


def gaussian_sample(files, per_file=2000, seed=0, loader=load_ply):
    """Pool a random subset of Gaussians across many objects.

    Per-object stats hide within-object structure; this gives you the
    population distribution without loading 889 files into RAM at once.
    """
    rng = np.random.default_rng(seed)
    out = {k: [] for k in
           ["xyz", "scale", "raw_scale", "opacity", "raw_opacity", "quat", "rgb", "obj"]}
    for i, f in enumerate(files):
        sp = loader(f)
        k = min(per_file, len(sp))
        idx = rng.choice(len(sp), k, replace=False)
        out["xyz"].append(sp.xyz[idx])
        out["scale"].append(sp.scale[idx])
        out["raw_scale"].append(sp.raw_scale[idx])
        out["opacity"].append(sp.opacity[idx])
        out["raw_opacity"].append(sp.raw_opacity[idx])
        out["quat"].append(sp.quat[idx])
        out["rgb"].append(sp.rgb[idx])
        out["obj"].append(np.full(k, i, np.int32))
    return {k: np.concatenate(v, 0) for k, v in out.items()}


def voxel_occupancy(xyz, res, normalize=True):
    """Occupancy and collision statistics for a res^3 grid.

    Directly relevant to GaussianCube-style methods: they map an unordered set
    of Gaussians onto a fixed voxel grid. If N >> occupied voxels, the mapping
    has to merge or drop Gaussians.
    """
    p = xyz.astype(np.float64)
    if normalize:
        c = 0.5 * (p.min(0) + p.max(0))
        p = p - c
        s = np.abs(p).max()
        p = p / (s + 1e-12)                            # -> [-1,1]
    idx = np.clip(((p + 1) * 0.5 * res).astype(np.int64), 0, res - 1)
    flat = (idx[:, 0] * res + idx[:, 1]) * res + idx[:, 2]
    counts = np.bincount(flat, minlength=res ** 3)
    occ = counts[counts > 0]
    return dict(
        res=res,
        n_gaussians=len(xyz),
        occupied=int(len(occ)),
        occupancy_frac=float(len(occ) / res ** 3),
        capacity_ratio=float(len(xyz) / res ** 3),
        per_voxel_mean=float(occ.mean()) if len(occ) else 0.0,
        per_voxel_max=int(occ.max()) if len(occ) else 0,
        frac_in_singleton=float((counts[flat] == 1).mean()),
    )


def save_norm_stats(sample, path):
    """Per-attribute mean/std for normalising diffusion model inputs."""
    stats = {
        "xyz": dict(mean=sample["xyz"].mean(0).tolist(), std=sample["xyz"].std(0).tolist()),
        "log_scale": dict(mean=sample["raw_scale"].mean(0).tolist(),
                          std=sample["raw_scale"].std(0).tolist()),
        "logit_opacity": dict(mean=float(sample["raw_opacity"].mean()),
                              std=float(sample["raw_opacity"].std())),
        "quat_canonical": dict(mean=canonicalize_quat(sample["quat"]).mean(0).tolist(),
                               std=canonicalize_quat(sample["quat"]).std(0).tolist()),
        "rgb": dict(mean=sample["rgb"].mean(0).tolist(), std=sample["rgb"].std(0).tolist()),
    }
    Path(path).write_text(json.dumps(stats, indent=2))
    return stats
