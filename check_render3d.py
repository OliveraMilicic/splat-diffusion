"""
check_render3d.py — checks for `eval/render3d.py`'s camera rig and token
decode, WITHOUT needing a GPU.

`gsplat` is CUDA-only and login01 has no `/dev/nvidia*`, so the rasterizer
itself can only be exercised in the GPU container. What can be checked
here is the part that is easy to get silently wrong and that the
rasterizer would not complain about: the world->camera matrix. A flipped
`y` sign renders every chair upside down, a wrong up-axis renders them on
their backs, and a wrong distance renders them clipped — none of which
raises. So the projection is recomputed by hand in numpy and asserted
against the geometry we already know (up axis = z, object inscribed in
the unit sphere).

Run: python check_render3d.py
"""

import container_env  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval.render3d import (FIT_RADIUS, UP_AXIS, orbit_cameras, gsplat_available,
                            render_views, tokens_to_gaussians)
from preprocessing import full_object_tokens
from splat_utils import find_ply_files, read_splat
from stage_config import StageConfig, load_split

IMAGE_SIZE = 256
N_VIEWS = 8

cfg = StageConfig()
split = load_split(cfg.shard_path)
files = {f.stem: f for f in find_ply_files(cfg.data_dir)}
name = split["val"][0]
tok, centre, radius = full_object_tokens(read_splat(files[name]))
print(f"object {name[:32]}  N={len(tok)}  radius={radius:.4f}")

viewmats, Ks = orbit_cameras(n_views=N_VIEWS, image_size=IMAGE_SIZE)
print(f"rig: {viewmats.shape[0]} cameras ({N_VIEWS} azimuths x "
      f"{viewmats.shape[0] // N_VIEWS} elevations), {IMAGE_SIZE}px")


def project(points, view, K):
    """world (N,3) -> pixel (N,2) + camera-space depth (N,)."""
    cam = points @ view[:3, :3].T + view[:3, 3]
    z = cam[:, 2]
    uv = (cam / z[:, None]) @ K.T
    return uv[:, :2], z


xyz = tok[:, 0:3].astype(np.float64)

# ----------------------------------------------------------------- C1
print("\n[check C1] the object is inscribed in the unit CUBE in this frame")
r = np.abs(xyz).max()
norm_max = np.linalg.norm(xyz, axis=1).max()
print(f"    max |coord| = {r:.4f}  (normalize_object makes this exactly 1 on one axis)")
print(f"    max ||xyz|| = {norm_max:.4f}  (up to sqrt(3)={np.sqrt(3):.4f} — the rig frames for this,"
      f" NOT for 1.0)")
assert 0.9 <= r <= 1.001, "not the per-object normalized frame — the rig assumes it"
assert norm_max <= FIT_RADIUS, "object exceeds the rig's framing radius — it will clip"
print("    PASS")

# ----------------------------------------------------------------- C2
print("\n[check C2] every camera sees the whole object, in front of it, in frame")
worst_in_frame, worst_margin = 1.0, 1e9
for i, (view, K) in enumerate(zip(viewmats, Ks)):
    uv, z = project(xyz, view.astype(np.float64), K.astype(np.float64))
    assert (z > 0).all(), f"camera {i}: {int((z <= 0).sum())} points BEHIND the camera"
    in_frame = ((uv >= 0) & (uv <= IMAGE_SIZE)).all(1).mean()
    margin = min(uv.min(), IMAGE_SIZE - uv.max())
    worst_in_frame = min(worst_in_frame, in_frame)
    worst_margin = min(worst_margin, margin)
print(f"    worst-case points in frame: {worst_in_frame:.4f} (must be 1.0)")
print(f"    worst-case border margin:   {worst_margin:.1f}px "
      f"(>0 = nothing clipped; the rig targets ~20% margin)")
assert worst_in_frame == 1.0
assert worst_margin > 0
print("    PASS")

# ----------------------------------------------------------------- C3
print("\n[check C3] up is UP: +z in the world maps to DECREASING pixel row")
# The trap this catches: OpenCV image coordinates have y pointing DOWN, so
# the view matrix's second row must be -true_up. With +true_up every render
# is vertically mirrored, which a rasterizer accepts happily and which is
# invisible in a metric (a flipped GT and a flipped prediction still match).
top_idx = xyz[:, UP_AXIS].argmax()
bot_idx = xyz[:, UP_AXIS].argmin()
for i, (view, K) in enumerate(zip(viewmats, Ks)):
    uv, _ = project(xyz[[top_idx, bot_idx]], view.astype(np.float64), K.astype(np.float64))
    assert uv[0, 1] < uv[1, 1], (
        f"camera {i}: the highest point renders BELOW the lowest — view matrix "
        f"y-sign is flipped")
print(f"    all {len(viewmats)} cameras: highest world point renders above the lowest")
print("    PASS")

# ----------------------------------------------------------------- C4
print("\n[check C4] azimuths are distinct views, not the same one repeated")
centroids = []
for view, K in zip(viewmats[:N_VIEWS], Ks[:N_VIEWS]):
    uv, _ = project(xyz, view.astype(np.float64), K.astype(np.float64))
    centroids.append(uv.mean(0))
spread = np.ptp(np.stack(centroids), axis=0)
# A stronger signal than centroid spread (a symmetric chair barely moves its
# centroid): the silhouette itself must change between views.
def silhouette(view, K):
    uv, _ = project(xyz, view.astype(np.float64), K.astype(np.float64))
    h, _, _ = np.histogram2d(uv[:, 0], uv[:, 1], bins=32,
                             range=[[0, IMAGE_SIZE], [0, IMAGE_SIZE]])
    return (h > 0).ravel()
s0 = silhouette(viewmats[0], Ks[0])
ious = []
for i in range(1, N_VIEWS):
    si = silhouette(viewmats[i], Ks[i])
    ious.append((s0 & si).sum() / max(1, (s0 | si).sum()))
print(f"    silhouette IoU of view 0 vs the other {N_VIEWS - 1} azimuths: "
      f"min={min(ious):.3f} max={max(ious):.3f}")
assert max(ious) < 0.98, "azimuths render (nearly) identical silhouettes"
print("    PASS")

# ----------------------------------------------------------------- C5
print("\n[check C5] token -> gaussian decode: activations and clamps")
means, quats, scales, opacities, colors = tokens_to_gaussians(tok)
print(f"    means   {tuple(means.shape)}  range [{means.min():.3f}, {means.max():.3f}]")
print(f"    scales  {tuple(scales.shape)}  range [{scales.min():.2e}, {scales.max():.4f}]")
print(f"    opacity {tuple(opacities.shape)} range [{opacities.min():.4f}, {opacities.max():.4f}]")
print(f"    colors  {tuple(colors.shape)}  range [{colors.min():.4f}, {colors.max():.4f}]")
assert (scales > 0).all() and (scales <= 1.0).all()
assert (opacities >= 0).all() and (opacities < 1.0).all()
assert (colors >= 0).all() and (colors <= 1.0).all()
assert quats.shape[1] == 4
# w-first and canonicalised: read_splat + canonicalize_quat guarantee w >= 0,
# and gsplat's "wxyz convension" means no reordering is needed on the way in.
assert (quats[:, 0] >= 0).all(), "quaternion is not w-first/canonicalised as gsplat expects"
# A scale clamp that fires on real ground-truth data would mean the ceiling is
# wrong, not that the data is bad.
n_clamped_hi = int((np.exp(tok[:, 3:6]) > 1.0).sum())
print(f"    ground-truth Gaussians hitting the scale ceiling: {n_clamped_hi} of "
      f"{tok.shape[0] * 3} components")
assert n_clamped_hi == 0, "the 1.0 scale ceiling clips real data — it is too tight"
print("    PASS")

# ----------------------------------------------------------------- C6
print("\n[check C6] render_views runs and produces images in [0,1]")
imgs, backend = render_views(tok, viewmats, Ks, image_size=IMAGE_SIZE)
print(f"    backend={backend}  images {imgs.shape}  range [{imgs.min():.3f}, {imgs.max():.3f}]")
assert imgs.shape == (len(viewmats), IMAGE_SIZE, IMAGE_SIZE, 3)
assert np.isfinite(imgs).all() and imgs.min() >= 0 and imgs.max() <= 1
if backend != "gsplat":
    print("    NOTE: gsplat is CUDA-only and this box has no GPU, so this is the")
    print("    point-renderer fallback. eval/metrics.py records the backend name")
    print("    with its numbers so a CPU run is never compared against a GPU one.")
    print(f"    gsplat_available()={gsplat_available()}")
print("    PASS")

# ----------------------------------------------------------------- figure
fig, axes = plt.subplots(2, 4, figsize=(14, 7.4))
for ax, (view, K) in zip(axes.ravel(), zip(viewmats, Ks)):
    uv, z = project(xyz, view.astype(np.float64), K.astype(np.float64))
    order = np.argsort(-z)
    ax.scatter(uv[order, 0], uv[order, 1], s=0.6,
               c=np.clip(tok[order, 11:14], 0, 1), linewidths=0)
    ax.set_xlim(0, IMAGE_SIZE)
    ax.set_ylim(IMAGE_SIZE, 0)     # image convention: row 0 at the top
    ax.set_aspect("equal")
    ax.axis("off")
plt.suptitle(f"eval/render3d.py camera rig — hand-computed projection of {name[:28]}\n"
             f"(chairs must be upright and centred in every panel)", fontsize=11)
plt.tight_layout()
out = "reports/check_render3d_rig.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nwrote {out} — the visual confirmation that C3 is really upright")

print("\nALL RENDER3D CHECKS PASSED.")
