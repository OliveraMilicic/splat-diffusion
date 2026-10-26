"""
check_augmentation.py — DESIGN_SPEC §3.6's "write a unit test" for the
mirror/rotation quaternion transforms, closing the Day-1 PLAN.md follow-up
("Mirror-augment quaternion-transform test... was never built").

DESIGN_SPEC's own suggested check (render augmented vs. a correspondingly
-moved camera, assert images match) can't actually catch a quaternion sign
bug: `splat_utils.render_points` draws isotropic dots sized by mean scale,
it never reads orientation. The real test, and CLAUDE.md's own idiom for
exactly this class of bug ("build the covariance from (scale, quaternion)
and confirm its eigenvalues equal scale**2"):

    build Sigma = R(q) diag(s^2) R(q)^T from the ORIGINAL (scale, quat)
    apply the geometric transform directly to Sigma:  Sigma' = T Sigma T^T
    build Sigma_hat from the AUGMENTED (scale, quat) data/augment.py produces
    assert Sigma' == Sigma_hat

If the quaternion transform has a sign/axis bug, this fails hard (covariances
disagree by O(1), not by numerical noise) — the render is added underneath
only as a cheap "does this still look like a chair, right way round" gut
check, not as the proof.

Run: python check_augmentation.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.augment import mirror, rotate_about_axis
from data.shards import load_shard_store
from splat_utils import render_points

SHARD_PATH = "/lustre/data/mls04/ShapeSplatsV1/shards"
REPORT_DIR = Path("reports")
N_CHECK_GAUSSIANS = 500
SEED = 0

REPORT_DIR.mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)


def quat_to_rotmat(q):
    """(N, 4) w-first unit quaternion -> (N, 3, 3) rotation matrix.
    Standard Hamilton convention, consistent with data/augment.py's
    quat_multiply (q1 (x) q2 = apply q2 then q1)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.stack([w, x, y, z], axis=-1)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    w, x, y, z = n[:, 0], n[:, 1], n[:, 2], n[:, 3]
    R = np.empty((len(q), 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_covariance(tok):
    """(N, 14) raw token array -> (N, 3, 3) covariance from (log_scale, quat)."""
    scale = np.exp(tok[:, 3:6].astype(np.float64))
    R = quat_to_rotmat(tok[:, 6:10].astype(np.float64))
    S2 = np.einsum("ni,ij->nij", scale ** 2, np.eye(3))
    return R @ S2 @ np.transpose(R, (0, 2, 1))


def mirror_matrix(axis):
    M = np.eye(3)
    M[axis, axis] = -1.0
    return M


def check_transform(tok, transform_fn, expected_R_fn, label):
    """`transform_fn(tok) -> tok'`. `expected_R_fn` gives the (3,3) matrix
    T such that positions should map p -> T @ p and covariance should map
    Sigma -> T @ Sigma @ T^T. Asserts both hold to numerical precision."""
    tok_sub = tok[:N_CHECK_GAUSSIANS]
    sigma_orig = build_covariance(tok_sub)
    pos_orig = tok_sub[:, 0:3].astype(np.float64)

    tok_aug = transform_fn(tok_sub)
    sigma_aug = build_covariance(tok_aug)
    pos_aug = tok_aug[:, 0:3].astype(np.float64)

    T = expected_R_fn()
    sigma_expected = np.einsum("ij,njk,lk->nil", T, sigma_orig, T)
    pos_expected = pos_orig @ T.T

    sigma_err = np.abs(sigma_aug - sigma_expected).max()
    pos_err = np.abs(pos_aug - pos_expected).max()
    quat_norm_err = np.abs(np.linalg.norm(tok_aug[:, 6:10], axis=1) - 1.0).max()
    w_neg_frac = float((tok_aug[:, 6] < 0).mean())

    print(f"[{label}]")
    print(f"  max |Sigma_transform - Sigma_expected| : {sigma_err:.3e}")
    print(f"  max |pos_transform   - pos_expected|    : {pos_err:.3e}")
    print(f"  max |quat norm - 1|                     : {quat_norm_err:.3e}")
    print(f"  fraction w < 0 after canonicalize        : {w_neg_frac:.4f}  (expect 0.0)")
    assert sigma_err < 1e-3, f"{label}: covariance transform mismatch"
    assert pos_err < 1e-4, f"{label}: position transform mismatch"
    assert quat_norm_err < 1e-4, f"{label}: quaternion not unit norm after transform"
    assert w_neg_frac == 0.0, f"{label}: canonicalize_quat left w<0 components"
    print(f"  PASS\n")
    return tok_aug


# ----------------------------------------------------------------- load real data
store = load_shard_store(SHARD_PATH)
name = store.split["train"][0]
tok = store.raw(name, 0)
print(f"checking against real object: {name}  ({len(tok)} Gaussians, "
      f"using first {N_CHECK_GAUSSIANS} for the covariance check)")
print(f"mirror_axis={store.mirror_axis}  up_axis={store.up_axis}\n")

mirror_axis = store.mirror_axis
up_axis = store.up_axis

# ----------------------------------------------------------------- 1. mirror
check_transform(
    tok, lambda t: mirror(t, mirror_axis),
    lambda: mirror_matrix(mirror_axis),
    f"mirror across axis {mirror_axis}",
)

# ----------------------------------------------------------------- 2. rotation
theta = float(rng.uniform(0.3, 2 * np.pi - 0.3))  # avoid near-zero angle (weak test)
from data.augment import rotation_matrix_about_axis
check_transform(
    tok, lambda t: rotate_about_axis(t, up_axis, theta),
    lambda: rotation_matrix_about_axis(up_axis, theta).astype(np.float64),
    f"rotate {np.degrees(theta):.1f} deg about axis {up_axis}",
)

# ----------------------------------------------------------------- 3. visual sanity (not the proof)
from preprocessing import tokens_to_splat_dict

meta = store.object_meta[name]
centre = np.asarray(meta["centre"], np.float32)
radius = meta["radius"]

splats = {
    "original": tokens_to_splat_dict(tok, centre, radius, name=name),
    "mirrored": tokens_to_splat_dict(mirror(tok, mirror_axis), centre, radius, name=name),
    "rotated": tokens_to_splat_dict(rotate_about_axis(tok, up_axis, theta), centre, radius, name=name),
}

fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
for ax, (label, splat) in zip(axes, splats.items()):
    img = render_points(splat, azim=40, elev=15, up_axis=up_axis, size=260)
    ax.imshow(img)
    ax.set_title(label, fontsize=9)
    ax.axis("off")
plt.tight_layout()
out_path = REPORT_DIR / "check_augmentation_render.png"
plt.savefig(out_path, dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out_path} (visual sanity check, not the correctness proof)")

print("\nALL AUGMENTATION CHECKS PASSED.")
