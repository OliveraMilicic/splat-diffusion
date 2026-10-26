"""
data/augment.py — DESIGN_SPEC §3.6 augmentation: mirror, rotation about the
up axis, global scale jitter, position jitter. Operates on a single
object's RAW (unstandardized, per-object-normalized) (N, 14) token array —
the form stored in tokens.dat, before `preprocessing.standardize()`.
Geometric ops (mirror/rotate) must happen here, not on standardized
values: quaternion channels are exempt from standardization but position
and log-scale are not, and composing a rotation matrix or negating an axis
is only exact in the raw, physically-meaningful units.

Mirror/rotation quaternion transforms are the part DESIGN_SPEC §3.6 flags
as "easy to get wrong" and asks for a unit test on. `check_augmentation.py`
is that test — it checks the covariance Σ = R diag(s²) Rᵀ built from the
augmented (scale, quat) against the directly-transformed Σ from the
original, the same idiom CLAUDE.md already prescribes for the loader's
quaternion convention (eigenvalues == scale²). DESIGN_SPEC's own suggested
check (render augmented vs. a correspondingly-moved camera) can't actually
catch a quaternion bug: `splat_utils.render_points` draws isotropic dots
sized by mean scale, it doesn't render orientation at all — so the
covariance check is the real test; `check_augmentation.py` also does the
render as a cheap visual sanity check on top, not as the proof.
"""

import numpy as np

from preprocessing import N_CHANNELS, canonicalize_quat

POS = slice(0, 3)
LOG_SCALE = slice(3, 6)
QUAT = slice(6, 10)


# --------------------------------------------------------------- quaternion ops
def quat_multiply(q1, q2):
    """Hamilton product q1 ⊗ q2, both (..., 4) w-first. Represents
    "apply the rotation of q2, then the rotation of q1" (world-frame
    composition, q1 on the left)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1)


def axis_angle_quat(axis, theta):
    """Rotation quaternion for angle `theta` about coordinate axis `axis`
    (0/1/2), w-first."""
    half = theta / 2.0
    w = np.cos(half)
    v = np.zeros(3, dtype=np.float64)
    v[axis] = np.sin(half)
    return np.array([w, v[0], v[1], v[2]], dtype=np.float32)


def rotation_matrix_about_axis(axis, theta):
    c, s = np.cos(theta), np.sin(theta)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    if axis == 2:
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    raise ValueError(f"axis must be 0, 1, or 2, got {axis}")


# --------------------------------------------------------------- individual augmentations
def mirror(tok, axis):
    """Reflect positions across the plane normal to `axis`; transform the
    quaternion to match. Derivation: reflecting the whole scene by
    M = diag(-1 at `axis`, +1 elsewhere) sends a Gaussian's covariance
    Σ = R S² Rᵀ to MΣM = (MRM) S² (MRM)ᵀ (using M²=I and M S² M = S² for
    any diagonal S² since M is a sign matrix) — so the reflected
    orientation is R' = M R M, same scale. In quaternion terms this is:
    keep w and the component matching `axis`, negate the other two vector
    components.
    """
    out = tok.copy()
    out[:, POS.start + axis] *= -1.0
    for ax in range(3):
        if ax != axis:
            out[:, QUAT.start + 1 + ax] *= -1.0
    out[:, QUAT] = canonicalize_quat(out[:, QUAT])
    return out


def rotate_about_axis(tok, axis, theta):
    """Rotate positions and compose the quaternion by `theta` about
    coordinate axis `axis` (DESIGN_SPEC §3.6: rotation about the up
    axis, uniform in [0, 2*pi))."""
    out = tok.copy()
    R = rotation_matrix_about_axis(axis, theta)
    out[:, POS] = out[:, POS] @ R.T
    q_rot = axis_angle_quat(axis, theta)
    out[:, QUAT] = quat_multiply(q_rot[None, :], out[:, QUAT])
    out[:, QUAT] = canonicalize_quat(out[:, QUAT])
    return out


def scale_jitter(tok, lo, hi, rng):
    """Global scale jitter: positions *= s, log_scale += log(s)."""
    s = rng.uniform(lo, hi)
    out = tok.copy()
    out[:, POS] *= s
    out[:, LOG_SCALE] += np.log(s).astype(np.float32)
    return out


def position_jitter(tok, std, rng):
    out = tok.copy()
    out[:, POS] += rng.normal(0.0, std, size=out[:, POS].shape).astype(np.float32)
    return out


# --------------------------------------------------------------- composed
def augment_object(tok, mirror_axis, up_axis, rng, cfg):
    """Apply DESIGN_SPEC §3.6's augmentation stack, in order: mirror (with
    probability `cfg.mirror_prob`), rotation about `up_axis` (if
    `cfg.rotation_aug`, angle always freshly sampled — no gating, angle 0
    is already the identity), global scale jitter, position jitter. `tok`
    is one object's raw (N, 14) array; re-rolled every call, per
    TRAINING_SPEC §4.
    """
    assert tok.shape[-1] == N_CHANNELS, f"expected {N_CHANNELS} channels, got {tok.shape[-1]}"
    out = tok

    if rng.random() < cfg.mirror_prob:
        out = mirror(out, mirror_axis)

    if cfg.rotation_aug:
        theta = rng.uniform(0.0, 2 * np.pi)
        out = rotate_about_axis(out, up_axis, theta)

    lo, hi = cfg.scale_jitter_range
    out = scale_jitter(out, lo, hi, rng)

    if cfg.position_jitter_std > 0:
        out = position_jitter(out, cfg.position_jitter_std, rng)

    return out
