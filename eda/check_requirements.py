#!/usr/bin/env python
"""
Check everything the chair-splat notebooks need, in one pass.

    python check_requirements.py           # from a terminal
    %run check_requirements.py             # from inside a notebook cell

Exits 0 if you are ready to run the notebooks, 1 otherwise. Anything missing
comes with the exact command to fix it.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
import traceback
from pathlib import Path

# Detect this FIRST: the optional-package probe below imports ipykernel, which
# would otherwise make any plain terminal run look like a notebook.
try:
    IN_NOTEBOOK = get_ipython().__class__.__name__ in (
        "ZMQInteractiveShell", "TerminalInteractiveShell")   # noqa: F821
except NameError:
    IN_NOTEBOOK = False

# ---------------------------------------------------------------- formatting
TTY = sys.stdout.isatty()
G, Y, R, B, D, X = (("\033[32m", "\033[33m", "\033[31m", "\033[1;36m",
                     "\033[2m", "\033[0m") if TTY else ("",) * 6)

fails: list[str] = []
warns: list[str] = []
pip_needed: list[str] = []


def head(t):
    print(f"\n{B}{t}{X}")


def ok(msg, detail=""):
    print(f"  {G}PASS{X}  {msg}" + (f"  {D}{detail}{X}" if detail else ""))


def bad(msg, fix=""):
    print(f"  {R}FAIL{X}  {msg}")
    if fix:
        print(f"        {D}-> {fix}{X}")
    fails.append(msg)


def warn(msg, fix=""):
    print(f"  {Y}WARN{X}  {msg}")
    if fix:
        print(f"        {D}-> {fix}{X}")
    warns.append(msg)


def _ver(module, dist):
    v = getattr(module, "__version__", None)
    if v:
        return v
    try:
        from importlib.metadata import version
        return version(dist)
    except Exception:
        return "installed"


print(f"{B}Requirements check — chair splat notebooks{X}")
print(f"{D}{time.strftime('%Y-%m-%d %H:%M')}   host {os.uname().nodename}{X}")

# ---------------------------------------------------------------- 1. python
head("1. Python interpreter")

v = sys.version_info
where = sys.executable
if v[:2] == (3, 13):
    bad(f"Python {v.major}.{v.minor}.{v.micro} — 3.13 is too new",
        "several 3D packages have no 3.13 wheels; use the 3.11 env: "
        "conda activate ~/envs/psiml")
elif (3, 9) <= v[:2] <= (3, 12):
    ok(f"Python {v.major}.{v.minor}.{v.micro}", where)
else:
    bad(f"Python {v.major}.{v.minor} is outside the supported 3.9–3.12 range",
        "conda activate ~/envs/psiml")

in_env = "envs/psiml" in where or os.environ.get("CONDA_PREFIX", "").endswith("psiml")
if in_env:
    ok("running inside the psiml environment")
elif "/opt/conda" in where:
    warn("running the container's base conda, not your project env",
         "conda activate ~/envs/psiml   (packages installed here vanish on rebuild)")
else:
    warn(f"interpreter is {where}", "expected something under ~/envs/psiml")

# ---------------------------------------------------------------- 2. packages
head("2. Packages")

REQUIRED = [
    ("numpy", "numpy", "array maths — everything"),
    ("plyfile", "plyfile", "reading the .ply splat files"),
    ("matplotlib", "matplotlib", "all plots and image display"),
    ("pandas", "pandas", "the stats table in notebook 02"),
    ("scipy", "scipy", "KD-tree and distribution stats"),
    ("pyarrow", "pyarrow", "caching stats to parquet"),
]
OPTIONAL = [
    ("plotly", "plotly", "one interactive 3D scatter cell"),
    ("torch", "torch", "not needed for EDA; needed to train"),
    ("gsplat", "gsplat", "proper CUDA rendering; splatlib works without it"),
    ("ipykernel", "ipykernel", "running these as notebooks"),
]

for mod, pipname, why in REQUIRED:
    try:
        m = __import__(mod)
        ok(f"{mod:<12} {_ver(m, pipname)}", why)
    except Exception as e:
        bad(f"{mod:<12} missing  ({why})")
        pip_needed.append(pipname)

for mod, pipname, why in OPTIONAL:
    try:
        m = __import__(mod)
        ok(f"{mod:<12} {_ver(m, pipname)}", why + "  [optional]")
    except Exception:
        warn(f"{mod:<12} missing  ({why})", f"python -m pip install {pipname}")

# numpy 2 removed ndarray.ptp; splatlib uses np.ptp, but flag old numpy anyway
try:
    import numpy as np
    if int(np.__version__.split(".")[0]) < 1:
        warn(f"numpy {np.__version__} is very old")
except Exception:
    pass

# ---------------------------------------------------------------- 3. splatlib
head("3. splatlib.py")

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE))
sl = None
try:
    import splatlib as sl
    need = ["load_ply", "describe_ply", "find_objects", "render",
            "object_stats", "voxel_occupancy", "canonicalize_quat",
            "covariance_3d", "gaussian_sample", "save_norm_stats"]
    missing = [n for n in need if not hasattr(sl, n)]
    if missing:
        bad(f"splatlib is missing: {missing}", "re-download splatlib.py")
    else:
        ok(f"splatlib imported, all {len(need)} entry points present", str(HERE))
except Exception as e:
    bad(f"cannot import splatlib: {e}",
        f"put splatlib.py next to the notebooks (looked in {HERE})")

for f in ["01_load_and_render.ipynb", "02_eda.ipynb", "preprocess.py"]:
    (ok if (HERE / f).exists() else warn)(
        f"{f} {'found' if (HERE / f).exists() else 'not in this directory'}")

# ---------------------------------------------------------------- 4. data
head("4. Dataset")

CANDIDATES = [
    "/lustre/data/mls04/ModelNet_Splats/extracted/train",
    "/data/ModelNet_Splats/extracted/train",
    (os.environ["DATASET_DIR"] + "/extracted/train")
    if os.environ.get("DATASET_DIR") else "",
]
CANDIDATES = [c for c in CANDIDATES if c]
root = next((Path(c) for c in CANDIDATES if c and Path(c).is_dir()), None)

files = []
if root is None:
    bad("no data root found", "checked:\n           " + "\n           ".join(
        c for c in CANDIDATES if c))
else:
    ok(f"data root {root}")
    t0 = time.time()
    all_ply = list(root.rglob("*.ply"))
    scan = time.time() - t0
    files = sorted(p for p in all_ply if p.stem.startswith("chair"))
    if not files:
        bad(f"no chair_*.ply under {root}",
            f"found {len(all_ply)} .ply total; check the category name")
    else:
        ok(f"{len(files)} chair files", f"{len(all_ply)} .ply total")
        if len(files) < 889:
            warn(f"expected 889 chairs in the ModelNet40 train split, found {len(files)}",
                 "extraction may be incomplete")
    if scan > 20:
        warn(f"directory scan took {scan:.0f}s — Lustre metadata is slow here",
             "run preprocess.py before training; per-epoch file reads will hurt")

# ---------------------------------------------------------------- 5. real load
head("5. Load one file for real")

sp = None
if files and sl is not None:
    try:
        t0 = time.time()
        sp = sl.load_ply(files[0])
        dt = time.time() - t0
        ok(f"parsed {files[0].name}", f"{len(sp):,} gaussians in {dt:.2f}s")

        import numpy as np
        checks = [
            ("positions finite", np.isfinite(sp.xyz).all()),
            ("scales positive after exp", (sp.scale > 0).all()),
            ("opacity within [0,1]", (sp.opacity >= 0).all() and (sp.opacity <= 1).all()),
            ("quaternions unit norm",
             np.allclose(np.linalg.norm(sp.quat, axis=1), 1, atol=1e-4)),
            ("rgb within [0,1]", (sp.rgb >= 0).all() and (sp.rgb <= 1).all()),
        ]
        for name, passed in checks:
            (ok if passed else bad)(name)

        cov = sl.covariance_3d(sp.scale[:100], sp.quat[:100])
        ev = np.linalg.eigvalsh(cov)
        rec = np.sort(np.sqrt(np.clip(ev, 0, None)), 1)
        if np.allclose(rec, np.sort(sp.scale[:100], 1), rtol=1e-3):
            ok("covariance eigenvalues match scale²")
        else:
            bad("covariance reconstruction is wrong",
                "quaternion convention or scale activation is off")

        n_props = 3 + 3 + 3 + sp.f_rest.shape[1] + 1 + 3 + 4
        print(f"        {D}SH degree {sp.sh_degree}, {sp.f_rest.shape[1]} f_rest "
              f"coefficients (~{n_props} properties per vertex){X}")
        if sp.f_rest.shape[1] not in (0, 9, 24, 45):
            warn(f"unusual f_rest count ({sp.f_rest.shape[1]})",
                 "run describe_ply() in notebook 01 cell 5 and check the layout")
    except Exception as e:
        bad(f"load_ply failed: {e}")
        traceback.print_exc()
else:
    warn("skipped — need both splatlib and at least one chair file")

# ---------------------------------------------------------------- 6. render
head("6. Render end to end")

if sp is not None:
    try:
        import numpy as np
        t0 = time.time()
        img = sl.render(sp, azim=35, elev=18, H=128, W=128)
        dt = time.time() - t0
        coverage = float((img.mean(-1) < 0.98).mean())
        if img.shape == (128, 128, 3) and np.isfinite(img).all():
            ok(f"rendered 128×128 in {dt:.2f}s", f"coverage {coverage:.1%}")
            est = dt * (384 / 128) ** 2
            print(f"        {D}a 384×384 view will take roughly {est:.1f}s{X}")
        else:
            bad("render produced a malformed image")
        if coverage < 0.02:
            warn("image is nearly empty",
                 "try a different up_axis, or check the quaternion convention")
    except Exception as e:
        bad(f"render failed: {e}")
        traceback.print_exc()
else:
    warn("skipped — nothing loaded")

# ---------------------------------------------------------------- 7. output
head("7. Output location and disk")

OUT = Path.home() / "psiml" / "outputs"
try:
    OUT.mkdir(parents=True, exist_ok=True)
    probe = OUT / f".write_test.{os.getpid()}"
    probe.touch()
    probe.unlink()
    free = shutil.disk_usage(OUT).free / 1e9
    ok(f"{OUT} writable", f"{free:.0f} GB free")
    if free < 5:
        warn(f"only {free:.1f} GB free where outputs go")
except Exception as e:
    bad(f"cannot write to {OUT}: {e}")

if root is not None:
    est_gb = len(files) * 32768 * 14 * 4 / 1e9 if files else 0
    print(f"        {D}preprocess.py at N=32768 would write ~{est_gb:.1f} GB{X}")

# ---------------------------------------------------------------- 8. kernel
head("8. JupyterLab kernel")

kern = Path.home() / ".local/share/jupyter/kernels/psiml/kernel.json"
if kern.exists():
    txt = kern.read_text()
    if "envs/psiml" in txt:
        ok("kernel 'psiml' registered and points at ~/envs/psiml")
    else:
        warn("kernel 'psiml' exists but points somewhere else",
             "conda activate ~/envs/psiml && python -m ipykernel install --user "
             "--name psiml --display-name 'PSI:ML py3.11'")
else:
    warn("kernel 'psiml' not registered",
         "conda activate ~/envs/psiml && python -m pip install ipykernel && "
         "python -m ipykernel install --user --name psiml "
         "--display-name 'PSI:ML py3.11'")

# ---------------------------------------------------------------- summary
print()
print("=" * 68)
if pip_needed:
    print(f"{Y}Install the missing packages:{X}")
    print(f"  python -m pip install {' '.join(sorted(set(pip_needed)))}")
    print()

if not fails:
    print(f"{G}Ready.{X} "
          + (f"{len(warns)} warning(s) — none blocking." if warns else "No warnings."))
    print("Open 01_load_and_render.ipynb with the 'PSI:ML py3.11' kernel.")
else:
    print(f"{R}{len(fails)} blocking problem(s):{X}")
    for f in fails:
        print(f"  - {f}")

if not IN_NOTEBOOK:
    sys.exit(1 if fails else 0)
