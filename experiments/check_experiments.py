"""
experiments/check_experiments.py — everything about the three experiments
that can be verified WITHOUT a GPU.

Per CLAUDE.md, "every step needs a check it can run itself". The expensive
part of these experiments is three training runs; the cheap part is being
sure that when the GPU frees up, the commands will do what they claim. This
checks the second part, so a typo costs seconds instead of an hour of A100
time and a figure that quietly measures the wrong thing.

    python experiments/check_experiments.py

Twelve checks, each printing PASS/FAIL and what it means. Exit code is
non-zero if any fail.
"""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from stage_config import SHARD_PATH, StageConfig, load_split

EXP_DIR = _REPO_ROOT / "experiments"
RUNS = _REPO_ROOT / "runs"

_results = []


def check(name: str, ok: bool, detail: str = ""):
    _results.append((name, ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if detail else ""))
    return ok


def load_names(path: Path) -> dict:
    return json.loads(path.read_text())


def main():
    print(__doc__.split("\n\n")[0].strip())
    print("\n--- subset files -------------------------------------------------")
    split = load_split(SHARD_PATH)
    train_set = set(split["train"])
    test_set = set(split["val"])

    colourful_p = EXP_DIR / "names_colourful.json"
    brown_p = EXP_DIR / "names_brown.json"
    ok = check("C1  both subset files exist",
               colourful_p.exists() and brown_p.exists(),
               "run: python experiments/select_chairs.py")
    if not ok:
        return finish()

    colourful, brown = load_names(colourful_p), load_names(brown_p)
    cn, bn = colourful["names"], brown["names"]

    check("C2  every selected object is in the TRAIN split",
          set(cn) <= train_set and set(bn) <= train_set,
          "an experiment that trains on a held-out object spends the only "
          "clean evidence experiment B has")
    check("C3  no selected object is in the TEST split",
          not (set(cn) & test_set) and not (set(bn) & test_set))
    check("C4  the two subsets do not overlap",
          not (set(cn) & set(bn)),
          f"shared: {sorted(set(cn) & set(bn))}")
    check("C5  subset sizes are multiples of batch 4",
          len(cn) % 4 == 0 and len(bn) % 4 == 0,
          f"colourful={len(cn)}, brown={len(bn)} — a non-multiple means "
          f"drop_last sits some objects out of each epoch")

    print("\n--- are the selections what they claim? --------------------------")
    cs = colourful["stats"]
    bs = brown["stats"]
    brown_fracs = np.array([bs[n]["frac_brown"] for n in bn])
    check("C6  the brown subset is actually brown",
          brown_fracs.min() > 0.7,
          f"min frac_brown {brown_fracs.min():.3f}, mean {brown_fracs.mean():.3f} "
          f"— experiment B's entire claim rests on this number")
    brown_hues = np.array([bs[n]["hue_mean"] for n in bn])
    check("C7  the brown subset is one COLOUR, not a mixture averaging to brown",
          brown_hues.std() < 15.0,
          f"hue {brown_hues.mean():.1f}° ± {brown_hues.std():.1f}°")

    chromas = np.array([cs[n]["chroma"] for n in cn])
    check("C8  the colourful subset is actually colourful",
          chromas.min() > 0.15,
          f"min chroma {chromas.min():.3f}, max {chromas.max():.3f}")
    hues = np.sort([cs[n]["hue_mean"] for n in cn])
    gaps = np.diff(np.concatenate([hues, [hues[0] + 360.0]]))
    check("C9  the colourful subset spans the hue circle",
          gaps.max() < 140.0,
          f"largest empty arc {gaps.max():.0f}° — a set clustered in one hue "
          f"could be reproduced by a model that learned one colour")
    c_brown = np.array([cs[n]["frac_brown"] for n in cn])
    check("C10 no brown chairs in the colourful subset",
          c_brown.max() < 0.5,
          f"max frac_brown {c_brown.max():.3f} — brown is the dataset's default "
          f"colour, so a brown object is not evidence of colour reproduction")

    gate_p = EXP_DIR / "names_gate4.json"
    if check("C10b the day-2 gate subset exists", gate_p.exists()):
        gn = load_names(gate_p)["names"]
        # runs/overfit_4obj_32k took split["train"][:4] (train/overfit.py
        # slices the split directly). Pinning that here means a future change
        # to the split file surfaces as a failed check rather than as
        # experiment A quietly training on different chairs than the figure
        # it is meant to reproduce.
        check("C10c it is the 4 objects runs/overfit_4obj_32k trained on",
              len(gn) == 4 and set(gn) <= train_set
              and gn == split["train"][:4],
              f"{len(gn)} names, = split['train'][:4]. 2 of the 4 are "
              f"achromatic (chroma 0.001), so this set is a MEMORISATION "
              f"demo, not a colour demo")

    print("\n--- the ablation's control ---------------------------------------")
    l2_subset = RUNS / "stage1_100obj" / "train_subset.json"
    if check("C11 the L2 run's object list exists",
             l2_subset.exists(),
             f"experiment C trains its second arm from {l2_subset}"):
        l2_names = load_names(l2_subset)["names"]
        check("C12 that list is a clean 100-object subset of TRAIN",
              len(l2_names) == 100 and set(l2_names) <= train_set,
              f"{len(l2_names)} names")

    print("\n--- the checkpoint loader ----------------------------------------")
    # The one piece of plumbing that is easy to get wrong and impossible to
    # notice: a run older than a StageConfig default rebuilt with today's
    # defaults. Verified against the real checkpoint, on CPU.
    try:
        import container_env  # noqa: F401
        import torch
        from eval.load_run import load_run
        model, cfg, meta = load_run("stage1_100obj", torch.device("cpu"))
        n_out = cfg.n_model_out_ch
        check("C13 stage1_100obj loads with the config its WEIGHTS imply",
              cfg.stage == 1 and cfg.learn_sigma is False and n_out == 3,
              f"stage {cfg.stage}, learn_sigma {cfg.learn_sigma}, out_ch {n_out} "
              f"— config.json alone would have built a 6-channel head")
    except Exception as e:
        check("C13 stage1_100obj loads with the config its WEIGHTS imply",
              False, f"{type(e).__name__}: {e}")

    print("\n--- the launch commands ------------------------------------------")
    sh = (EXP_DIR / "run_all.sh").read_text()
    check("C14 experiment C points at the L2 run's own object list",
          "--train-names-file runs/stage1_100obj/train_subset.json" in sh,
          "a fresh --n-train-objects 100 would draw a DIFFERENT 100 objects")
    check("C15 experiment C matches the L2 run's 500-step warmup",
          "--warmup-steps 500" in sh,
          "the current default is 1000; an unmatched schedule is a second "
          "variable in a one-variable experiment")
    check("C16 both colour experiments are stage 1 (colour only)",
          sh.count("--stage 1") >= 3)

    # Every flag the script passes to train/loop.py must actually exist.
    # argparse rejects an unknown flag, so a typo here costs the whole
    # launch — and it would land AFTER the previous run finished, i.e. at
    # exactly the moment nobody is watching.
    import re
    import subprocess
    helptext = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "train" / "loop.py"), "--help"],
        capture_output=True, text=True, cwd=_REPO_ROOT).stdout
    # `[a-z0-9_-]` and not `[a-z0-9-]`: a flag mistyped with an underscore
    # (`--warmup_steps`) is THE typo this check exists to catch, and a
    # pattern that stops at the underscore reads it as `--warmup`.
    FLAG = r"(?<![\w-])--[a-z][a-z0-9_-]+"
    # Scan ONLY the lines that actually invoke train/loop.py. The first
    # version scanned the whole file minus a hand-maintained exclusion list,
    # which flagged `--views` (a reconstruction_grid.py flag) the moment one
    # was added — a check that cries wolf gets switched off, so it has to be
    # precise rather than broad.
    #
    # Two preprocessing steps, both load-bearing:
    #   comments   run_all.sh documents its flags in prose (`# --render-loss-*
    #              the L_image ramp...`), which is not a command.
    #   line joins the invocations are backslash-continued across ~8 lines, so
    #              without joining, only the first line of each is seen.
    no_comments = [ln for ln in sh.splitlines() if not ln.lstrip().startswith("#")]
    joined, buf = [], ""
    for ln in no_comments:
        buf += ln.rstrip("\\") if ln.rstrip().endswith("\\") else ln
        if ln.rstrip().endswith("\\"):
            continue
        joined.append(buf)
        buf = ""
    if buf:
        joined.append(buf)
    loop_cmds = [ln for ln in joined if "train/loop.py" in ln]
    used = set(re.findall(FLAG, " ".join(loop_cmds)))
    # EXACT membership, not `in helptext`: `--warmup` is a substring of
    # "--warmup-steps" in the help output and would pass by accident.
    known = set(re.findall(FLAG, helptext))
    missing = sorted(f for f in used if f not in known)
    check("C17 every train/loop.py flag used by run_all.sh exists",
          not missing and bool(helptext) and len(loop_cmds) >= 3,
          f"unknown: {missing}" if missing else
          f"{len(used)} flags across {len(loop_cmds)} loop.py invocations, "
          f"checked against loop.py --help")

    return finish()


def finish() -> int:
    n_fail = sum(1 for _n, ok in _results if not ok)
    print(f"\n{'=' * 68}")
    if n_fail:
        print(f"{n_fail} of {len(_results)} checks FAILED — fix before spending "
              f"GPU time")
    else:
        print(f"all {len(_results)} checks passed — the experiments are ready "
              f"to launch when the GPU frees up")
    print("=" * 68)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
