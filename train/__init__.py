"""
train/ — dataset, training loop, overfit gate (STAGE_SPEC §8).

Same repo convention as `data/__init__.py`/`eval/__init__.py`: put repo
root on sys.path here once, so submodules can `import preprocessing`,
`import stage_config`, etc. regardless of where the entry point lives.
Entry-point scripts in this package (`overfit.py`) also do this at their
own top, redundantly but harmlessly, since direct script execution
(`python train/overfit.py`) never triggers this file — see their
docstrings.
"""

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
