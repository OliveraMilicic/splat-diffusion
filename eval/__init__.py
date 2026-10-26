"""
eval/ — decode/round-trip checks, baselines, metrics (STAGE_SPEC §8).

Same repo convention as `data/__init__.py`: flat-root modules
(preprocessing.py, splat_utils.py, stage_config.py) live at repo root, not
under a `src/` prefix, so put repo root on sys.path here once, so every
submodule can `import preprocessing` etc. regardless of where the entry
point lives. `eval/decode.py` also does this at its own top, redundantly
but harmlessly, so `python eval/decode.py` (direct script execution, which
does NOT trigger this file) still works — see its docstring.
"""

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
