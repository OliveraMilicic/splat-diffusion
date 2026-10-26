"""
data/ — shard reading, augmentation, and the torch Dataset (TRAINING_SPEC §1).

Repo convention is flat-root modules (preprocessing.py, splat_utils.py,
ptv3.py, ...), not a `src/` package — this deviates from TRAINING_SPEC §1's
literal `src/data/...` paths only by dropping the `src/` prefix, to match
what's already here. `data/augment.py` and `data/dataset.py` both need
`preprocessing.py` (repo root). That import only resolves automatically
when the entry script itself lives at repo root; once `scripts/train.py`
exists (TRAINING_SPEC phase D+) that won't hold, so put repo root on
sys.path here, once, so every submodule of this package can rely on
`import preprocessing` working regardless of where the entry point lives.
"""

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
