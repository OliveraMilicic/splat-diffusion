"""model/ — PTv3 backbone, conditioning glue, denoiser (STAGE_SPEC §8).

Same repo convention as `data/`, `eval/`, `train/`: put repo root on
sys.path once so submodules can `import stage_config`, `import
batching`, etc. regardless of where the entry point lives.
"""

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
