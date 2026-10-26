"""
container_env.py — make this code importable inside the GPU Jupyter
container, where the uid has no `/etc/passwd` entry.

Symptom this fixes (seen in the GPU container, shell prompt shows
`I have no name!`):

    KeyError: 'getpwuid(): uid not found: 1565'

raised from `getpass.getuser()` inside `torch._inductor`'s cache-dir
setup, reached at `import torch._dynamo` time — so it fires on plain
`import torch` (via `diffusers`), long before any of our code runs.
Nothing to do with the pipeline; purely a container/uid-mapping artifact.

Two independent fixes, both applied (either alone is sufficient, but
they cover different future call sites):

  1. `getpass.getuser()` consults `$LOGNAME`/`$USER`/`$LNAME`/`$USERNAME`
     *before* falling back to the password database, so exporting one is
     enough to stop the crash.
  2. `$TORCHINDUCTOR_CACHE_DIR` / `$TRITON_CACHE_DIR` set explicitly, so
     torch skips the username-derived default path entirely. Pointed at
     `$HOME` (NFS, persists) rather than `/tmp` — CLAUDE.md: `/tmp` is
     container-local and gone on the next rebuild.

**Import this before `torch`.** Applied on import (side effect is the
entire point), so entry scripts just need `import container_env` at the
top, above any torch import.
"""

import getpass
import os
from pathlib import Path


def _resolve_username() -> str:
    for key in ("LOGNAME", "USER", "LNAME", "USERNAME"):
        value = os.environ.get(key)
        if value:
            return value
    try:
        import pwd
        return pwd.getpwuid(os.getuid()).pw_name
    except (ImportError, KeyError):
        # the container case: uid has no passwd entry. Any stable,
        # filesystem-safe string works — it only ever names a cache dir.
        return f"uid{os.getuid()}"


def apply() -> dict:
    """Idempotent. Returns what it set, for logging."""
    username = _resolve_username()
    for key in ("LOGNAME", "USER", "LNAME", "USERNAME"):
        os.environ.setdefault(key, username)

    home = Path(os.environ.get("HOME") or Path.home())
    cache_root = home / ".cache"
    for var, sub in (("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
                      ("TRITON_CACHE_DIR", "triton")):
        if not os.environ.get(var):
            path = cache_root / sub
            try:
                path.mkdir(parents=True, exist_ok=True)
                os.environ[var] = str(path)
            except OSError:
                pass  # unwritable HOME: leave it to torch's own default

    # verify the actual failing call now works, so a future torch version
    # reaching it by a different path still fails loudly here, not 200
    # lines into someone's training run
    getpass.getuser()

    return dict(username=username,
                torchinductor_cache_dir=os.environ.get("TORCHINDUCTOR_CACHE_DIR"))


_APPLIED = apply()
