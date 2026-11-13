"""Kaggle T4 x2 offload for labelImg Plus heavy annotation tasks.

The local annotation helper stays usable on CPU.  When the user opts in,
YOLO prediction and SAM 2 tracking are packaged into a self-contained job,
run on a Kaggle GPU kernel, and the resulting YOLO labels are merged back into
the local ``labels/`` folder with a normal Undo snapshot.

This package intentionally has no mandatory third-party dependencies at import
time.  The Kaggle CLI itself is installed into a separate Python 3.11+ venv
(``.kaggle-venv``), because the labelImg Plus helper environment may still be
Python 3.10.
"""

__version__ = "1.0.0"

from .config import Config, ConfigError, load_config  # noqa: F401
from .spec import JobSpec  # noqa: F401

__all__ = ["Config", "ConfigError", "JobSpec", "load_config", "__version__"]
