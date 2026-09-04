from __future__ import annotations

import pathlib
import sys

try:
    import deepgate
except ImportError:
    deepgate_root = pathlib.Path(__file__).resolve().parents[1] / "third-party" / "python-deepgate"
    sys.path.insert(0, str(deepgate_root))
    import deepgate

from deepgate import AigParser  # noqa: F401
from deepgate import Model  # noqa: F401
