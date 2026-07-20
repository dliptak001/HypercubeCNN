"""HypercubeCNN: dependency-free hypercube CNN core.

Phase 0 package surface: version + loadable native extension.
The public ``HCNN`` API is implemented in later phases (see
``docs/python_sdk_plan.md``).
"""

from __future__ import annotations

from ._core import __version__ as _core_version

__version__ = _core_version
__all__ = ["__version__"]
