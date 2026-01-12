"""Public surface for the :mod:`ogn_sdk` package.

External callers should import :class:`RunRequest` and :func:`run_local` from
here rather than reaching into internal modules. This module is treated as the
stable Python API over the OGN engine; other modules under :mod:`ogn_sdk` are
implementation details.
"""

from .client import RunRequest, run_local

# Optional gRPC client (stubs generated during release)
try:
  from .grpc_client import OGNClient  # type: ignore
except Exception:  # pragma: no cover
  OGNClient = None  # type: ignore

__all__ = ["RunRequest", "run_local", "OGNClient"]
__version__ = "0.9.1"
