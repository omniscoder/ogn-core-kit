"""Top-level CLI package for the `ogn` command.

This package intentionally keeps its surface small and focused on the junior
workflow. For the MVP product slice the blessed commands are:

- ``ogn doctor``
- ``ogn setup demo``
- ``ogn run demo``

Advanced usage remains available via:

- ``ogn setup <bundle>`` (e.g. ``hg002``)
- ``ogn run profile=<name> sample=<name> [...]``
- ``ogn results [...]``

Only :func:`main` is considered a stable public entrypoint; all submodules
(`ogn_cli.run`, `ogn_cli.setup`, etc.) are internal and may change between
releases. The CLI surface is versioned alongside the engine so we can evolve
behaviour without breaking external integrations.
"""

from __future__ import annotations

__all__ = ["main"]

from .main import main
