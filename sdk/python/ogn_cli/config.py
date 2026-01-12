from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OGNPaths:
  data_root: Path
  runs_root: Path
  profiles_root: Path


def default_data_root() -> Path:
  env = os.getenv("OGN_DATA_ROOT")
  if env:
    return Path(env).expanduser()
  return Path("/data")


def default_runs_root() -> Path:
  env = os.getenv("OGN_RUNS_ROOT")
  if env:
    return Path(env).expanduser()
  # Default to a local runs/ directory under the current working tree.
  return Path.cwd() / "runs"


def default_profiles_root() -> Path:
  env = os.getenv("OGN_PROFILES_ROOT")
  if env:
    return Path(env).expanduser()
  return Path.cwd() / "profiles"


def resolve_paths(
    *, data_root: str | None = None, runs_root: str | None = None, profiles_root: str | None = None
) -> OGNPaths:
  return OGNPaths(
      data_root=Path(data_root).expanduser() if data_root else default_data_root(),
      runs_root=Path(runs_root).expanduser() if runs_root else default_runs_root(),
      profiles_root=Path(profiles_root).expanduser()
      if profiles_root
      else default_profiles_root(),
  )
