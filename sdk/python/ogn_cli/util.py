from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class GPUInfo:
  present: bool
  name: Optional[str] = None
  count: int = 0


def detect_gpu() -> GPUInfo:
  """Best-effort CUDA GPU detection via nvidia-smi."""

  nvidia_smi = shutil.which("nvidia-smi")
  if not nvidia_smi:
    return GPUInfo(present=False, name=None, count=0)

  try:
    out = subprocess.check_output(
        [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=5,
    )
  except Exception:
    return GPUInfo(present=False, name=None, count=0)

  lines = [line.strip() for line in out.splitlines() if line.strip()]
  if not lines:
    return GPUInfo(present=False, name=None, count=0)
  return GPUInfo(present=True, name=lines[0], count=len(lines))


def find_binary(candidates: List[str]) -> Optional[Path]:
  """Locate a pipeline binary quickly, avoiding expensive full tree scans by default.

  Search order:
  1) $PATH (fast)
  2) Common install dirs (/opt/ogn/bin, /opt/ogn/install/bin)
  3) Optional build tree scan under ./build (only if OGN_FIND_BINARY_SCAN_BUILD=1)
  """

  # 1) PATH
  for name in candidates:
    resolved = shutil.which(name)
    if resolved:
      return Path(resolved)

  # 2) Well-known install prefixes
  hint_dirs = [
      Path("/opt/ogn/bin"),
      Path("/opt/ogn/install/bin"),
      Path.cwd() / "install" / "bin",
  ]
  # Allow callers to supply extra hint dirs via env.
  extra = os.environ.get("OGN_BINARY_HINTS")
  if extra:
    for part in extra.split(":"):
      if part.strip():
        hint_dirs.append(Path(part.strip()))

  for hint in hint_dirs:
    for name in candidates:
      candidate = hint / name
      if candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate

  # 3) Optional build tree scan (can be very slow on large trees)
  scan_build = os.environ.get("OGN_FIND_BINARY_SCAN_BUILD", "0").lower() in (
      "1",
      "true",
      "yes",
  )
  if scan_build:
    repo_root = Path.cwd()
    build_dir = repo_root / "build"
    if build_dir.is_dir():
      for path in build_dir.rglob("*"):
        if path.is_file() and path.name in candidates and os.access(path, os.X_OK):
          return path

  return None


def run_binary_version(path: Path) -> Optional[str]:
  try:
    out = subprocess.check_output(
        [str(path), "--version"],
        stderr=subprocess.STDOUT,
        text=True,
        timeout=5,
    ).strip()
  except Exception:
    return None
  return out or None


def print_json(data: Dict[str, Any]) -> None:
  json.dump(data, fp=os.sys.stdout, indent=2)
  os.sys.stdout.write("\n")
