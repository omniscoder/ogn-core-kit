#!/usr/bin/env python3
"""
Lightweight invariant checker for the OGN demo run.

Intended to be called after `ogn run demo` (e.g., via scripts/ogn-demo-smoke.sh).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


def _find_latest_run(runs_root: Path) -> Path:
  if not runs_root.is_dir():
    raise SystemExit(f"Runs root does not exist: {runs_root}")

  candidates: List[Path] = []
  for profile_dir in runs_root.iterdir():
    if not profile_dir.is_dir():
      continue
    for run_dir in profile_dir.iterdir():
      if run_dir.is_dir():
        candidates.append(run_dir)
  if not candidates:
    raise SystemExit(f"No run directories found under {runs_root}")
  return max(candidates)


def _load_json(path: Path) -> Dict[str, Any]:
  if not path.is_file():
    raise SystemExit(f"Expected JSON file does not exist: {path}")
  with path.open("r", encoding="utf-8") as f:
    return json.load(f)


def _check_status(status: Dict[str, Any]) -> None:
  state = status.get("state")
  if state != "succeeded":
    raise SystemExit(f"Run status not succeeded (got {state!r})")

  if "updated_utc" not in status:
    raise SystemExit("status.json missing updated_utc field")


def _check_metrics(metrics: Dict[str, Any], run_dir: Path) -> None:
  wall = metrics.get("wall_time_seconds") or metrics.get("runtime_seconds")
  if wall is None or wall <= 0:
    raise SystemExit(f"Invalid wall_time_seconds/runtime_seconds in metrics: {wall!r}")

  # Sanity bound to catch obviously broken runs; adjust if needed.
  if wall > 4 * 3600:
    raise SystemExit(
        f"Elapsed time {wall} s looks unreasonable for the demo pipeline"
    )

  vcf_path = metrics.get("vcf_path") or metrics.get("result_vcf")
  if not vcf_path:
    raise SystemExit("metrics.json missing vcf_path/result_vcf field")

  vcf_file = (run_dir / vcf_path) if not os.path.isabs(vcf_path) else Path(vcf_path)
  if not vcf_file.is_file():
    raise SystemExit(f"Result VCF does not exist: {vcf_file}")
  if vcf_file.stat().st_size == 0:
    raise SystemExit(f"Result VCF is empty: {vcf_file}")

  accuracy = metrics.get("accuracy") or {}
  snp_f1 = accuracy.get("snp_f1")
  indel_f1 = accuracy.get("indel_f1")
  # Only enforce thresholds if accuracy block is present (truth might be absent
  # in some environments).
  if snp_f1 is not None:
    try:
      snp_f1_val = float(snp_f1)
    except (TypeError, ValueError):
      raise SystemExit(f"Invalid snp_f1 value in metrics: {snp_f1!r}")
    if snp_f1_val < 0.99:
      raise SystemExit(f"snp_f1 {snp_f1_val:.4f} below expected demo threshold 0.99")
  if indel_f1 is not None:
    try:
      indel_f1_val = float(indel_f1)
    except (TypeError, ValueError):
      raise SystemExit(f"Invalid indel_f1 value in metrics: {indel_f1!r}")
    if indel_f1_val < 0.97:
      raise SystemExit(
          f"indel_f1 {indel_f1_val:.4f} below expected demo threshold 0.97"
      )


def main(argv: List[str]) -> None:
  if len(argv) > 2:
    raise SystemExit(f"Usage: {argv[0]} [RUN_DIR]")

  runs_root = Path(os.environ.get("OGN_RUNS_ROOT", "/work/runs")).resolve()

  if len(argv) == 2:
    run_dir = Path(argv[1]).resolve()
  else:
    run_dir = _find_latest_run(runs_root)

  print(f"[ogn-demo-metrics] checking run directory: {run_dir}")

  status = _load_json(run_dir / "status.json")
  metrics = _load_json(run_dir / "metrics.json")

  _check_status(status)
  _check_metrics(metrics, run_dir)

  print("[ogn-demo-metrics] invariants satisfied")


if __name__ == "__main__":  # pragma: no cover
  main(sys.argv)
