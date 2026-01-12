from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from .config import resolve_paths
from .util import GPUInfo, detect_gpu, find_binary, print_json, run_binary_version


def add_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
  parser = subparsers.add_parser(
      "doctor",
      help="Check GPU, OGN build, and data_root readiness.",
  )
  parser.add_argument(
      "--data-root",
      metavar="PATH",
      help="Override data root (default: /data or $OGN_DATA_ROOT).",
  )
  parser.add_argument(
      "--json",
      action="store_true",
      help="Emit a JSON summary instead of a human sentence.",
  )
  parser.add_argument(
      "--verbose",
      action="store_true",
      help="Include individual check results in human output.",
  )


def _check_data_root(path: Path) -> Dict[str, Any]:
  exists = path.exists()
  readable = os.access(path, os.R_OK) if exists else os.access(path.parent, os.R_OK)
  writable = os.access(path, os.W_OK) if exists else os.access(path.parent, os.W_OK)
  return {
      "path": str(path),
      "exists": exists,
      "readable": bool(readable),
      "writable": bool(writable),
  }


def _check_hg002_bundle(data_root: Path) -> Dict[str, Any]:
  bundle_root = data_root / "hg002"
  summary = data_root / "profiles" / "hg002_summary.json"
  status = "missing"
  missing: list[str] = []
  if bundle_root.is_dir():
    status = "present"
  else:
    missing.append(str(bundle_root))
  if summary.is_file():
    if status == "missing":
      status = "partial"
  else:
    missing.append(str(summary))
  return {
      "status": status,
      "bundle_root": str(bundle_root),
      "summary_path": str(summary),
      "missing": missing,
  }


def _check_hg002_truth(data_root: Path) -> Dict[str, Any]:
  expected = [
      data_root / "HG002_wgs.bam",
      data_root / "HG002_wgs.bam.bai",
      data_root / "GRCh38_noalt.fa",
      data_root / "GRCh38_noalt.fa.fai",
      data_root / "HG002_GRCh38_1_22_v4.2.1_benchmark.vcf.gz",
      data_root / "HG002_GRCh38_1_22_v4.2.1_benchmark.vcf.gz.tbi",
      data_root / "HG002_GRCh38_1_22_v4.2.1_benchmark.bed",
  ]
  missing: list[str] = []
  for path in expected:
    if not path.is_file():
      missing.append(str(path))
  if not expected:
    status = "missing"
  elif len(missing) == 0:
    status = "present"
  elif len(missing) == len(expected):
    status = "missing"
  else:
    status = "partial"
  return {
    "status": status,
    "missing": missing,
    "expected_count": len(expected),
    "data_root": str(data_root),
  }


def _run_core_doctor(repo_root: Path) -> Optional[Dict[str, Any]]:
  """Invoke scripts/ogn-doctor.sh and return its JSON payload, if any.

  The core shell doctor is the single source of truth for CUDA/OGX diagnostics.
  We drive it with OGN_DOCTOR_JSON_OUT so the Python CLI can pretty-print a
  stable summary.
  """
  script = repo_root / "scripts" / "ogn-doctor.sh"
  if not script.is_file():
    return None

  with tempfile.TemporaryDirectory() as tmpdir:
    json_path = Path(tmpdir) / "doctor_summary.json"
    env = os.environ.copy()
    env["OGN_DOCTOR_JSON_OUT"] = str(json_path)
    env.setdefault("OGN_DOCTOR_SKIP_COMPILERS", "0")
    try:
      subprocess.run(
          ["bash", str(script)],
          check=False,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          text=True,
          env=env,
      )
    except Exception:
      return None

    if not json_path.is_file():
      return None
    try:
      return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
      return None


def run_doctor(args: argparse.Namespace) -> int:
  paths = resolve_paths(data_root=args.data_root)

  gpu = detect_gpu()
  ogn_binary = find_binary(["ogn_run", "ogn_variant_runner"])
  ogn_version = run_binary_version(ogn_binary) if ogn_binary else None
  data_root_info = _check_data_root(paths.data_root)
  hg002_truth = _check_hg002_truth(paths.data_root)
  hg002_info = _check_hg002_bundle(paths.data_root)

  # When running from a source checkout or the demo Docker image, prefer the
  # core shell-based doctor for CUDA/OGX details.
  repo_root = Path(__file__).resolve().parents[3]
  core_doctor = _run_core_doctor(repo_root)

  status = "ok"
  messages: list[str] = []

  if not gpu.present:
    status = "warn"
    messages.append("GPU missing, CPU-only mode")
  if not ogn_binary:
    status = "fail"
    messages.append("OGN pipeline binary (ogn_run or ogn_variant_runner) not found")

  if hg002_truth["status"] == "missing":
    status = "fail"
    messages.append(
        "HG002 truth assets missing under "
        f"{paths.data_root}. Expected BAM, reference, VCF, BED (see docs/quickstart_demo.md)."
    )
  elif hg002_truth["status"] == "partial":
    status = "fail"
    missing = ", ".join(hg002_truth["missing"])
    messages.append(
        "HG002 truth assets incomplete under "
        f"{paths.data_root}. Missing: {missing}"
    )
  elif hg002_info["status"] != "present":
    if status != "fail":
      status = "warn"
    messages.append(
        "HG002 truth assets found; derived bundle not staged yet. "
        f'Run "ogn setup demo" to prepare under {paths.data_root}/hg002.'
    )

  if core_doctor and isinstance(core_doctor.get("summary"), str):
    summary = core_doctor["summary"]
  else:
    if status == "ok":
      summary = "OK GPU ready, OGN ready, GIAB bundle available"
    elif status == "warn":
      summary = "; ".join(["WARN"] + messages)
    else:
      summary = "; ".join(["FAIL"] + messages)

  payload: Dict[str, Any] = {
      "status": status,
      "summary": summary,
      "checks": {
          "gpu": {
              "present": gpu.present,
              "name": gpu.name,
          },
          "ogn_binary": {
              "path": str(ogn_binary) if ogn_binary else None,
              "version": ogn_version,
          },
          "data_root": data_root_info,
          "hg002_truth": hg002_truth,
          "hg002_bundle": hg002_info,
          "core_doctor": core_doctor or {},
      },
  }

  if args.json:
    print_json(payload)
  else:
    print(summary)
    if args.verbose:
      if gpu.present:
        print(f"- GPU: {gpu.name}")
      else:
        print("- GPU: none detected")
      if ogn_binary:
        ver = ogn_version or "unknown version"
        print(f"- OGN binary: {ogn_binary} ({ver})")
      else:
        print("- OGN binary: not found (ogn_run / ogn_variant_runner)")
      print(f"- data_root: {data_root_info}")
      print(f"- hg002 truth: {hg002_truth['status']}")
      print(f"- hg002 bundle: {hg002_info['status']}")

  return 0 if status in ("ok", "warn") else 1
