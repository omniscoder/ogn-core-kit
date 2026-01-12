from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .config import resolve_paths
from .util import print_json


def add_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
  parser = subparsers.add_parser(
      "results",
      help="Inspect outputs for the latest or a specific run.",
  )
  group = parser.add_mutually_exclusive_group()
  group.add_argument(
      "--latest",
      action="store_true",
      help="Show results for the most recent run.",
  )
  group.add_argument(
      "--run",
      metavar="PATH",
      help="Run directory path (e.g. runs/illumina_wgs_hg002/...).",
  )
  parser.add_argument(
      "--json",
      action="store_true",
      help="Emit a JSON summary instead of human text.",
  )


def _find_latest_run(runs_root: Path) -> Optional[Path]:
  if not runs_root.is_dir():
    return None
  candidates = []
  for profile_dir in runs_root.iterdir():
    if not profile_dir.is_dir():
      continue
    for run_dir in profile_dir.iterdir():
      if run_dir.is_dir():
        candidates.append(run_dir)
  if not candidates:
    return None
  return max(candidates)


def _load_json(path: Path) -> Dict[str, Any]:
  if not path.is_file():
    return {}
  try:
    return json.loads(path.read_text(encoding="utf-8"))
  except Exception:
    return {}


def run_results(args: argparse.Namespace) -> int:
  paths = resolve_paths()

  if args.run:
    run_dir = Path(args.run)
  else:
    run_dir = _find_latest_run(paths.runs_root)
    if run_dir is None:
      print("No runs found", flush=True)
      return 1

  config = _load_json(run_dir / "config.json")
  status = _load_json(run_dir / "status.json")
  metrics = _load_json(run_dir / "metrics.json")

  results_dir = run_dir / "results"
  vcf = None
  bam = None
  if results_dir.is_dir():
    for path in results_dir.iterdir():
      suffixes = path.suffixes
      if (len(suffixes) >= 2 and suffixes[-2:] == [".vcf", ".gz"]) or path.suffix == ".vcf":
        vcf = path
      if path.suffix in (".bam", ".cram"):
        bam = path

  summary: Dict[str, Any] = {
      "run_dir": str(run_dir),
      "profile": config.get("profile"),
      "sample": config.get("sample"),
      "state": status.get("state"),
      "vcf": str(vcf) if vcf else None,
      "bam": str(bam) if bam else None,
      "metrics": metrics,
  }

  if args.json:
    print_json(summary)
  else:
    print(f"Run: {run_dir}")
    print(f"Profile: {summary['profile']}")
    print(f"Sample: {summary['sample']}")
    print(f"State: {summary['state']}")
    print("Outputs:")
    print(f"  VCF: {summary['vcf']}")
    print(f"  BAM/CRAM: {summary['bam']}")

    acc = metrics.get("accuracy") or {}
    if acc:
      print()
      print("Accuracy vs GIAB HG002")
      print(f"  Panel        {acc.get('panel') or 'unknown'}")
      tv = acc.get("truth_vcf_version")
      tb = acc.get("truth_bed_version")
      if tv or tb:
        print(f"  Truth build  GRCh38 VCF {tv or '-'} BED {tb or '-'}")
      snp_f1 = acc.get("snp_f1")
      indel_f1 = acc.get("indel_f1")
      if snp_f1 is not None:
        try:
          print(f"  SNP F1       {float(snp_f1):.4f}")
        except (TypeError, ValueError):
          print(f"  SNP F1       {snp_f1}")
      if indel_f1 is not None:
        try:
          print(f"  INDEL F1     {float(indel_f1):.4f}")
        except (TypeError, ValueError):
          print(f"  INDEL F1     {indel_f1}")
      status_str = (acc.get("status") or "unknown").upper()
      print(f"  Status       {status_str}")

  return 0
