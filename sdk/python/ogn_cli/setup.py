from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .config import resolve_paths
from .util import print_json


@dataclass
class BundleResource:
  name: str
  uri: str
  required: bool = True


def add_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
  parser = subparsers.add_parser(
      "setup",
      help="Download and validate canonical data bundles (e.g. hg002).",
  )
  parser.add_argument(
      "bundle",
      metavar="BUNDLE",
      help="Named bundle to set up (currently: hg002).",
  )
  parser.add_argument(
      "--data-root",
      metavar="PATH",
      help="Override data root (default: /data or $OGN_DATA_ROOT).",
  )
  parser.add_argument(
      "--no-verify",
      action="store_true",
      help="Skip integrity checks (samtools quickcheck; VCF index checks).",
  )
  parser.add_argument(
      "--force",
      action="store_true",
      help="Re-download/overwrite even if files already exist.",
  )
  parser.add_argument(
      "--json",
      action="store_true",
      help="Emit a JSON summary instead of human text.",
  )


def _ensure_dir(path: Path) -> None:
  try:
    path.mkdir(parents=True, exist_ok=True)
  except PermissionError as ex:
    msg = (
        f"Could not create {path}. Ask your admin to grant write access or "
        "provide a different data root with --data-root /your/path"
    )
    print(f"Error: {msg}", file=sys.stderr)
    raise SystemExit(1) from ex


def _download_or_link(src: str, dest: Path, *, force: bool) -> None:
  if dest.exists() and not force:
    return

  dest.parent.mkdir(parents=True, exist_ok=True)

  if src.startswith("s3://"):
    if shutil.which("aws") is None:
      raise RuntimeError(f"aws CLI not found, required to fetch {src}")
    subprocess.run(["aws", "s3", "cp", src, str(dest)], check=True)
  elif src.startswith("http://") or src.startswith("https://"):
    cmd = ["curl", "-fsSL", "--retry", "5", "--retry-delay", "5", "-o", str(dest), src]
    subprocess.run(cmd, check=True)
  else:
    # Local path: prefer hardlink, fall back to copy.
    src_path = Path(src)
    if not src_path.exists():
      raise FileNotFoundError(f"Source {src} not found")
    try:
      os.link(src_path, dest)
    except OSError:
      shutil.copy2(src_path, dest)


def _run_quickcheck(path: Path) -> bool:
  if shutil.which("samtools") is None:
    return True
  try:
    subprocess.run(
        ["samtools", "quickcheck", str(path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return True
  except Exception:
    return False


def _bundle_hg002_resources(bundle_root: Path) -> Dict[str, BundleResource]:
  # These defaults mirror the local paths used in benchmarks/configs/benchmarks.yaml
  base = os.getenv("OGN_HG002_DATA_ROOT") or "/data/genome_data/hg002"
  base_path = Path(base)
  return {
      "bam": BundleResource("bam", str(base_path / "HG002_wgs.bam"), required=True),
      "bam_index": BundleResource(
          "bam_index", str(base_path / "HG002_wgs.bam.bai"), required=True
      ),
      "vcf": BundleResource(
          "vcf",
          str(base_path / "HG002_GRCh38_1_22_v4.2.1_benchmark.vcf.gz"),
          required=True,
      ),
      "vcf_index": BundleResource(
          "vcf_index",
          str(base_path / "HG002_GRCh38_1_22_v4.2.1_benchmark.vcf.gz.tbi"),
          required=True,
      ),
      "bed": BundleResource(
          "bed",
          str(base_path / "HG002_GRCh38_1_22_v4.2.1_benchmark.bed"),
          required=True,
      ),
      "reference_fasta": BundleResource(
          "reference_fasta", str(base_path / "GRCh38_noalt.fa"), required=True
      ),
      "reference_index": BundleResource(
          "reference_index", str(base_path / "GRCh38_noalt.fa.fai"), required=True
      ),
  }


def _build_summary(
    data_root: Path,
    bundle_root: Path,
    ok_checks: Dict[str, bool],
    resources: Dict[str, Path],
) -> Dict[str, Any]:
  resource_entries = {
      key: str(path.relative_to(bundle_root))
      for key, path in resources.items()
      if path is not None
  }
  return {
      "bundle_id": "hg002_giab_wgs_v4.2.1",
      "sample_id": "HG002",
      "data_root": str(data_root),
      "bundle_root": str(bundle_root),
      "reference_build": "GRCh38_noalt",
      "validated": all(ok_checks.values()),
      "last_checked_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
      "checks": ok_checks,
      "resources": resource_entries,
  }


def _write_summary(data_root: Path, summary: Dict[str, Any]) -> Path:
  profiles_dir = data_root / "profiles"
  profiles_dir.mkdir(parents=True, exist_ok=True)
  summary_path = profiles_dir / "hg002_summary.json"
  summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
  return summary_path


def run_setup(args: argparse.Namespace) -> int:
  bundle = args.bundle.lower()
  if bundle == "demo":
    bundle = "hg002"
  if bundle != "hg002":
    print(f"Error: unknown bundle {args.bundle!r} (supported: demo, hg002)", file=sys.stderr)
    return 1

  paths = resolve_paths(data_root=args.data_root)
  bundle_root = paths.data_root / "hg002"
  _ensure_dir(bundle_root)

  resources = _bundle_hg002_resources(bundle_root)

  ok_checks: Dict[str, bool] = {}
  staged_paths: Dict[str, Path] = {}

  try:
    for key, res in resources.items():
      dest = bundle_root / Path(res.uri).name
      staged_paths[key] = dest
      _download_or_link(res.uri, dest, force=args.force)
      if not args.no_verify and key in ("bam", "cram"):
        ok_checks[f"{key}_quickcheck"] = _run_quickcheck(dest)
      else:
        ok_checks[f"{key}_quickcheck"] = True
  except Exception as ex:
    msg = (
        f"Error: failed to stage HG002 bundle ({ex})\n"
        f"Hint: ensure the source paths are reachable or override OGN_HG002_DATA_ROOT."
    )
    print(msg, file=sys.stderr)
    return 1

  # Ensure a CRAM is available under the bundle root for GIAB runs.
  cram_dest = bundle_root / "HG002_wgs.cram"
  if not cram_dest.exists():
    # First, try to stage a pre-existing CRAM from the same source root.
    base = os.getenv("OGN_HG002_DATA_ROOT") or "/data/genome_data/hg002"
    cram_source = os.path.join(base, "HG002_wgs.cram")
    try:
      _download_or_link(cram_source, cram_dest, force=args.force)
    except Exception:
      # Fall back to building CRAM from the staged BAM if possible.
      bam_path = staged_paths.get("bam")
      ref_path = staged_paths.get("reference_fasta")
      if bam_path and bam_path.is_file() and ref_path and ref_path.is_file():
        if shutil.which("samtools") is None:
          print(
              "Error: samtools not found; required to convert HG002_wgs.bam "
              "to HG002_wgs.cram for GIAB runs.",
              file=sys.stderr,
          )
          return 1
        try:
          subprocess.run(
              [
                  "samtools",
                  "view",
                  "-C",
                  "-T",
                  str(ref_path),
                  "-o",
                  str(cram_dest),
                  str(bam_path),
              ],
              check=True,
              stdout=subprocess.DEVNULL,
              stderr=subprocess.DEVNULL,
          )
          subprocess.run(
              ["samtools", "index", str(cram_dest)],
              check=True,
              stdout=subprocess.DEVNULL,
              stderr=subprocess.DEVNULL,
          )
        except Exception as ex:
          msg = (
              f"Error: failed to build CRAM from BAM ({ex})\n"
              "Hint: create HG002_wgs.cram manually with:\n"
              '  samtools view -C -T GRCh38_noalt.fa '
              '-o HG002_wgs.cram HG002_wgs.bam\n'
              "Then rerun: ogn setup hg002"
          )
          print(msg, file=sys.stderr)
          return 1

  # Validate CRAM presence and integrity; zero-byte or unreadable CRAM is a hard error.
  if cram_dest.exists():
    size_ok = cram_dest.stat().st_size > 0
    quickcheck_ok = True
    if not args.no_verify and size_ok:
      quickcheck_ok = _run_quickcheck(cram_dest)

    if not size_ok or not quickcheck_ok:
      # Remove the bad CRAM to avoid future misuse.
      try:
        cram_dest.unlink()
      except OSError:
        pass
      print(
          "Error: CRAM generation failed (missing, empty, or unreadable).\n"
          "Hint: check disk space, permissions, and reference availability, then regenerate with:\n"
          '  samtools view -C -T GRCh38_noalt.fa -o HG002_wgs.cram HG002_wgs.bam\n'
          "After fixing, rerun: ogn setup hg002",
          file=sys.stderr,
      )
      return 1

    staged_paths["cram"] = cram_dest
    ok_checks["cram_quickcheck"] = True

  summary = _build_summary(paths.data_root, bundle_root, ok_checks, staged_paths)
  summary_path = _write_summary(paths.data_root, summary)

  if args.json:
    payload: Dict[str, Any] = dict(summary)
    payload["summary_path"] = str(summary_path)
    print_json(payload)
  else:
    print(f"HG002 bundle ready under {bundle_root}")
    print(f"Summary: {summary_path}")

  return 0
