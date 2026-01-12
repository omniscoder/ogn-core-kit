from __future__ import annotations

import argparse
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import List, Optional

from . import doctor as doctor_cmd
from . import estimate_cost as estimate_cost_cmd
from . import proof_bundle as proof_bundle_cmd
from . import results as results_cmd
from . import run as run_cmd
from . import setup as setup_cmd


def _version_string() -> str:
  try:
    pkg_version = metadata.version("ogn-sdk")
  except metadata.PackageNotFoundError:
    pkg_version = "unknown"
  return f"ogn {pkg_version}"


def _maybe_run_script(argv: List[str]) -> Optional[int]:
  """If the first arg looks like a shell script path, run it directly.

  This makes `docker run ... ogn-demo scripts/ogn-demo-smoke.sh` behave as
  documented while keeping the CLI entrypoint for regular use.
  """
  if not argv:
    return None

  script_arg = argv[0]
  if not (script_arg.endswith(".sh") or "/" in script_arg):
    return None

  ogn_home = Path(os.environ.get("OGN_HOME", "/opt/ogn"))
  candidate = Path(script_arg)
  if not candidate.is_absolute():
    candidate = (ogn_home / candidate).resolve()

  if not candidate.exists():
    return None

  cmd = ["/bin/bash", str(candidate), *argv[1:]]
  return subprocess.call(cmd)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      prog="ogn",
      description="OGN command-line helper for junior-friendly workflows "
      "(doctor → setup → run → results).",
  )
  parser.add_argument(
      "--version",
      action="version",
      version=_version_string(),
      help="Show CLI version and exit.",
  )
  subparsers = parser.add_subparsers(dest="command", required=True)

  doctor_cmd.add_subparser(subparsers)
  setup_cmd.add_subparser(subparsers)
  run_cmd.add_subparser(subparsers)
  results_cmd.add_subparser(subparsers)
  proof_bundle_cmd.add_subparser(subparsers)
  estimate_cost_cmd.add_subparser(subparsers)

  return parser


def main(argv: Optional[List[str]] = None) -> int:
  argv = list(sys.argv[1:] if argv is None else argv)
  script_rc = _maybe_run_script(argv)
  if script_rc is not None:
    return script_rc

  # Friendlier handling for the demo path and missing SAMPLE.
  if argv[:1] == ["run"]:
    if len(argv) == 2 and argv[1].lower() == "demo":
      # Allow `ogn run demo` to imply the canonical HG002 sample.
      argv = ["run", "demo", "hg002", *argv[2:]]
    elif len(argv) < 3:
      print(
          "Error: ogn run expects PROFILE and SAMPLE "
          "(e.g. ogn run demo HG002 --data-root /data/genome_data/hg002)",
          file=sys.stderr,
      )
      return 2

  parser = build_parser()
  args = parser.parse_args(argv)

  if args.command == "doctor":
    return doctor_cmd.run_doctor(args)
  if args.command == "setup":
    return setup_cmd.run_setup(args)
  if args.command == "run":
    # Convenience: "ogn run demo" is treated as the canonical HG002 GIAB demo.
    if getattr(args, "profile", "").lower() == "demo":
      args.profile = "illumina_wgs"
      args.sample = "hg002"
    return run_cmd.run_run(args)
  if args.command == "results":
    return results_cmd.run_results(args)
  if args.command == "proof-bundle":
    return proof_bundle_cmd.run_proof_bundle(args)
  if args.command == "estimate-cost":
    return estimate_cost_cmd.run_estimate_cost(args)

  parser.error(f"Unknown command {args.command!r}")
  return 1


if __name__ == "__main__":  # pragma: no cover
  raise SystemExit(main())
