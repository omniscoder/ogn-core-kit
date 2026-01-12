from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import resolve_paths
from .util import detect_gpu, print_json


@dataclass(frozen=True)
class CostModel:
  currency: str
  gpu_hourly_usd: Dict[str, float]
  egress_gb_usd: Optional[float]
  storage_gb_month_usd: Optional[float]
  cpu_hourly_usd: Optional[float]
  source: Optional[str] = None


def _env_float(name: str) -> Optional[float]:
  raw = os.environ.get(name)
  if raw is None:
    return None
  try:
    return float(raw)
  except Exception:
    return None


def _resolve_cost_model_path(explicit: Optional[str]) -> Optional[Path]:
  if explicit:
    return Path(explicit).expanduser()
  env = os.environ.get("OGN_COST_MODEL")
  if env:
    return Path(env).expanduser()

  for candidate in (
      Path.cwd() / "config" / "cost_model.json",
      Path(os.environ.get("OGN_HOME", "/opt/ogn")) / "config" / "cost_model.json",
      Path("/opt/ogn") / "config" / "cost_model.json",
  ):
    if candidate.is_file():
      return candidate
  return None


def _load_cost_model(path: Optional[Path]) -> CostModel:
  gpu_table: Dict[str, float] = {}
  egress = None
  storage = None
  cpu = None
  currency = "USD"
  source = str(path) if path else None

  payload: Dict[str, Any] = {}
  if path and path.is_file():
    try:
      payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
      payload = {}

  table = payload.get("gpu_hourly_usd") if isinstance(payload, dict) else None
  if isinstance(table, dict):
    for key, value in table.items():
      try:
        gpu_table[str(key)] = float(value)
      except Exception:
        continue

  if isinstance(payload, dict):
    if payload.get("egress_gb_usd") is not None:
      try:
        egress = float(payload.get("egress_gb_usd"))
      except Exception:
        egress = None
    if payload.get("storage_gb_month_usd") is not None:
      try:
        storage = float(payload.get("storage_gb_month_usd"))
      except Exception:
        storage = None
    if payload.get("cpu_hourly_usd") is not None:
      try:
        cpu = float(payload.get("cpu_hourly_usd"))
      except Exception:
        cpu = None

  # Environment overrides (as documented in README).
  gpu_override = _env_float("OGN_GPU_HOURLY_USD")
  if gpu_override is not None:
    gpu_table = {"ENV": gpu_override}
  egress_override = _env_float("OGN_EGRESS_GB_USD")
  if egress_override is not None:
    egress = egress_override
  storage_override = _env_float("OGN_STORAGE_GB_MONTH_USD")
  if storage_override is not None:
    storage = storage_override
  cpu_override = _env_float("OGN_CPU_HOURLY_USD")
  if cpu_override is not None:
    cpu = cpu_override

  return CostModel(
      currency=currency,
      gpu_hourly_usd=gpu_table,
      egress_gb_usd=egress,
      storage_gb_month_usd=storage,
      cpu_hourly_usd=cpu,
      source=source,
  )


def _sum_file_bytes(root: Path) -> int:
  total = 0
  if not root.exists():
    return 0
  if root.is_file():
    try:
      return int(root.stat().st_size)
    except Exception:
      return 0
  for path in root.rglob("*"):
    try:
      if path.is_file() and not path.is_symlink():
        total += int(path.stat().st_size)
    except Exception:
      continue
  return total


def _read_json(path: Path) -> Dict[str, Any]:
  if not path.is_file():
    return {}
  try:
    return json.loads(path.read_text(encoding="utf-8"))
  except Exception:
    return {}


def _find_latest_run_for_profile(
    runs_root: Path, profile: str, sample: Optional[str]
) -> Optional[Path]:
  if not runs_root.is_dir():
    return None

  candidates: List[Path] = []
  prefix = f"{profile}_"
  exact = f"{profile}_{sample}" if sample else None

  for group_dir in runs_root.iterdir():
    if not group_dir.is_dir():
      continue
    if group_dir.name == profile:
      candidates.append(group_dir)
      continue
    if exact and group_dir.name != exact:
      continue
    if not exact and not group_dir.name.startswith(prefix):
      continue
    for run_dir in group_dir.iterdir():
      if run_dir.is_dir():
        candidates.append(run_dir)

  if not candidates:
    return None

  # Use mtime so both timestamped and non-timestamped layouts behave.
  def _mtime(path: Path) -> float:
    try:
      return path.stat().st_mtime
    except Exception:
      return 0.0

  return max(candidates, key=_mtime)


def _select_gpu_rate(
    model: CostModel, *, gpu_sku: Optional[str], gpu_name: Optional[str]
) -> Tuple[Optional[str], Optional[float]]:
  if not model.gpu_hourly_usd:
    return None, None

  if gpu_sku:
    if gpu_sku not in model.gpu_hourly_usd:
      raise RuntimeError(
          f"Unknown GPU SKU {gpu_sku!r}; available: {', '.join(sorted(model.gpu_hourly_usd))}"
      )
    return gpu_sku, model.gpu_hourly_usd[gpu_sku]

  if gpu_name:
    name = gpu_name.lower()
    matches: List[Tuple[str, float]] = []
    for sku, rate in model.gpu_hourly_usd.items():
      token = sku.split("_", 1)[0].lower()
      if token and token in name:
        matches.append((sku, rate))
    if len(matches) == 1:
      return matches[0]
    if len(matches) > 1:
      # Prefer the longest SKU key match (more specific).
      matches.sort(key=lambda item: len(item[0]), reverse=True)
      return matches[0]

  if len(model.gpu_hourly_usd) == 1:
    sku, rate = next(iter(model.gpu_hourly_usd.items()))
    return sku, rate

  return None, None


def _default_bundle_dir(run_dir: Path, runs_root: Path) -> Path:
  try:
    rel = run_dir.relative_to(runs_root)
  except ValueError:
    return Path.cwd() / "proof_bundles" / run_dir.name
  return runs_root / "proof_bundles" / rel


def build_cost_estimate(
    *,
    profile: str,
    run_dir: Path,
    runs_root: Path,
    cost_model: CostModel,
    gpu_sku: Optional[str],
    retention_days: int,
    egress_scope: str,
) -> Dict[str, Any]:
  metrics = _read_json(run_dir / "metrics.json")
  status = _read_json(run_dir / "status.json")

  wall = metrics.get("wall_time_seconds")
  if wall is None:
    wall = metrics.get("runtime_seconds")
  try:
    wall_seconds = float(wall) if wall is not None else None
  except Exception:
    wall_seconds = None

  if wall_seconds is None:
    raise RuntimeError(f"Run metrics missing wall_time_seconds/runtime_seconds in {run_dir}/metrics.json")

  gpu = metrics.get("gpu") or {}
  gpu_name = gpu.get("device") if isinstance(gpu, dict) else None
  gpu_count = 0
  if isinstance(gpu, dict) and gpu.get("count") is not None:
    try:
      gpu_count = int(gpu.get("count"))
    except Exception:
      gpu_count = 0

  # If the run did not record a GPU name, fall back to current detection.
  if not gpu_name:
    detected = detect_gpu()
    gpu_name = detected.name

  gpu_hours = max(0.0, wall_seconds) / 3600.0 * max(0, gpu_count)
  cpu_hours = max(0.0, wall_seconds) / 3600.0 if gpu_count <= 0 else 0.0

  storage_bytes = _sum_file_bytes(run_dir)
  results_dir = run_dir / "results"
  bundle_dir = _default_bundle_dir(run_dir, runs_root)

  egress_bytes = 0
  egress_note = None
  scope = egress_scope.lower()
  if scope == "none":
    egress_bytes = 0
  elif scope == "results":
    egress_bytes = _sum_file_bytes(results_dir)
    if not results_dir.is_dir():
      egress_note = "results/ missing"
  elif scope in ("run", "run-dir", "all"):
    egress_bytes = storage_bytes
  elif scope in ("bundle", "proof-bundle"):
    if bundle_dir.is_dir():
      egress_bytes = _sum_file_bytes(bundle_dir)
    else:
      egress_note = f"proof bundle missing at {bundle_dir}"
      egress_bytes = 0
  else:
    raise RuntimeError("egress-scope must be one of: results, run-dir, bundle, none")

  storage_gb = storage_bytes / 1e9
  egress_gb = egress_bytes / 1e9

  selected_sku, gpu_hourly = _select_gpu_rate(
      cost_model, gpu_sku=gpu_sku, gpu_name=gpu_name
  )

  compute_cost = None
  cpu_cost = None
  if gpu_hourly is not None:
    compute_cost = gpu_hours * gpu_hourly
  if cost_model.cpu_hourly_usd is not None and cpu_hours > 0:
    cpu_cost = cpu_hours * cost_model.cpu_hourly_usd

  storage_cost = None
  if cost_model.storage_gb_month_usd is not None:
    storage_cost = storage_gb * cost_model.storage_gb_month_usd * (retention_days / 30.0)

  egress_cost = None
  if cost_model.egress_gb_usd is not None:
    egress_cost = egress_gb * cost_model.egress_gb_usd

  components = [compute_cost, cpu_cost, storage_cost, egress_cost]
  total_cost = sum(v for v in components if isinstance(v, (int, float)))

  missing_rates: List[str] = []
  if gpu_hours > 0 and gpu_hourly is None:
    missing_rates.append("gpu_hourly_usd")
  if cpu_hours > 0 and cost_model.cpu_hourly_usd is None:
    missing_rates.append("cpu_hourly_usd")
  if storage_bytes > 0 and cost_model.storage_gb_month_usd is None:
    missing_rates.append("storage_gb_month_usd")
  if egress_bytes > 0 and cost_model.egress_gb_usd is None:
    missing_rates.append("egress_gb_usd")
  total_is_partial = bool(missing_rates)

  return {
      "profile": profile,
      "run_dir": str(run_dir),
      "run_state": status.get("state"),
      "wall_time_seconds": wall_seconds,
      "gpu": {"device": gpu_name, "count": gpu_count},
      "estimates": {
          "gpu_hours": gpu_hours,
          "cpu_hours": cpu_hours,
          "storage_bytes": storage_bytes,
          "storage_gb": storage_gb,
          "egress_scope": egress_scope,
          "egress_note": egress_note,
          "egress_bytes": egress_bytes,
          "egress_gb": egress_gb,
          "retention_days": retention_days,
      },
      "cost_model": {
          "source": cost_model.source,
          "currency": cost_model.currency,
          "gpu_sku": selected_sku,
          "gpu_hourly_usd": gpu_hourly,
          "cpu_hourly_usd": cost_model.cpu_hourly_usd,
          "storage_gb_month_usd": cost_model.storage_gb_month_usd,
          "egress_gb_usd": cost_model.egress_gb_usd,
      },
      "cost_estimate": {
          "compute_usd": compute_cost,
          "cpu_usd": cpu_cost,
          "storage_usd": storage_cost,
          "egress_usd": egress_cost,
          "total_usd": total_cost,
          "total_is_partial": total_is_partial,
          "missing_rates": missing_rates,
      },
  }


def add_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
  parser = subparsers.add_parser(
      "estimate-cost",
      help="Estimate per-run GPU/storage/egress cost from the latest run of a profile.",
  )
  parser.add_argument(
      "profile",
      metavar="PROFILE",
      help="Profile name (e.g. illumina_wgs). Uses the most recent run under $OGN_RUNS_ROOT.",
  )
  parser.add_argument(
      "--sample",
      metavar="SAMPLE",
      help="Optional sample filter (selects the latest run under runs/<profile>_<sample>/).",
  )
  parser.add_argument(
      "--run",
      metavar="PATH",
      help="Explicit run directory path (bypasses profile lookup).",
  )
  parser.add_argument(
      "--runs-root",
      metavar="PATH",
      help="Override runs root (default: $OGN_RUNS_ROOT or ./runs).",
  )
  parser.add_argument(
      "--cost-model",
      metavar="PATH",
      help="Cost model JSON (default: $OGN_COST_MODEL or ./config/cost_model.json).",
  )
  parser.add_argument(
      "--gpu-sku",
      metavar="SKU",
      help="Explicit GPU SKU key from cost_model.gpu_hourly_usd (default: auto-detect).",
  )
  parser.add_argument(
      "--retention-days",
      type=int,
      default=30,
      help="Retention window used to convert storage GB-month into per-run cost (default: 30).",
  )
  parser.add_argument(
      "--egress-scope",
      choices=["results", "run-dir", "bundle", "none"],
      default="results",
      help="What counts as egress (default: results).",
  )
  parser.add_argument(
      "--json",
      action="store_true",
      help="Emit JSON instead of human text.",
  )


def run_estimate_cost(args: argparse.Namespace) -> int:
  paths = resolve_paths(runs_root=args.runs_root)
  runs_root = paths.runs_root

  if args.run:
    run_dir = Path(args.run).expanduser()
  else:
    run_dir = _find_latest_run_for_profile(runs_root, args.profile, args.sample)
    if run_dir is None:
      print(
          f"Error: no runs found for profile={args.profile!r} under {runs_root}. "
          f'Run "ogn run {args.profile} <sample>" first or pass --run <path>.',
          file=sys.stderr,
      )
      return 2

  cost_model_path = _resolve_cost_model_path(args.cost_model)
  model = _load_cost_model(cost_model_path)

  try:
    estimate = build_cost_estimate(
        profile=args.profile,
        run_dir=run_dir,
        runs_root=runs_root,
        cost_model=model,
        gpu_sku=args.gpu_sku,
        retention_days=max(0, int(args.retention_days or 0)),
        egress_scope=args.egress_scope,
    )
  except Exception as ex:
    print(f"Error: {ex}", file=sys.stderr)
    return 2

  if args.json:
    print_json(estimate)
    return 0

  est = estimate["estimates"]
  cost = estimate["cost_estimate"]
  cm = estimate["cost_model"]
  gpu = estimate["gpu"]

  print(f"Profile: {estimate['profile']}")
  print(f"Run: {estimate['run_dir']}")
  print(f"State: {estimate.get('run_state') or 'unknown'}")
  print(f"GPU: {gpu.get('device') or 'none'} (count={gpu.get('count')})")
  print(f"Wall: {estimate['wall_time_seconds']:.2f}s")
  print()
  print("Estimates (per run):")
  print(f"  GPU hours:  {est['gpu_hours']:.4f}")
  if est.get("cpu_hours"):
    print(f"  CPU hours:  {est['cpu_hours']:.4f}")
  print(f"  Storage:   {est['storage_gb']:.4f} GB ({est['storage_bytes']} bytes)")
  print(
      f"  Egress:    {est['egress_gb']:.4f} GB ({est['egress_bytes']} bytes; scope={est['egress_scope']})"
  )
  if est.get("egress_note"):
    print(f"            note: {est['egress_note']}")
  print(f"  Retention: {est['retention_days']} days")
  print()
  print(f"Costs ({cm.get('currency') or 'USD'}; {cm.get('source') or 'no cost model'}):")
  if cost.get("compute_usd") is None and est.get("gpu_hours", 0) > 0:
    print("  GPU compute: (rate unknown; set --gpu-sku or OGN_GPU_HOURLY_USD)")
  else:
    sku = cm.get("gpu_sku") or "unknown"
    rate = cm.get("gpu_hourly_usd")
    if rate is None:
      print("  GPU compute: $0.00")
    else:
      print(f"  GPU compute: ${cost['compute_usd']:.4f} @ ${rate:.4f}/hr ({sku})")
  if cost.get("cpu_usd") is not None:
    rate = cm.get("cpu_hourly_usd")
    print(f"  CPU compute: ${cost['cpu_usd']:.4f} @ ${rate:.4f}/hr")
  if cost.get("storage_usd") is not None:
    rate = cm.get("storage_gb_month_usd")
    print(
        f"  Storage:    ${cost['storage_usd']:.4f} @ ${rate:.4f}/GB-month (scaled to {est['retention_days']}d)"
    )
  if cost.get("egress_usd") is not None:
    rate = cm.get("egress_gb_usd")
    print(f"  Egress:     ${cost['egress_usd']:.4f} @ ${rate:.4f}/GB")
  total_suffix = " (partial)" if cost.get("total_is_partial") else ""
  print(f"  Total{total_suffix}: ${cost['total_usd']:.4f}")
  if cost.get("missing_rates"):
    print(f"  Missing rates: {', '.join(cost['missing_rates'])}")

  return 0
