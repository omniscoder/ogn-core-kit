from __future__ import annotations

import argparse
import csv
import json
import os
import importlib.util
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .config import resolve_paths
from .util import detect_gpu, find_binary, print_json


@dataclass
class RunContext:
  profile: str
  sample: str
  data_root: Path
  runs_root: Path
  run_dir: Path
  gpu_name: Optional[str] = None


@dataclass
class BundleSummary:
  data_root: Path
  bundle_root: Path
  reference_fasta: Path
  truth_vcf: Optional[Path]
  truth_bed: Optional[Path]
  cram: Optional[Path]
  bam: Optional[Path]


@dataclass
class SimpleInputs:
  sample: str
  sample_dir: Path
  cram: Optional[Path]
  bam: Optional[Path]
  reference_fasta: Path
  reference_index: Path


def add_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
  parser = subparsers.add_parser(
      "run",
      help="Run a standard analysis profile (golden path: illumina_wgs + hg002).",
  )
  parser.add_argument(
      "profile",
      metavar="PROFILE",
      help="Profile name (e.g. illumina_wgs or demo).",
  )
  parser.add_argument(
      "sample",
      metavar="SAMPLE",
      help="Sample name (e.g. hg002). For the demo path, 'ogn run demo' will default SAMPLE to HG002.",
  )
  parser.add_argument(
      "--data-root",
      metavar="PATH",
      help="Override data root (default: /data or $OGN_DATA_ROOT).",
  )
  parser.add_argument(
      "--explain",
      choices=["yes", "no"],
      default="no",
      help="Explain major stages in plain language.",
  )
  parser.add_argument(
      "--json",
      action="store_true",
      help="Emit a JSON summary as well as text.",
  )
  parser.add_argument(
      "--require-giab-validation",
      action="store_true",
      help="Fail the run if GIAB validation tooling is unavailable (default: warn and continue).",
  )
  parser.add_argument(
      "--skip-validation",
      action="store_true",
      help="Skip GIAB validation entirely (pipeline still runs).",
  )
  parser.add_argument(
      "--watchdog-seconds",
      type=int,
      default=120,
      help="Watchdog timeout for run progress (default: 120s).",
  )


def _load_hg002_summary(data_root: Path) -> Optional[Dict[str, Any]]:
  summary_path = data_root / "profiles" / "hg002_summary.json"
  if not summary_path.is_file():
    return None
  try:
    return json.loads(summary_path.read_text(encoding="utf-8"))
  except Exception:
    return None


def _run_variant_runner(
    cmd: list[str], workdir: Path, log_path: Path, spawn_info: Dict[str, Any]
) -> Tuple[int, float, str, str]:
  """Run ogn_variant_runner, streaming output to both stdout and a log file."""
  import time

  start_ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
  start = time.time()

  log_path.parent.mkdir(parents=True, exist_ok=True)
  # Prepend a spawn header so an empty log still tells us what we tried.
  with log_path.open("w", encoding="utf-8") as log_handle:
    header = {
        "cmd": cmd,
        "cwd": str(workdir),
        "input_type": spawn_info.get("input_type"),
        "env": spawn_info.get("env", {}),
    }
    log_handle.write(f"[ogn-run] spawn {json.dumps(header)}\n")
    log_handle.flush()
    if log_path.stat().st_size == 0:
      raise RuntimeError(f"Failed to write engine log header at {log_path}")
    proc = subprocess.Popen(
        cmd,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(f"[ogn-run] running_variant_runner pid={proc.pid}", file=sys.stderr, flush=True)
    stream = getattr(proc, "stdout", None)
    if stream is not None:
      for line in stream:
        sys.stdout.write(line)
        log_handle.write(line)
    rc = proc.wait()

  wall = time.time() - start
  finish_ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
  return rc, wall, start_ts, finish_ts


def _derive_run_dir(paths, profile: str, sample: str) -> Path:
  ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  name = f"{profile}_{sample}"
  run_dir = paths.runs_root / name / ts
  run_dir.mkdir(parents=True, exist_ok=True)
  (run_dir / "logs").mkdir(exist_ok=True)
  (run_dir / "results").mkdir(exist_ok=True)
  return run_dir


def _write_status(run_dir: Path, state: str, error: Optional[Dict[str, Any]] = None) -> None:
  status_path = run_dir / "status.json"
  now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
  payload: Dict[str, Any] = {
      "state": state,
      "updated_utc": now,
  }
  if error:
    payload["error"] = error
  status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_initial_status_and_metrics(
    run_ctx: RunContext, gpu_present: bool, gpu_name: Optional[str], gpu_count: int
) -> None:
  now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
  # Status: running
  status_path = run_ctx.run_dir / "status.json"
  status_payload = {
      "state": "running",
      "updated_utc": now,
      "profile": run_ctx.profile,
      "sample": run_ctx.sample,
      "run_dir": str(run_ctx.run_dir),
  }
  status_path.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")

  # Metrics stub
  metrics_path = run_ctx.run_dir / "metrics.json"
  metrics_payload = {
      "sample": run_ctx.sample,
      "profile": run_ctx.profile,
      "started_utc": now,
      "finished_utc": None,
      "wall_time_seconds": None,
      "runtime_seconds": None,
      "reference_build": None,
      "vcf_path": None,
      "bam_or_cram": None,
      "gpu": {"device": gpu_name if gpu_present else None, "count": gpu_count},
      "validation_status": None,
      "validation_detail": {},
  }
  metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")


def _write_config(run_ctx: RunContext, engine_cmd: list[str]) -> None:
  config_path = run_ctx.run_dir / "config.json"
  payload = {
      "profile": run_ctx.profile,
      "sample": run_ctx.sample,
      "data_root": str(run_ctx.data_root),
      "run_dir": str(run_ctx.run_dir),
      "engine": {
          "argv": engine_cmd,
      },
  }
  config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _bundle_from_summary(data_root: Path, summary: Dict[str, Any]) -> BundleSummary:
  bundle_root = Path(summary.get("bundle_root", data_root / "hg002"))
  resources = summary.get("resources", {})

  def _resolve(key: str, default_names: Tuple[str, ...]) -> Optional[Path]:
    rel = resources.get(key)
    if rel:
      candidate = bundle_root / rel
      if candidate.is_file():
        return candidate
    for name in default_names:
      candidate = bundle_root / name
      if candidate.is_file():
        return candidate
    return None

  reference = _resolve("reference_fasta", ("GRCh38_noalt.fa",))
  truth_vcf = _resolve(
      "vcf", ("HG002_GRCh38_1_22_v4.2.1_benchmark.vcf.gz",)
  )
  truth_bed = _resolve(
      "bed", ("HG002_GRCh38_1_22_v4.2.1_benchmark.bed",)
  )
  cram = _resolve("cram", ("HG002_wgs.cram",))
  bam = _resolve("bam", ("HG002_wgs.bam",))

  if reference is None:
    raise RuntimeError(f"Reference FASTA not found under {bundle_root}")

  return BundleSummary(
      data_root=data_root,
      bundle_root=bundle_root,
      reference_fasta=reference,
      truth_vcf=truth_vcf,
      truth_bed=truth_bed,
      cram=cram,
      bam=bam,
  )


def _resolve_simple_inputs(data_root: Path, sample: str) -> SimpleInputs:
  sample_dir = data_root / sample
  cram = sample_dir / f"{sample}.cram"
  bam = sample_dir / f"{sample}.bam"

  if not cram.is_file() and not bam.is_file():
    raise RuntimeError(
        f"No CRAM or BAM found for sample {sample} under {sample_dir}"
    )
  if cram.is_file():
    input_path = cram
  else:
    # Runner currently expects CRAM; fail fast with guidance.
    raise RuntimeError(
        f"CRAM required for sample {sample}. Found BAM but no CRAM under {sample_dir}. "
        'Convert once with: samtools view -C -T GRCh38_noalt.fa -o '
        f"{sample}.cram {sample}.bam"
    )

  ref_dir = data_root / "reference"
  reference = ref_dir / "GRCh38_noalt.fa"
  reference_index = ref_dir / "GRCh38_noalt.fa.fai"
  if not reference.is_file() or not reference_index.is_file():
    raise RuntimeError(f"Reference FASTA or index not found under {ref_dir}")

  return SimpleInputs(
      sample=sample,
      sample_dir=sample_dir,
      cram=input_path if input_path.is_file() else None,
      bam=None,
      reference_fasta=reference,
      reference_index=reference_index,
  )


def _extract_accuracy_from_hap_summary(
    csv_path: Path, giab_metrics: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
  if not csv_path or not csv_path.is_file():
    return None

  snp_f1: Optional[float] = None
  indel_f1: Optional[float] = None

  with csv_path.open("r", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      t = (row.get("Type") or row.get("TYPE") or "").upper()
      f1 = (
          row.get("F1_Score")
          or row.get("F1")
          or row.get("F1_Score_Overall")
      )
      if not f1:
        continue
      try:
        f1_val = float(f1)
      except ValueError:
        continue
      if t == "SNP" and snp_f1 is None:
        snp_f1 = f1_val
      elif t == "INDEL" and indel_f1 is None:
        indel_f1 = f1_val

  if snp_f1 is None and indel_f1 is None:
    return None

  return {
      "panel": giab_metrics.get("panel"),
      "truth_vcf_version": giab_metrics.get("truth_vcf_version"),
      "truth_bed_version": giab_metrics.get("truth_bed_version"),
      "snp_f1": snp_f1,
      "indel_f1": indel_f1,
      "status": "pass",
  }


def _run_hg002_pipeline_only(
    run_ctx: RunContext,
    bundle: BundleSummary,
    gpu_count: int,
    validation_status: str,
    validation_detail: Optional[Dict[str, Any]] = None,
    finalize: bool = True,
    set_phase=None,
) -> Tuple[int, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
  """Run the pipeline without GIAB validation, treating validation as skipped."""
  print("[planning] build_stage_plan step=find_binary start", file=sys.stderr, flush=True)
  runner = find_binary(["ogn_variant_runner"])
  if not runner:
    raise RuntimeError("ogn_variant_runner binary not found (build the project first)")
  print(f"[planning] build_stage_plan step=find_binary done path={runner}", file=sys.stderr, flush=True)

  logs_dir = run_ctx.run_dir / "logs"
  results_dir = run_ctx.run_dir / "results"
  logs_dir.mkdir(parents=True, exist_ok=True)
  results_dir.mkdir(parents=True, exist_ok=True)

  sample_id = "HG002"
  ref = bundle.reference_fasta
  # Input selection: runner currently supports CRAM, not BAM. Enforce that and validate.
  input_path = None
  input_type = None
  quickcheck_enabled = os.environ.get("OGN_CRAM_QUICKCHECK", "1").lower() in (
      "1",
      "true",
      "yes",
  )

  if bundle.cram and bundle.cram.is_file():
    input_path = bundle.cram
    input_type = "cram"
    # Verify CRAM is readable to avoid opaque hangs.
    if quickcheck_enabled:
      try:
        subprocess.check_call(
            ["samtools", "quickcheck", str(input_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
      except FileNotFoundError:
        # samtools not present; continue without verification.
        pass
      except subprocess.CalledProcessError:
        raise RuntimeError(
            f"CRAM failed quickcheck: {input_path}. Regenerate with:\n"
            'samtools view -C -T GRCh38_noalt.fa -o HG002_wgs.cram HG002_wgs.bam'
        )
  elif bundle.bam and bundle.bam.is_file():
    # Runner has no --bam flag; fail fast with guidance.
    raise RuntimeError(
        "ogn_variant_runner requires CRAM input. Found BAM but no CRAM; "
        'convert once with: samtools view -C -T GRCh38_noalt.fa -o HG002_wgs.cram HG002_wgs.bam'
    )
  if not input_path or not input_path.is_file():
    raise RuntimeError("No CRAM or BAM found for HG002 bundle")
  if input_path.suffix.lower() == ".bam":
    # Defensive: never pass BAM to --cram.
    raise RuntimeError(
        "internal error: selected BAM but would invoke CRAM reader; aborting"
    )
  if input_type != "cram":
    raise RuntimeError("internal error: input type undetermined; aborting before spawn")
  print(
      f"[planning] build_stage_plan step=select_input done input={input_path}",
      file=sys.stderr,
      flush=True,
  )

  vcf_name = f"{sample_id}.g.vcf.gz"
  vcf_path = results_dir / vcf_name

  cmd = [
      str(runner),
      "--reference",
      str(ref),
      "--sample",
      sample_id,
      "--streams-per-gpu",
      "1",
  ]
  # Only CRAM is supported; guard against misuse.
  cmd.extend(["--cram", str(input_path)])

  engine_log = logs_dir / "engine.log"

  if set_phase:
    set_phase("launching_variant_runner")
  print("stage=pipeline name=variant_runner start", flush=True)
  if set_phase:
    set_phase("running_variant_runner")
  spawn_env = {
      "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
      "OGN_SKIP_VALIDATION": os.environ.get("OGN_SKIP_VALIDATION"),
  }
  rc, wall, started_ts, finished_ts = _run_variant_runner(
      cmd,
      results_dir,
      engine_log,
      {"input_type": input_type, "env": spawn_env},
  )
  if set_phase:
    set_phase(None)
  print(f"stage=pipeline name=variant_runner done exit_code={rc}", flush=True)

  if rc != 0:
    error = {
        "title": "Variant calling failed",
        "hint": "Inspect the engine log for details.",
        "log_path": str(engine_log.relative_to(run_ctx.run_dir)),
    }
    if finalize:
      _write_status(run_ctx.run_dir, "failed", error)
      print("Run failed")
      print(f"Profile {run_ctx.profile}")
      print(f"Sample {run_ctx.sample}")
      log_display = error["log_path"] or "(no log)"
      print(
          f"Error: {error['title']}\nHint: {error['hint']}\nDetails: {log_display}"
      )
    return rc or 1, None, error

  rel_vcf = vcf_path.relative_to(run_ctx.run_dir)
  metrics = {
      "sample": sample_id,
      "profile": run_ctx.profile,
      "wall_time_seconds": wall,
      "runtime_seconds": wall,
      "started_utc": started_ts,
      "finished_utc": finished_ts,
      "reference_build": "GRCh38",
      "vcf_path": str(rel_vcf),
      "bam_or_cram": str(input_path),
      "gpu": {
          "device": run_ctx.gpu_name,
          "count": gpu_count,
      },
      "accuracy": None,
      "validation_status": validation_status,
      "giab_metrics_path": None,
      "hap_py_summary_path": None,
      "validation_detail": validation_detail or {},
  }

  if finalize:
    metrics_path = run_ctx.run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_status(run_ctx.run_dir, "succeeded", None)

  return 0, metrics, None


def _run_hg002_giab(
    run_ctx: RunContext,
    bundle: BundleSummary,
    gpu_count: int,
    require_validation: bool,
    skip_validation: bool,
    set_phase=None,
) -> Tuple[int, Optional[Dict[str, Any]], str, Dict[str, Any]]:
  # Discover helper as a package module.
  helper_spec = importlib.util.find_spec("ogn_cli.tools.run_giab_validation")
  helper_available = helper_spec is not None

  # GIAB path still requires the CRAM to exist; pipeline may run from BAM, but
  # validation needs the CRAM layout to be present.
  if not skip_validation and not (bundle.cram and bundle.cram.is_file()):
    raise RuntimeError("GIAB mode currently requires CRAM input for HG002")

  # First, always run the core pipeline.
  print("[ogn-run] planning:build_stage_plan start", file=sys.stderr, flush=True)
  rc_pipe, pipeline_metrics, pipe_error = _run_hg002_pipeline_only(
      run_ctx,
      bundle,
      gpu_count,
      validation_status="pending",
      validation_detail={"reason": "validation_pending"},
      finalize=False,
      set_phase=set_phase,
  )
  print("[ogn-run] planning:build_stage_plan done", file=sys.stderr, flush=True)
  if rc_pipe != 0 or pipe_error:
    err = pipe_error or {
        "title": "Variant calling failed",
        "hint": "Inspect the engine log for details.",
        "log_path": None,
    }
    _write_status(run_ctx.run_dir, "failed", err)
    print("Run failed")
    print(f"Profile {run_ctx.profile}")
    print(f"Sample {run_ctx.sample}")
    log_display = err.get("log_path") or "(no log)"
    print(
        f"Error: {err.get('title')}\nHint: {err.get('hint')}\nDetails: {log_display}"
    )
    return rc_pipe or 1, None, "pipeline_failed", err

  metrics = pipeline_metrics or {}

  if skip_validation:
    print("[ogn-run] validation: skip=true", file=sys.stderr, flush=True)
    metrics["validation_status"] = "skipped"
    metrics["validation_detail"] = {"reason": "skip_validation"}
    metrics_path = run_ctx.run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_status(run_ctx.run_dir, "succeeded", None)
    print("Validation skipped by user request; pipeline completed.")
    return 0, metrics, "skipped", {"reason": "skip_validation"}

  print("[ogn-run] validation: skip=false (running GIAB)", file=sys.stderr, flush=True)

  if not helper_available:
    if require_validation:
      error = {
          "title": "GIAB validation helper missing",
          "hint": "Install ogn-cli with validation tooling or rebuild the demo image.",
          "log_path": None,
      }
      _write_status(run_ctx.run_dir, "failed", error)
      print("Run failed (validation required)")
      print(f"Profile {run_ctx.profile}")
      print(f"Sample {run_ctx.sample}")
      print(
          "Error: GIAB validation helper missing\n"
          "Hint: install validation tools or set OGN_REQUIRE_GIAB_VALIDATION=0 to skip."
      )
      return 1, None, "missing_helper", error

    # No helper: treat as soft skip.
    metrics["validation_status"] = "missing_helper"
    metrics["validation_detail"] = {"reason": "helper_missing"}
    metrics_path = run_ctx.run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_status(run_ctx.run_dir, "succeeded", None)
    print("Validation helper missing; pipeline completed. Skipping GIAB.")
    return 0, metrics, "missing_helper", {"reason": "helper_missing"}

  runner = find_binary(["ogn_variant_runner"])
  if not runner:
    raise RuntimeError("ogn_variant_runner binary not found (build the project first)")

  giab_dir = run_ctx.run_dir / "giab"
  giab_dir.mkdir(parents=True, exist_ok=True)

  sample_id = "HG002"
  ref = bundle.reference_fasta
  truth_vcf = bundle.truth_vcf
  truth_bed = bundle.truth_bed

  cmd = [
      sys.executable or "python3",
      "-m",
      "ogn_cli.tools.run_giab_validation",
      "--reference",
      str(ref),
      "--sample",
      sample_id,
      "--panel",
      "HG002_wgs",
      "--platform",
      "illumina",
      "--coverage-estimate",
      "35x",
      "--reference-build",
      "GRCh38",
      "--truth-vcf-version",
      "v4.2.1",
      "--truth-bed-version",
      "v4.2.1",
      "--source-bucket",
      "",
      "--output",
      str(giab_dir),
      "--runner",
      str(runner),
      "--gpu-count",
      str(max(gpu_count, 0)),
      "--gpu-hour-cost",
      "0.0",
      "--hap-py",
      "hap.py",
  ]

  print("stage=pipeline name=giab_variant_runner start")
  if truth_vcf and truth_bed:
    cmd.extend(
        [
            "--truth-vcf",
            str(truth_vcf),
            "--truth-bed",
            str(truth_bed),
            "--hap-py-reference",
            str(ref),
        ]
    )

  if bundle.cram and bundle.cram.is_file():
    cmd.extend(["--cram", str(bundle.cram)])
  elif bundle.bam and bundle.bam.is_file():
    cmd.extend(["--cram", str(bundle.bam)])
  else:
    raise RuntimeError("No CRAM or BAM found for HG002 bundle")

  giab_log = giab_dir / f"{sample_id}_giab.log"
  with giab_log.open("w", encoding="utf-8") as log_f:
    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        cwd=giab_dir,
    )
    rc = proc.wait()
  print(f"stage=pipeline name=giab_variant_runner done exit_code={rc}")

  # Summarize GIAB log (best effort).
  summary_line = "unknown error"
  try:
    lines = giab_log.read_text(encoding="utf-8").splitlines()
    for line in reversed(lines):
      if line.strip():
        summary_line = line.strip()
        break
  except Exception:
    summary_line = "unable to read GIAB log"

  if rc != 0:
    validation_detail = {
        "exit_code": rc,
        "summary": summary_line,
        "log_path": str(giab_log.relative_to(run_ctx.run_dir)),
    }
    error = {
        "title": "GIAB validation failed",
        "hint": summary_line,
        "log_path": validation_detail["log_path"],
    }
    metrics["validation_status"] = "validation_failed"
    metrics["validation_detail"] = validation_detail
    metrics["giab_metrics_path"] = str(giab_log.relative_to(run_ctx.run_dir))
    # Preserve pipeline runtime fields already present in metrics.
    metrics_path = run_ctx.run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if require_validation:
      _write_status(run_ctx.run_dir, "failed", error)
      print("Run failed (validation required)")
      print(f"Profile {run_ctx.profile}")
      print(f"Sample {run_ctx.sample}")
      log_display = error["log_path"] or "(no log)"
      print(
          f"Error: {error['title']}\nHint: {error['hint']}\nDetails: {log_display}"
      )
      return rc or 1, metrics, "validation_failed", validation_detail

    print("Run complete (pipeline OK, GIAB validation failed; not required)")
    print(
        f"Validation: status=failed, exit_code={rc}, summary=\"{summary_line}\""
    )
    print(f"GIAB log: {validation_detail['log_path']}")
    _write_status(run_ctx.run_dir, "succeeded", None)
    return 0, metrics, "validation_failed", validation_detail

  giab_metrics_path = giab_dir / f"{sample_id}_metrics.json"
  if not giab_metrics_path.is_file():
    raise RuntimeError(f"GIAB metrics JSON missing at {giab_metrics_path}")

  giab_metrics = json.loads(giab_metrics_path.read_text(encoding="utf-8"))
  hap_summary_str = giab_metrics.get("hap_py_summary")
  if hap_summary_str:
    candidate = Path(hap_summary_str)
    hap_summary_path = (
        candidate if candidate.is_absolute() else giab_dir / candidate
    )
  else:
    hap_summary_path = giab_dir / f"{sample_id}_hap_py.summary.csv"

  accuracy = _extract_accuracy_from_hap_summary(hap_summary_path, giab_metrics)

  hap_summary_rel: Optional[Path] = None
  if hap_summary_path and hap_summary_path.is_file():
    try:
      hap_summary_rel = hap_summary_path.relative_to(run_ctx.run_dir)
    except ValueError:
      hap_summary_rel = Path("giab") / hap_summary_path.name

  # Copy or link the gVCF into the top-level results/ directory.
  gvcf_src = giab_dir / f"{sample_id}.g.vcf.gz"
  vcf_rel = None
  if gvcf_src.is_file():
    vcf_dest = run_ctx.run_dir / "results" / gvcf_src.name
    try:
      os.link(gvcf_src, vcf_dest)
    except OSError:
      import shutil

      shutil.copy2(gvcf_src, vcf_dest)
    vcf_rel = Path("results") / gvcf_src.name

  metrics["validation_status"] = "success"
  metrics["validation_detail"] = {"exit_code": 0, "summary": "ok"}
  metrics["giab_metrics_path"] = str(giab_metrics_path.relative_to(run_ctx.run_dir))
  metrics["hap_py_summary_path"] = (
      str(hap_summary_rel) if hap_summary_rel else None
  )
  metrics["hap_py_runtime_seconds"] = giab_metrics.get("hap_py_runtime_seconds")
  metrics["coverage_estimate"] = giab_metrics.get("coverage_estimate")
  metrics["accuracy"] = accuracy
  if vcf_rel is not None:
    metrics["vcf_path"] = str(vcf_rel)
  metrics_path = run_ctx.run_dir / "metrics.json"
  metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

  _write_status(run_ctx.run_dir, "succeeded", None)

  return 0, metrics, "success", {"exit_code": 0, "summary": "ok"}


def _run_simple_variant(
    run_ctx: RunContext, simple: SimpleInputs, gpu_count: int
) -> Tuple[int, Optional[Dict[str, Any]]]:
  runner = find_binary(["ogn_variant_runner"])
  if not runner:
    raise RuntimeError(
        "ogn_variant_runner binary not found (build the project first)"
    )

  logs_dir = run_ctx.run_dir / "logs"
  results_dir = run_ctx.run_dir / "results"
  logs_dir.mkdir(parents=True, exist_ok=True)
  results_dir.mkdir(parents=True, exist_ok=True)

  sample = simple.sample
  ref = simple.reference_fasta

  if simple.cram and simple.cram.is_file():
    input_path = simple.cram
  else:
    input_path = simple.bam

  vcf_name = f"{sample}.g.vcf.gz"
  vcf_path = results_dir / vcf_name

  cmd = [
      str(runner),
      "--reference",
      str(ref),
      "--sample",
      sample,
      "--streams-per-gpu",
      "1",
  ]
  # ogn_variant_runner expects CRAM-style input; BAM is accepted via the same flag.
  cmd.extend(["--cram", str(input_path)])

  engine_log = logs_dir / "engine.log"

  import time

  start = time.time()
  with engine_log.open("w", encoding="utf-8") as log_handle:
    proc = subprocess.Popen(
        cmd,
        cwd=results_dir,
        stdout=log_handle,
        stderr=log_handle,
    )
    rc = proc.wait()
  wall = time.time() - start

  if rc != 0:
    error = {
        "title": "Variant calling failed",
        "hint": "Inspect the engine log for details.",
        "log_path": str(engine_log.relative_to(run_ctx.run_dir)),
    }
    return rc or 1, error

  rel_vcf = vcf_path.relative_to(run_ctx.run_dir)
  metrics = {
      "sample": sample,
      "profile": run_ctx.profile,
      "wall_time_seconds": wall,
      "reference_build": "GRCh38",
      "vcf_path": str(rel_vcf),
      "bam_or_cram": str(input_path),
      "gpu": {
          "device": run_ctx.gpu_name,
          "count": gpu_count,
      },
      "accuracy": None,
  }

  return 0, metrics


def run_run(args: argparse.Namespace) -> int:
  # Earliest breadcrumb to verify code path.
  print("[ogn-run] boot: entered main", file=sys.stderr, flush=True)
  paths = resolve_paths(data_root=args.data_root)
  gpu = detect_gpu()

  run_dir = _derive_run_dir(paths, args.profile, args.sample)
  run_ctx = RunContext(
      profile=args.profile,
      sample=args.sample,
      data_root=paths.data_root,
      runs_root=paths.runs_root,
      run_dir=run_dir,
      gpu_name=gpu.name if gpu.present else None,
  )

  # Determine run mode.
  hg002_summary_json = None
  run_mode = "simple"
  if args.profile == "illumina_wgs" and args.sample.lower() == "hg002":
    hg002_summary_json = _load_hg002_summary(paths.data_root)
    if not hg002_summary_json:
      print(
          f"Error: HG002 bundle summary missing under {paths.data_root}/profiles "
          '(run "ogn setup hg002" first)',
          file=sys.stderr,
      )
      return 1
    run_mode = "giab"
  require_giab = bool(
      getattr(args, "require_giab_validation", False)
      or str(os.environ.get("OGN_REQUIRE_GIAB_VALIDATION", "0")).lower()
      in ("1", "true", "yes")
  )
  skip_validation = bool(
      getattr(args, "skip_validation", False)
      or str(os.environ.get("OGN_SKIP_VALIDATION", "0")).lower() in ("1", "true", "yes")
  )
  watchdog_seconds = int(getattr(args, "watchdog_seconds", 120) or 120)
  debug_plan = os.environ.get("OGN_DEBUG_PLAN", "0").lower() in ("1", "true", "yes")
  print("[ogn-run] args parsed", file=sys.stderr, flush=True)

  # Early breadcrumbs.
  _write_initial_status_and_metrics(
      run_ctx, gpu.present, gpu.name, gpu.count
  )

  # Plan print must happen before heavy work.
  print("Resolved stages:", file=sys.stderr, flush=True)
  print("  1) mapper", file=sys.stderr, flush=True)
  print("  2) pairhmm", file=sys.stderr, flush=True)
  print("  3) deep_learning", file=sys.stderr, flush=True)
  print("  4) genotyper", file=sys.stderr, flush=True)
  print("  5) emit", file=sys.stderr, flush=True)
  post_label = "giab_validation (skipped)" if skip_validation else "giab_validation"
  print("Post steps:", file=sys.stderr, flush=True)
  print(f"  - {post_label}", file=sys.stderr, flush=True)

  def _start_watchdog(label: str, timeout: int):
    stop_event = threading.Event()
    def _heartbeat():
      start_t = time.time()
      while not stop_event.wait(timeout=5):
        elapsed = time.time() - start_t
        print(
            f"[ogn-run] heartbeat phase={label} elapsed={int(elapsed)}s run_dir={run_ctx.run_dir}",
            file=sys.stderr,
            flush=True,
        )
        if elapsed >= timeout and not stop_event.is_set():
          print(
              f"[ogn-run] suspected hang phase={label} elapsed={int(elapsed)}s run_dir={run_ctx.run_dir}",
              file=sys.stderr,
              flush=True,
          )
          break
    threading.Thread(target=_heartbeat, daemon=True).start()
    return stop_event

  current_watchdog = _start_watchdog("planning", watchdog_seconds)

  def set_phase(label: Optional[str]):
    nonlocal current_watchdog
    if current_watchdog:
      current_watchdog.set()
    if label:
      current_watchdog = _start_watchdog(label, watchdog_seconds)
    else:
      current_watchdog = None

  def log_step(name: str, state: str):
    print(f"[ogn-run] planning:{name} {state}", file=sys.stderr, flush=True)

  log_step("resolve_paths", "start")
  log_step("resolve_paths", "done")

  log_step("resolve_profile", "start")
  print(f"Run directory: {run_ctx.run_dir}")
  print("Plan")
  print(f"Profile {run_ctx.profile}")
  print(f"Sample {run_ctx.sample}")
  print(f"Data root {run_ctx.data_root}")
  print(f"Output {run_ctx.run_dir}")
  if not gpu.present:
    print("WARN GPU missing, running in CPU-only mode")
  log_step("resolve_profile", "done")

  if args.explain == "yes":
    print("Stage: variant calling (identifying SNVs and indels)")
  if debug_plan or args.profile == "illumina_wgs":
    print("Resolved stages:")
    print("  1) mapper")
    print("  2) pairhmm")
    print("  3) deep_learning")
    print("  4) genotyper")
    print("  5) emit")
    print("Post steps:")
    if skip_validation:
      print("  - giab_validation (skipped)")
    else:
      print("  - giab_validation")

  # Marker right after stage list.
  print("[ogn-run] planning:enter", file=sys.stderr, flush=True)

  # Write plan.json early.
  log_step("write_plan_json", "start")
  plan_payload = {
      "profile": run_ctx.profile,
      "sample": run_ctx.sample,
      "data_root": str(run_ctx.data_root),
      "run_dir": str(run_ctx.run_dir),
      "stages": ["mapper", "pairhmm", "deep_learning", "genotyper", "emit"],
      "post_steps": [] if skip_validation else ["giab_validation"],
      "validation_skipped": skip_validation,
      "gpu": {"device": gpu.name if gpu.present else None, "count": gpu.count},
      "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
  }
  # Also persist a planning trace so we have breadcrumbs even if we hang later.
  (run_ctx.run_dir / "planning_trace.json").write_text(
      json.dumps(plan_payload, indent=2), encoding="utf-8"
  )
  (run_ctx.run_dir / "plan.json").write_text(
      json.dumps(plan_payload, indent=2), encoding="utf-8"
  )
  log_step("write_plan_json", "done")

  if run_mode == "giab":
    try:
      log_step("verify_inputs", "start")
      bundle = _bundle_from_summary(paths.data_root, hg002_summary_json or {})
      log_step("verify_inputs", "done")
      rc, metrics = 0, None
      rc, metrics, validation_status, validation_detail = _run_hg002_giab(
          run_ctx,
          bundle,
          gpu.count,
          require_validation=require_giab,
          skip_validation=skip_validation,
          set_phase=set_phase,
      )
    except Exception as ex:
      msg = str(ex) or "GIAB validation failed"
      bundle_root = paths.data_root / "hg002"
      title = msg
      if "requires CRAM input" in msg:
        hint = (
            "GIAB mode for HG002 expects a CRAM input under "
            f"{bundle_root}. Convert BAM to CRAM once with "
            '"samtools view -C -T GRCh38_noalt.fa -o HG002_wgs.cram HG002_wgs.bam" '
            'and rerun "ogn setup hg002".'
        )
      elif "No CRAM or BAM" in msg:
        hint = (
            "GIAB mode for HG002 expects CRAM or BAM under "
            f"{bundle_root}. Ensure the bundle was staged correctly or rerun "
            '"ogn setup hg002".'
        )
      else:
        title = "GIAB validation failed"
        hint = msg
      error = {
          "title": title,
          "hint": hint,
          "log_path": None,
      }
      _write_status(run_ctx.run_dir, "failed", error)
      print("Run failed")
      print(f"Profile {run_ctx.profile}")
      print(f"Sample {run_ctx.sample}")
      log_display = error["log_path"] or "(no log)"
      print(
          f"Error: {error['title']}\nHint: {error['hint']}\nDetails: {log_display}"
      )
      return 1

    if rc != 0:
      return rc
    if validation_status != "success":
      detail = validation_detail if isinstance(validation_detail, dict) else {}
      summary = detail.get("summary", validation_status)
      exit_code = detail.get("exit_code", "n/a")
      print(
          f"Validation status={validation_status} exit_code={exit_code} summary=\"{summary}\""
      )

    print("Run complete")
    print(f"Profile {run_ctx.profile}")
    print(f"Sample {run_ctx.sample}")
    if gpu.present:
      print(f"GPU {gpu.name}")
    print(f"Outputs {run_ctx.run_dir}/results")

    if args.json and metrics is not None:
      print_json(metrics)

    return 0

  # Simple mode for non-HG002 samples.
  try:
    simple_inputs = _resolve_simple_inputs(paths.data_root, args.sample)
  except Exception as ex:
    msg = str(ex) or "Pre-flight validation failed"
    sample_dir = paths.data_root / args.sample
    if "No CRAM or BAM" in msg:
      title = msg
      hint = (
          f"No CRAM or BAM found for sample {args.sample} under {sample_dir}. "
          f"Expected {args.sample}.cram or {args.sample}.bam."
      )
    elif "Reference FASTA or index not found" in msg:
      title = "Reference FASTA not found"
      ref_dir = paths.data_root / "reference"
      hint = (
          f"Expected GRCh38_noalt.fa and GRCh38_noalt.fa.fai under {ref_dir}. "
          "Stage the reference there or configure a different data root."
      )
    else:
      title = "Pre-flight validation failed"
      hint = msg
    error = {
        "title": title,
        "hint": hint,
        "log_path": None,
    }
    _write_status(run_ctx.run_dir, "failed", error)
    print("Run failed")
    print(f"Profile {run_ctx.profile}")
    print(f"Sample {run_ctx.sample}")
    print(
        f"Error: {error['title']}\nHint: {error['hint']}\nDetails: (no log)"
    )
    return 1

  try:
    rc, result = _run_simple_variant(run_ctx, simple_inputs, gpu.count)
  except Exception as ex:
    msg = str(ex) or "Variant calling failed"
    error = {
        "title": "Variant calling failed",
        "hint": msg,
        "log_path": None,
    }
    _write_status(run_ctx.run_dir, "failed", error)
    print("Run failed")
    print(f"Profile {run_ctx.profile}")
    print(f"Sample {run_ctx.sample}")
    print(
        f"Error: {error['title']}\nHint: {error['hint']}\nDetails: (no log)"
    )
    return 1

  if rc != 0:
    err = result or {}
    error = {
        "title": err.get("title", "Variant calling failed"),
        "hint": err.get("hint", "Inspect the engine log for details."),
        "log_path": err.get("log_path"),
    }
    _write_status(run_ctx.run_dir, "failed", error)
    log_display = error["log_path"] or "(no log)"
    print("Run failed")
    print(f"Profile {run_ctx.profile}")
    print(f"Sample {run_ctx.sample}")
    print(
        f"Error: {error['title']}\nHint: {error['hint']}\nDetails: {log_display}"
    )
    return rc or 1

  metrics = result or {}
  metrics_path = run_ctx.run_dir / "metrics.json"
  metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
  _write_status(run_ctx.run_dir, "succeeded", None)

  print("Run complete")
  print(f"Profile {run_ctx.profile}")
  print(f"Sample {run_ctx.sample}")
  if gpu.present:
    print(f"GPU {gpu.name}")
  print(f"Outputs {run_ctx.run_dir}/results")

  if args.json and metrics is not None:
    print_json(metrics)

  return 0
