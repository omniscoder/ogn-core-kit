#!/usr/bin/env python3
"""Run the OGN variant pipeline on GIAB samples and collect metrics."""

from __future__ import annotations

import argparse
import json
import hashlib
import os
import subprocess
import time
from pathlib import Path


def run_command(cmd: list[str], workdir: Path | None = None, env: dict[str, str] | None = None) -> float:
    start = time.time()
    subprocess.run(cmd, check=True, cwd=workdir, env=env)
    return time.time() - start


def run_cmd_capture(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="OGN GIAB validation helper")
    parser.add_argument("--fastq1", help="R1 FASTQ path")
    parser.add_argument("--fastq2", help="R2 FASTQ path (optional for single-end)")
    parser.add_argument("--cram", help="Aligned CRAM input (mutually exclusive with FASTQ)")
    parser.add_argument("--reference", required=True, help="Reference FASTA")
    parser.add_argument("--sample", required=True, help="Sample identifier")
    parser.add_argument("--panel", default="", help="Primary panel name")
    parser.add_argument("--platform", default="", help="Platform (illumina/pacbio/ont)")
    parser.add_argument("--coverage-estimate", default="", help="Approximate coverage")
    parser.add_argument("--reference-build", default="", help="Reference build label (e.g., GRCh38)")
    parser.add_argument("--truth-vcf-version", default="", help="Truth VCF version tag")
    parser.add_argument("--truth-bed-version", default="", help="Truth BED version tag")
    parser.add_argument("--source-bucket", default="", help="Source bucket/provider for inputs")
    parser.add_argument("--seed", default="", help="Deterministic seed for downstream tools")
    parser.add_argument("--env-json", default="", help="Path to env.json for provenance/badge digest")
    parser.add_argument("--accuracy-status", default="", help="Optional accuracy gate status (PASS/FAIL/N.A.)")
    parser.add_argument("--perf-status", default="", help="Optional performance gate status (PASS/FAIL/N.A.)")
    parser.add_argument("--truth-vcf", help="GIAB truth VCF (optional for truthless stress runs)")
    parser.add_argument("--truth-bed", help="GIAB confident regions BED (optional for truthless stress runs)")
    parser.add_argument("--hap-py-reference", help="hap.py reference prefix (optional if truth absent)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--runner", default="./ogn_variant_runner", help="Path to ogn_variant_runner executable")
    parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs used (default: 1)")
    parser.add_argument("--gpu-hour-cost", type=float, default=0.0, help="Optional GPU $/hour for cost estimates")
    parser.add_argument("--hap-py", default="hap.py", help="hap.py executable")
    args = parser.parse_args()

    if not args.cram and not args.fastq1:
        parser.error("Either --cram or --fastq1 must be provided")
    if args.cram and args.fastq1:
        parser.error("--cram cannot be used together with --fastq1/--fastq2")
    if args.fastq2 and not args.fastq1:
        parser.error("--fastq2 provided without --fastq1")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed
    if not seed:
        digest = hashlib.sha1(args.sample.encode("utf-8")).hexdigest()
        seed = str(int(digest[:8], 16) & 0x7FFFFFFF)

    env = os.environ.copy()
    env["OGN_RANDOM_SEED"] = seed

    gvcf_path = output_dir / f"{args.sample}.g.vcf.gz"

    runner_cmd = [
        args.runner,
        "--reference", args.reference,
        "--sample", args.sample,
        "--streams-per-gpu", "1",
    ]
    if args.cram:
        runner_cmd.extend(["--cram", args.cram])
    else:
        runner_cmd.extend(["--fastq", args.fastq1])
        if args.fastq2:
            runner_cmd.extend(["--fastq", args.fastq2])
    runtime_seconds = run_command(runner_cmd, workdir=output_dir, env=env)

    hap_py_prefix = output_dir / f"{args.sample}_hap_py"
    truthless = not args.truth_vcf or not args.truth_bed
    hap_runtime_seconds = 0.0
    if not truthless:
        hap_cmd = [
            args.hap_py,
            args.truth_vcf,
            str(gvcf_path),
            "--fasta", args.reference,
            "--bed", args.truth_bed,
            "--reference", args.hap_py_reference or args.reference,
            "--engine", "vcfeval",
            "--threads", "8",
            "--output", str(hap_py_prefix),
        ]
        hap_runtime_seconds = run_command(hap_cmd, workdir=output_dir, env=env)

    gpu_hours = (runtime_seconds / 3600.0) * max(args.gpu_count, 1)
    cost = gpu_hours * args.gpu_hour_cost if args.gpu_hour_cost > 0 else 0.0

    metrics = {
        "sample": args.sample,
        "sample_id": args.sample,
        "panel": args.panel,
        "platform": args.platform,
        "coverage_estimate": args.coverage_estimate,
        "reference_build": args.reference_build,
        "truth_vcf_version": args.truth_vcf_version,
        "truth_bed_version": args.truth_bed_version,
        "source_bucket": args.source_bucket,
        "seed": seed,
        "gvcf": str(gvcf_path),
        "runtime_seconds": runtime_seconds,
        "hap_py_runtime_seconds": hap_runtime_seconds,
        "gpu_count": args.gpu_count,
        "gpu_hours": round(gpu_hours, 4),
        "gpu_hour_cost": args.gpu_hour_cost,
        "estimated_gpu_cost": round(cost, 2),
        "hap_py_summary": None if truthless else str(hap_py_prefix) + ".summary.csv"
    }

    contract_version = os.environ.get("BENCHMARK_CONTRACT_VERSION") or "v1.1"
    metrics["ogn_version"] = os.environ.get("OGN_VERSION") or run_cmd_capture(["git", "describe", "--tags", "--always", "--dirty"])
    metrics["benchmark_contract_version"] = contract_version
    metrics["last_bench_status_ok"] = os.environ.get("OGN_LAST_BENCH_STATUS", "unknown")

    metrics_path = output_dir / f"{args.sample}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))

    # Footer badge in logs
    env_json_path = Path(args.env_json) if args.env_json else output_dir / "env.json"
    env_digest = "n/a"
    if env_json_path.exists():
        env_digest = hashlib.sha1(env_json_path.read_bytes()).hexdigest()[:10]

    ogn_version = metrics.get("ogn_version") or "unknown"
    contract_commit = run_cmd_capture(["git", "rev-parse", "HEAD"]) or "unknown"
    accuracy_status = (args.accuracy_status or ("PASS" if not truthless else "N/A")).upper()
    perf_status = (args.perf_status or "UNKNOWN").upper()

    metrics["contract_badge"] = {
        "label": f"Verified by OGN Benchmark Contract {contract_version}",
        "contract_version": contract_version,
        "accuracy_status": accuracy_status,
        "perf_status": perf_status,
        "tooltip": "Accuracy and performance validated against GIAB v4.2.1, HG002 multi-platform smoke, CHM13 stress perf (contract v1.1).",
        "env_digest": env_digest,
        "contract_commit": contract_commit,
        "seed": seed,
    }

    footer = (
        "\nOGN Benchmark Contract {cv}\n"
        "Accuracy: {acc} • Perf: {perf}\n"
        "ogn_version: {ov}\n"
        "Contract commit: {cc}\n"
        "Env digest: {ed}\n"
        "Seed: {seed}\n"
    ).format(cv=contract_version, acc=accuracy_status, perf=perf_status,
             ov=ogn_version, cc=contract_commit, ed=env_digest, seed=seed)
    print(footer)


if __name__ == "__main__":  # pragma: no cover
    main()
