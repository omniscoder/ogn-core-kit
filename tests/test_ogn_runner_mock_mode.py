from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ogn_runner import receipt as receipt_lib


def _runner_cmd() -> list[str]:
    return [sys.executable, "-m", "ogn_runner"]


def _mock_spec(tmp_path: Path) -> Path:
    job = {
        "schema_version": "v1",
        "run_id": "run-mock-001",
        "tenant_id": "tenant-mock",
        "engine": {
            "version": "demo",
            "profile": "demo-tiny",
            "parameters": {"cpu_only": True},
        },
        "inputs": {
            "fastq_uris": ["file:///tmp/unused.fastq.gz"],
            "reference_uri": "file:///tmp/unused.fa",
            "extra": {},
        },
        "outputs": {
            "vcf": {"uri": str(tmp_path / "out.vcf.gz")},
            "provenance": {"uri": str(tmp_path / "provenance.json")},
            "logs": {"uri": str(tmp_path / "logs.jsonl")},
            "receipt": {"uri": str(tmp_path / "receipt.json")},
        },
    }
    path = tmp_path / "job_spec.json"
    path.write_text(json.dumps(job), encoding="utf-8")
    return path


def test_ogn_runner_validate(tmp_path: Path) -> None:
    spec = _mock_spec(tmp_path)
    proc = subprocess.run(_runner_cmd() + ["--validate", str(spec)], capture_output=True, text=True)
    assert proc.returncode == 0
    assert proc.stdout.strip() == "VALID Job Spec v1"


def test_ogn_runner_validate_invalid(tmp_path: Path) -> None:
    spec = tmp_path / "job_spec.json"
    spec.write_text("{}", encoding="utf-8")
    proc = subprocess.run(_runner_cmd() + ["--validate", str(spec)], capture_output=True, text=True)
    assert proc.returncode == 2
    assert "job spec invalid" in proc.stderr


def test_ogn_runner_mock_mode(tmp_path: Path) -> None:
    spec = _mock_spec(tmp_path)
    proc = subprocess.run(_runner_cmd() + ["--mock", str(spec)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    vcf_path = tmp_path / "out.vcf.gz"
    prov_path = tmp_path / "provenance.json"
    logs_path = tmp_path / "logs.jsonl"
    receipt_path = tmp_path / "receipt.json"

    assert vcf_path.exists()
    assert prov_path.exists()
    assert logs_path.exists()
    assert receipt_path.exists()

    vcf_contents = vcf_path.read_text(encoding="utf-8").splitlines()
    assert vcf_contents[0] == "##fileformat=VCFv4.3"
    assert "\t".join(vcf_contents[2].split("\t")[:4]) == "chr20\t1\t.\tA"

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == "ogn.receipt.v1"
    assert receipt["run_id"] == "run-mock-001"
    assert len(receipt["artifacts"]) >= 3
    assert receipt_lib.validate_receipt_payload_hash(receipt)
