from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import shutil
from pathlib import Path

from ogn_runner.receipt import (
    build_receipt,
    compute_receipt_payload_sha256,
    make_artifact_receipt,
)
from ogn_runner.verify_receipt import verify_receipt


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _runner_path() -> list[str]:
    return [sys.executable, "-m", "ogn_runner.verify_receipt"]


def _mock_examples_spec(tmp_path: Path) -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / "examples" / "minimal-job" / "job_spec.json"
    local_spec = tmp_path / "job_spec.json"
    shutil.copy2(source, local_spec)
    return local_spec, tmp_path


def _write_receipt_with_artifact(tmp_path: Path) -> tuple[Path, Path]:
    artifact_path = tmp_path / "artifact.vcf"
    artifact_contents = "chr20\t1\t.\tA\tC\t.\tPASS\t.\tGT\t0/1\n"
    artifact_path.write_text(artifact_contents, encoding="utf-8")

    artifact = make_artifact_receipt(
        name="artifact",
        path=artifact_path,
        media_type="text/plain",
        optional=False,
    )
    receipt = build_receipt(
        run_id="run-verify-001",
        tenant_id="tenant-verify",
        created_at="2026-01-01T00:00:00+00:00",
        engine={"version": "demo", "profile": "demo-tiny"},
        inputs={"fastq_uris": ["reads.fastq.gz"], "reference_uri": "reference.fa"},
        outputs={"vcf": {"uri": str(artifact_path)}},
        artifacts=[artifact],
        verification={"status": "unsigned"},
    )
    receipt["artifacts"][0]["sha256"] = _sha256_text(artifact_contents)

    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return receipt_path, artifact_path


def test_verify_receipt_valid(tmp_path: Path) -> None:
    receipt_path, artifact_path = _write_receipt_with_artifact(tmp_path)
    proc = subprocess.run(_runner_path() + [str(receipt_path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "receipt valid" in proc.stdout
    assert len(verify_receipt(receipt_path, metadata_only=False)) == 0


def test_verify_receipt_invalid_hash(tmp_path: Path) -> None:
    receipt_path, artifact_path = _write_receipt_with_artifact(tmp_path)
    artifact_path.write_text("mutated", encoding="utf-8")
    proc = subprocess.run(_runner_path() + [str(receipt_path)], capture_output=True, text=True)
    assert proc.returncode == 1
    assert "hash mismatch" in proc.stderr + proc.stdout


def test_verify_receipt_metadata_only(tmp_path: Path) -> None:
    receipt_path, artifact_path = _write_receipt_with_artifact(tmp_path)
    artifact_path.unlink()
    proc = subprocess.run(_runner_path() + ["--metadata-only", str(receipt_path)], capture_output=True, text=True)
    assert proc.returncode == 0


def test_verify_receipt_after_portable_copy(tmp_path: Path) -> None:
    mock_runner = [sys.executable, "-m", "ogn_runner", "--mock"]
    local_spec, mock_workdir = _mock_examples_spec(tmp_path)
    proc = subprocess.run(mock_runner + [str(local_spec)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    receipt_path = mock_workdir / "receipt.json"
    out_vcf = mock_workdir / "out.vcf.gz"
    provenance = mock_workdir / "provenance.json"
    logs = mock_workdir / "logs.jsonl"

    assert receipt_path.exists()
    assert out_vcf.exists()
    assert provenance.exists()
    assert logs.exists()

    portable_dir = tmp_path / "portable-bundle"
    portable_dir.mkdir()
    shutil.copy2(receipt_path, portable_dir / "receipt.json")
    shutil.copy2(out_vcf, portable_dir / "out.vcf.gz")
    shutil.copy2(provenance, portable_dir / "provenance.json")
    shutil.copy2(logs, portable_dir / "logs.jsonl")

    proc = subprocess.run(
        _runner_path() + [str(portable_dir / "receipt.json")],
        capture_output=True,
        text=True,
        cwd="/tmp",
    )
    assert proc.returncode == 0, proc.stderr
    assert "receipt valid" in proc.stdout

    proc = subprocess.run(
        _runner_path()
        + ["--base-dir", str(portable_dir), str(portable_dir / "receipt.json")],
        capture_output=True,
        text=True,
        cwd="/tmp",
    )
    assert proc.returncode == 0, proc.stderr


def test_verify_receipt_rejects_artifact_path_outside_bundle(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    outside = tmp_path / "outside.vcf"
    outside.write_text("external artifact", encoding="utf-8")
    receipt_path, _ = _write_receipt_with_artifact(bundle_dir)

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifacts"][0]["path"] = "../outside.vcf"
    receipt["artifacts"][0]["sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    receipt["hashes"]["receipt_payload_sha256"] = compute_receipt_payload_sha256(
        receipt
    )
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
    )

    messages = verify_receipt(receipt_path)

    assert messages == ["artifact path escapes verification root: ../outside.vcf"]


def test_verify_receipt_rejects_artifact_symlink_outside_bundle(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    outside = tmp_path / "outside.vcf"
    outside.write_text("external artifact", encoding="utf-8")
    receipt_path, artifact_path = _write_receipt_with_artifact(bundle_dir)
    artifact_path.unlink()
    artifact_path.symlink_to(outside)

    messages = verify_receipt(receipt_path)

    assert messages == [f"artifact path escapes verification root: {artifact_path}"]
