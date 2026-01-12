from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingTCPServer
from typing import Any


class _PutHandler(SimpleHTTPRequestHandler):
    upload_root: Path

    def do_PUT(self) -> None:  # noqa: N802 (http handler naming)
        dest = self.upload_root / self.path.lstrip("/")
        dest.parent.mkdir(parents=True, exist_ok=True)

        length = int(self.headers.get("Content-Length", "0"))
        remaining = length
        with dest.open("wb") as out:
            while remaining > 0:
                chunk = self.rfile.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                out.write(chunk)
                remaining -= len(chunk)

        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Keep pytest output clean.
        return


def _write_fake_ogn_run(bin_dir: Path) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    path = bin_dir / "ogn_run"
    path.write_text(
        """#!/usr/bin/env python3
import os
import sys

def main(argv):
    if "--version" in argv:
        sys.stdout.write("ogn_run version 0.0.0-test (git deadbeef)\\n")
        return 0

    # Minimal arg parse: find --vcf <path>.
    out = None
    for i, a in enumerate(argv):
        if a == "--vcf" and i + 1 < len(argv):
            out = argv[i + 1]
            break
    if not out:
        sys.stderr.write("missing --vcf\\n")
        return 2

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(b"##fileformat=VCFv4.3\\n")
        f.write(b"#CHROM\\tPOS\\tID\\tREF\\tALT\\tQUAL\\tFILTER\\tINFO\\tFORMAT\\tsample\\n")
        f.write(b"chr20\\t1\\t.\\tA\\tC\\t.\\tPASS\\t.\\tGT\\t0/1\\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _runner_cmd() -> list[str]:
    return [sys.executable, "-m", "ogn_runner"]


def _env_with_sdk(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    sdk_python = repo_root / "sdk" / "python"
    env["PYTHONPATH"] = str(sdk_python) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_ogn_runner_job_spec_v1_uploads_provenance(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    fake_bin = tmp_path / "fake-bin"
    _write_fake_ogn_run(fake_bin)

    # Local reference + .fai so the runner doesn't need samtools in this unit test.
    ref = tmp_path / "ref.fa"
    ref.write_text(">chr20\nACGT\n", encoding="utf-8")
    (tmp_path / "ref.fa.fai").write_text("chr20\t4\t6\t4\t5\n", encoding="utf-8")

    # Inputs served over HTTP to exercise downloader code.
    server_root = tmp_path / "http"
    inputs_dir = server_root / "inputs"
    inputs_dir.mkdir(parents=True)
    fastq = inputs_dir / "reads.fastq"
    fastq.write_text("@r1\nACGT\n+\n!!!!\n", encoding="utf-8")

    uploads_dir = tmp_path / "uploads"
    handler_cls = _PutHandler
    handler_cls.upload_root = uploads_dir  # type: ignore[attr-defined]

    httpd = ThreadingTCPServer(
        ("127.0.0.1", 0),
        lambda *a, **k: handler_cls(*a, directory=str(server_root), **k),
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = httpd.server_address
        fastq_url = f"http://{host}:{port}/inputs/reads.fastq"
        put_base = f"http://{host}:{port}/upload"

        job = {
            "schema_version": "v1",
            "run_id": "run_test_001",
            "tenant_id": "tenant_test",
            "submitted_at": "2025-01-01T00:00:00Z",
            "engine": {"version": "0.0.0-test", "profile": "demo-tiny", "parameters": {"cpu_only": True}},
            "inputs": {"fastq_uris": [fastq_url], "reference_uri": str(ref), "extra": {}},
            "outputs": {
                "vcf": {
                    "uri": "s3://bucket/tenants/tenant_test/runs/run_test_001/out.vcf.gz",
                    "put_url": f"{put_base}/out.vcf.gz",
                },
                "provenance": {
                    "uri": "s3://bucket/tenants/tenant_test/runs/run_test_001/provenance.json",
                    "put_url": f"{put_base}/provenance.json",
                },
                "logs": {
                    "uri": "s3://bucket/tenants/tenant_test/runs/run_test_001/logs.jsonl",
                    "put_url": f"{put_base}/logs.jsonl",
                },
            },
        }
        spec_path = tmp_path / "job_spec.json"
        spec_path.write_text(json.dumps(job), encoding="utf-8")

        env = _env_with_sdk(repo_root)
        env["PATH"] = str(fake_bin) + os.pathsep + env.get("PATH", "")
        proc = subprocess.run(_runner_cmd() + [str(spec_path)], env=env, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr

        prov_path = uploads_dir / "upload" / "provenance.json"
        vcf_path = uploads_dir / "upload" / "out.vcf.gz"
        logs_path = uploads_dir / "upload" / "logs.jsonl"
        assert prov_path.exists()
        assert vcf_path.exists()
        assert logs_path.exists()

        prov = json.loads(prov_path.read_text(encoding="utf-8"))
        assert prov["schema"] == "ogn_provenance@1"
        assert prov["schema_version"] == "v1"
        assert prov["run_id"] == "run_test_001"
        assert prov["tenant_id"] == "tenant_test"
        assert prov["engine_version"] == "0.0.0-test"
        assert prov["engine_git_sha"] == "deadbeef"
        assert prov["engine"]["profile"] == "demo-tiny"
        assert prov["engine"]["resolved_version"] == "0.0.0-test"
        assert prov["inputs"]["fastq_uris"] == [fastq_url]
        assert prov["outputs"]["provenance"].endswith("/provenance.json")
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_ogn_runner_job_spec_v1_put_url_optional(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    fake_bin = tmp_path / "fake-bin"
    _write_fake_ogn_run(fake_bin)

    # Local reference + .fai so the runner doesn't need samtools in this unit test.
    ref = tmp_path / "ref.fa"
    ref.write_text(">chr20\nACGT\n", encoding="utf-8")
    (tmp_path / "ref.fa.fai").write_text("chr20\t4\t6\t4\t5\n", encoding="utf-8")

    server_root = tmp_path / "http"
    inputs_dir = server_root / "inputs"
    inputs_dir.mkdir(parents=True)
    fastq = inputs_dir / "reads.fastq"
    fastq.write_text("@r1\nACGT\n+\n!!!!\n", encoding="utf-8")

    # Upload destinations are stable local file URIs (no presigned PUT URLs).
    stable_dir = tmp_path / "stable"
    stable_dir.mkdir(parents=True)
    vcf_uri = str(stable_dir / "out.vcf.gz")
    prov_uri = str(stable_dir / "provenance.json")

    handler_cls = _PutHandler
    handler_cls.upload_root = tmp_path / "unused-uploads"  # type: ignore[attr-defined]

    httpd = ThreadingTCPServer(
        ("127.0.0.1", 0),
        lambda *a, **k: handler_cls(*a, directory=str(server_root), **k),
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = httpd.server_address
        fastq_url = f"http://{host}:{port}/inputs/reads.fastq"

        job = {
            "schema_version": "v1",
            "run_id": "run_test_002",
            "tenant_id": "tenant_test",
            # NOTE: submitted_at is optional by contract (and must not be required by the runner).
            "engine": {"version": "0.0.0-test", "profile": "demo-tiny", "parameters": {"cpu_only": True}},
            "inputs": {"fastq_uris": [fastq_url], "reference_uri": str(ref), "extra": {}},
            "outputs": {
                "vcf": {"uri": vcf_uri},
                "provenance": {"uri": prov_uri},
            },
        }
        spec_path = tmp_path / "job_spec.json"
        spec_path.write_text(json.dumps(job), encoding="utf-8")

        env = _env_with_sdk(repo_root)
        env["PATH"] = str(fake_bin) + os.pathsep + env.get("PATH", "")
        proc = subprocess.run(_runner_cmd() + [str(spec_path)], env=env, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr

        vcf_path = stable_dir / "out.vcf.gz"
        prov_path = stable_dir / "provenance.json"
        assert vcf_path.exists()
        assert prov_path.exists()

        prov = json.loads(prov_path.read_text(encoding="utf-8"))
        assert prov["schema"] == "ogn_provenance@1"
        assert prov["schema_version"] == "v1"
        assert prov["run_id"] == "run_test_002"
        assert prov["tenant_id"] == "tenant_test"
        assert "submitted_at" not in prov
    finally:
        httpd.shutdown()
        httpd.server_close()

