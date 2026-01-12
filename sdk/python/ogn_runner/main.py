from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import http.client
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import traceback
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping


def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _expect_mapping(v: Any, path: str) -> Mapping[str, Any]:
    if isinstance(v, Mapping):
        return v
    raise ValueError(f"{path} must be an object")


def _expect_list(v: Any, path: str) -> list[Any]:
    if isinstance(v, list):
        return v
    raise ValueError(f"{path} must be a list")


def _expect_str(v: Any, path: str) -> str:
    if isinstance(v, str):
        s = v.strip()
        if s:
            return s
    raise ValueError(f"{path} must be a non-empty string")


def _expect_optional_str(v: Any, path: str) -> str | None:
    if v is None:
        return None
    return _expect_str(v, path)


@dataclasses.dataclass(frozen=True)
class ArtifactTarget:
    name: str
    uri: str
    put_url: str | None = None
    optional: bool = False
    media_type: str = "application/octet-stream"


def _parse_artifacts(outputs: Any) -> dict[str, ArtifactTarget]:
    out: dict[str, ArtifactTarget] = {}

    if not isinstance(outputs, Mapping):
        raise ValueError("outputs must be an object")

    # Canonical: { "artifacts": [ { "id": "...", "uri": "...", "put_url": "..." } ] }
    if "artifacts" in outputs and isinstance(outputs.get("artifacts"), list):
        items = outputs["artifacts"]
        for i, raw in enumerate(items):
            spec = _expect_mapping(raw, f"outputs.artifacts[{i}]")
            name = _expect_str(spec.get("id"), f"outputs.artifacts[{i}].id")
            uri = _expect_str(
                spec.get("uri") or spec.get("stable_uri") or spec.get("stable"),
                f"outputs.artifacts[{i}].uri",
            )
            put_url = _expect_optional_str(
                spec.get("put_url")
                or spec.get("put")
                or spec.get("presigned_put")
                or spec.get("upload_url"),
                f"outputs.artifacts[{i}].put_url",
            )
            optional = bool(spec.get("optional", False))
            media_type = str(spec.get("media_type") or "application/octet-stream")
            out[name] = ArtifactTarget(
                name=name, uri=uri, put_url=put_url, optional=optional, media_type=media_type
            )
        return out

    # Common: { "vcf": { "uri": "...", "put_url": "..." }, ... }
    if any(isinstance(v, Mapping) for v in outputs.values()):
        for name, raw in outputs.items():
            if not isinstance(raw, Mapping):
                continue
            uri = raw.get("uri") or raw.get("stable_uri") or raw.get("stable")
            put_url = raw.get("put_url") or raw.get("put") or raw.get("upload_url")
            if uri is None and put_url is None:
                continue
            out[name] = ArtifactTarget(
                name=name,
                uri=_expect_str(uri, f"outputs.{name}.uri"),
                put_url=_expect_optional_str(put_url, f"outputs.{name}.put_url"),
                optional=bool(raw.get("optional", False)),
                media_type=str(raw.get("media_type") or "application/octet-stream"),
            )
        if out:
            return out

    # Legacy-ish flat keys: { "vcf_uri": "...", "vcf_put_url": "...", ... }
    tmp: dict[str, dict[str, Any]] = {}
    for key, value in outputs.items():
        if not isinstance(key, str):
            continue
        if key.endswith("_uri"):
            tmp.setdefault(key[: -len("_uri")], {})["uri"] = value
        elif key.endswith("_put_url"):
            tmp.setdefault(key[: -len("_put_url")], {})["put_url"] = value
        elif key.endswith("_put"):
            tmp.setdefault(key[: -len("_put")], {})["put_url"] = value
    for name, spec in tmp.items():
        if "uri" not in spec:
            continue
        out[name] = ArtifactTarget(
            name=name,
            uri=_expect_str(spec["uri"], f"outputs.{name}_uri"),
            put_url=_expect_optional_str(spec.get("put_url"), f"outputs.{name}_put_url"),
        )
    if out:
        return out

    raise ValueError(
        "outputs must contain artifact targets (expected outputs.artifacts[] or mapping values with uri/put_url)"
    )


def _write_log_jsonl_line(fp: Any, stream: str, line: str) -> None:
    rec = {"ts": _utc_now_iso(), "stream": stream, "line": line.rstrip("\n")}
    fp.write(json.dumps(rec, separators=(",", ":")) + "\n")


def _download_http(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=300) as resp:
        if getattr(resp, "status", 200) >= 300:
            raise RuntimeError(f"GET {url} failed with status {resp.status}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as out:
            shutil.copyfileobj(resp, out, length=1024 * 1024)


def _resolve_local_path(raw: str, base_dir: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _materialize_input_uri(uri: str, dest: Path, base_dir: Path) -> Path:
    parsed = urllib.parse.urlparse(uri)
    scheme = (parsed.scheme or "").lower()

    if scheme in {"", "file"}:
        raw_path = parsed.path if scheme == "file" else uri
        path = _resolve_local_path(raw_path, base_dir)
        if not path.exists():
            raise FileNotFoundError(f"local input not found: {path}")
        return path

    if scheme in {"http", "https"}:
        _download_http(uri, dest)
        return dest

    if scheme == "s3":
        aws = shutil.which("aws")
        if not aws:
            raise RuntimeError(
                f"s3:// input requires aws cli (missing) or a presigned https URL: {uri}"
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([aws, "s3", "cp", uri, str(dest)], check=True)
        return dest

    raise RuntimeError(f"unsupported input URI scheme: {scheme} ({uri})")


def _http_put_file(url: str, src: Path, media_type: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise RuntimeError(f"PUT url must be http(s): {url}")
    host = parsed.hostname
    if not host:
        raise RuntimeError(f"PUT url missing host: {url}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    # NOTE: For presigned URLs, the Host header is part of the SigV4 signature.
    # When the URL contains an explicit port (e.g. MinIO on :9000), the Host
    # header must include it or the server will reject the request.
    host_header = parsed.netloc or host
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query

    size = src.stat().st_size
    conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(host, port, timeout=300)
    try:
        # Avoid http.client auto-emitting Host so we can control it precisely
        # for SigV4 presigned URLs (and avoid duplicate Host headers).
        conn.putrequest("PUT", path, skip_host=True)
        conn.putheader("Host", host_header)
        conn.putheader("Content-Type", media_type)
        conn.putheader("Content-Length", str(size))
        conn.endheaders()

        with src.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                conn.send(chunk)

        resp = conn.getresponse()
        body = resp.read()  # consume for keepalive
        if resp.status >= 300:
            snippet = body[:4096].decode("utf-8", errors="replace")
            raise RuntimeError(f"PUT {url} failed: {resp.status} {resp.reason}: {snippet}")
    finally:
        conn.close()


def _upload_to_uri(uri: str, src: Path, base_dir: Path) -> None:
    parsed = urllib.parse.urlparse(uri)
    scheme = (parsed.scheme or "").lower()

    if scheme == "s3":
        aws = shutil.which("aws")
        if not aws:
            raise RuntimeError(f"s3:// output requires aws cli in PATH: {uri}")
        subprocess.run([aws, "s3", "cp", str(src), uri], check=True)
        return

    if scheme in {"", "file"}:
        if scheme == "file" and parsed.netloc not in {"", "localhost"}:
            raise RuntimeError(f"file:// output with host is unsupported: {uri}")
        raw_path = urllib.parse.unquote(parsed.path) if scheme == "file" else uri
        dest = _resolve_local_path(raw_path, base_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        return

    raise RuntimeError(f"unsupported output URI scheme: {scheme} ({uri})")


def _ensure_fai(reference_path: Path) -> None:
    fai = reference_path.with_suffix(reference_path.suffix + ".fai")
    if fai.exists():
        return
    samtools = shutil.which("samtools")
    if not samtools:
        raise RuntimeError(f"reference index missing and samtools not found: {fai}")
    subprocess.run([samtools, "faidx", str(reference_path)], check=True)


def _ensure_vcf_tbi(vcf_gz_path: Path) -> Path:
    index_path = Path(str(vcf_gz_path) + ".tbi")
    if index_path.exists():
        return index_path
    tabix = shutil.which("tabix")
    if tabix:
        subprocess.run([tabix, "-f", "-p", "vcf", str(vcf_gz_path)], check=True)
        if index_path.exists():
            return index_path
    bcftools = shutil.which("bcftools")
    if bcftools:
        subprocess.run([bcftools, "index", "-t", "-f", str(vcf_gz_path)], check=True)
        if index_path.exists():
            return index_path
    raise RuntimeError(f"failed to build tabix index (need tabix or bcftools): {index_path}")


def _detect_engine_version(engine_bin: str) -> dict[str, str]:
    try:
        out = subprocess.check_output([engine_bin, "--version"], text=True, stderr=subprocess.STDOUT)
    except Exception as ex:
        return {"raw": f"error: {ex}"}
    raw = out.strip()
    # Expected: "ogn_run version X (git Y)"
    parts = raw.split()
    version = ""
    git_sha = ""
    if len(parts) >= 4 and parts[1] == "version":
        version = parts[2]
        if "(git" in raw and raw.endswith(")"):
            try:
                git_sha = raw.split("(git", 1)[1].rstrip(")").strip()
            except Exception:
                git_sha = ""
    return {"raw": raw, "version": version, "git_sha": git_sha}


def _run_engine(
    *,
    engine_bin: str,
    profile: str,
    fastq1: Path,
    fastq2: Path | None,
    reference: Path,
    output_vcf: Path,
    parameters: Mapping[str, Any],
    log_path: Path,
    cwd: Path,
) -> int:
    cmd: list[str] = [engine_bin, profile, "--fastq", str(fastq1)]
    if fastq2 is not None:
        cmd += ["--fastq2", str(fastq2)]
    cmd += ["--ref", str(reference), "--vcf", str(output_vcf)]

    # Stable minimal knobs (extend cautiously).
    if "lane" in parameters:
        cmd += ["--lane", str(parameters["lane"])]
    if "gpu_streams" in parameters:
        cmd += ["--gpu-streams", str(int(parameters["gpu_streams"]))]
    if _is_truthy(parameters.get("cpu_only")):
        cmd.append("--cpu-only")
    if "bundle" in parameters:
        cmd += ["--bundle", str(parameters["bundle"])]
    if "pipeline" in parameters:
        cmd += ["--pipeline", str(parameters["pipeline"])]

    # Optional: preserve additional args via a single explicit escape hatch.
    extra_args = parameters.get("extra_args")
    if isinstance(extra_args, list) and all(isinstance(x, str) for x in extra_args):
        cmd += extra_args

    env = os.environ.copy()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_fp:
        _write_log_jsonl_line(log_fp, "runner", f"exec: {' '.join(cmd)}")
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            _write_log_jsonl_line(log_fp, "engine", line)
    return proc.wait()


def _build_provenance(
    *,
    job: Mapping[str, Any],
    engine_version: Mapping[str, str],
    outputs: Mapping[str, ArtifactTarget],
    started_at: str,
    finished_at: str,
    ok: bool,
    error: str | None,
    engine_exit_code: int | None,
    local_artifacts: Mapping[str, Path],
) -> dict[str, Any]:
    engine = _expect_mapping(job.get("engine"), "engine")
    inputs = _expect_mapping(job.get("inputs"), "inputs")
    params = engine.get("parameters") if isinstance(engine.get("parameters"), Mapping) else {}

    out_uris = {name: target.uri for name, target in outputs.items()}
    artifact_hashes: dict[str, str] = {}
    for name, path in local_artifacts.items():
        if path.exists():
            artifact_hashes[name] = _sha256_file(path)

    prov: dict[str, Any] = {
        "schema": "ogn_provenance@1",
        "schema_version": "v1",
        "run_id": job.get("run_id"),
        "tenant_id": job.get("tenant_id"),
        "started_at": started_at,
        "finished_at": finished_at,
        "status": {"ok": ok, "error": error},
        # Convenience fields for the pilot contract (duplicated from engine.*).
        "engine_version": engine_version.get("version") or engine.get("version") or None,
        "engine_git_sha": engine_version.get("git_sha") or None,
        "engine": {
            "profile": engine.get("profile"),
            "requested_version": engine.get("version"),
            "resolved_version": engine_version.get("version") or None,
            "resolved_git_sha": engine_version.get("git_sha") or None,
            "version_raw": engine_version.get("raw") or None,
            "exit_code": engine_exit_code,
        },
        "parameters": dict(params),
        "inputs": {
            "fastq_uris": inputs.get("fastq_uris"),
            "reference_uri": inputs.get("reference_uri"),
            "extra": inputs.get("extra") if isinstance(inputs.get("extra"), Mapping) else {},
        },
        "outputs": out_uris,
        "artifact_sha256": artifact_hashes,
        "runner": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
    }
    submitted_at = job.get("submitted_at")
    if submitted_at is not None:
        prov["submitted_at"] = submitted_at
    # Drop noisy nulls to keep the payload clean.
    for k in ["resolved_version", "resolved_git_sha", "version_raw"]:
        if prov["engine"].get(k) in {"", None}:
            prov["engine"].pop(k, None)
    for k in ["engine_version", "engine_git_sha"]:
        if prov.get(k) in {"", None}:
            prov.pop(k, None)
    if prov["engine"].get("exit_code") is None:
        prov["engine"].pop("exit_code", None)
    if error is None:
        prov["status"].pop("error", None)
    return prov


def _validate_job(job: Any) -> Mapping[str, Any]:
    spec = _expect_mapping(job, "job_spec")
    if spec.get("schema_version") != "v1":
        raise ValueError("job_spec.schema_version must be 'v1'")
    _expect_str(spec.get("run_id"), "job_spec.run_id")
    _expect_str(spec.get("tenant_id"), "job_spec.tenant_id")
    if spec.get("submitted_at") is not None:
        _expect_str(spec.get("submitted_at"), "job_spec.submitted_at")

    engine = _expect_mapping(spec.get("engine"), "job_spec.engine")
    _expect_str(engine.get("profile"), "job_spec.engine.profile")
    _expect_str(engine.get("version"), "job_spec.engine.version")
    params = engine.get("parameters")
    if params is not None and not isinstance(params, Mapping):
        raise ValueError("job_spec.engine.parameters must be an object if provided")

    inputs = _expect_mapping(spec.get("inputs"), "job_spec.inputs")
    fastq_uris = _expect_list(inputs.get("fastq_uris"), "job_spec.inputs.fastq_uris")
    if not (1 <= len(fastq_uris) <= 2) or not all(isinstance(u, str) and u.strip() for u in fastq_uris):
        raise ValueError("job_spec.inputs.fastq_uris must be a list of 1 or 2 non-empty strings")
    _expect_str(inputs.get("reference_uri"), "job_spec.inputs.reference_uri")
    extra = inputs.get("extra")
    if extra is not None and not isinstance(extra, Mapping):
        raise ValueError("job_spec.inputs.extra must be an object if provided")

    outputs = _parse_artifacts(spec.get("outputs"))
    if "provenance" not in outputs and "provenance.json" not in outputs:
        raise ValueError("job_spec.outputs must include a provenance artifact target")
    if "vcf" not in outputs and "out.vcf.gz" not in outputs:
        raise ValueError("job_spec.outputs must include a vcf artifact target")
    return spec


def _artifact_by_alias(artifacts: Mapping[str, ArtifactTarget], *names: str) -> ArtifactTarget | None:
    for n in names:
        if n in artifacts:
            return artifacts[n]
    return None


def run_ogn_runner(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="ogn-runner", description="OGN engine runner for Job Spec JSON v1")
    ap.add_argument("job_spec", nargs="?", help="Path to Job Spec JSON v1 (or '-' for stdin)")
    ap.add_argument("--version", action="store_true", help="Print engine version and exit")
    ap.add_argument(
        "-E",
        "--engine-bin",
        help="Engine executable to run (default: $OGN_ENGINE_BIN or ogn_run on PATH)",
    )
    ap.add_argument("--keep-workdir", action="store_true", help="Do not delete the temp work directory")
    ap.add_argument("--workdir", help="Explicit work directory (default: temp dir)")
    args = ap.parse_args(argv[1:])

    engine_bin = (
        args.engine_bin or os.environ.get("OGN_ENGINE_BIN") or shutil.which("ogn_run") or "ogn_run"
    )
    if args.version:
        info = _detect_engine_version(engine_bin)
        sys.stdout.write(info.get("raw", "unknown") + "\n")
        return 0

    if not args.job_spec:
        ap.error("job_spec is required")

    job_spec_path = Path(args.job_spec) if args.job_spec != "-" else None
    base_dir = job_spec_path.parent.resolve() if job_spec_path else Path.cwd()
    try:
        raw = sys.stdin.read() if job_spec_path is None else job_spec_path.read_text(encoding="utf-8")
        job = json.loads(raw)
        spec = _validate_job(job)
    except Exception as ex:
        sys.stderr.write(f"[ogn-runner] job spec invalid: {ex}\n")
        return 2

    workdir = Path(args.workdir).resolve() if args.workdir else Path(tempfile.mkdtemp(prefix="ogn-runner-"))
    inputs_dir = workdir / "inputs"
    outputs_dir = workdir / "outputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    started_at = _utc_now_iso()
    engine_version = _detect_engine_version(engine_bin)
    ok = False
    error: str | None = None
    exit_code = 1

    artifacts = _parse_artifacts(spec.get("outputs"))
    vcf_target = _artifact_by_alias(artifacts, "vcf", "out.vcf.gz")
    prov_target = _artifact_by_alias(artifacts, "provenance", "provenance.json")
    logs_target = _artifact_by_alias(artifacts, "logs", "logs.jsonl")
    if vcf_target is None or prov_target is None:
        sys.stderr.write("[ogn-runner] outputs missing required vcf/provenance targets\n")
        return 2

    local_vcf = outputs_dir / "out.vcf.gz"
    local_vcf_tbi = outputs_dir / "out.vcf.gz.tbi"
    local_logs = outputs_dir / "logs.jsonl"
    local_prov = outputs_dir / "provenance.json"

    local_artifacts: dict[str, Path] = {
        "vcf": local_vcf,
        "vcf_tbi": local_vcf_tbi,
        "logs": local_logs,
        "provenance": local_prov,
    }

    try:
        inputs = _expect_mapping(spec.get("inputs"), "inputs")
        fastq_uris = [str(u) for u in _expect_list(inputs.get("fastq_uris"), "inputs.fastq_uris")]
        ref_uri = _expect_str(inputs.get("reference_uri"), "inputs.reference_uri")

        fastq1 = _materialize_input_uri(fastq_uris[0], inputs_dir / "reads_R1.fastq.gz", base_dir)
        fastq2 = (
            _materialize_input_uri(fastq_uris[1], inputs_dir / "reads_R2.fastq.gz", base_dir)
            if len(fastq_uris) == 2
            else None
        )
        reference = _materialize_input_uri(ref_uri, inputs_dir / "reference.fa", base_dir)
        _ensure_fai(reference)

        engine = _expect_mapping(spec.get("engine"), "engine")
        profile = _expect_str(engine.get("profile"), "engine.profile")
        parameters = engine.get("parameters") if isinstance(engine.get("parameters"), Mapping) else {}

        exit_code = _run_engine(
            engine_bin=engine_bin,
            profile=profile,
            fastq1=fastq1,
            fastq2=fastq2,
            reference=reference,
            output_vcf=local_vcf,
            parameters=parameters,
            log_path=local_logs,
            cwd=workdir,
        )
        ok = exit_code == 0
        if not ok:
            error = f"engine failed with exit code {exit_code}"
        else:
            # If the job requests an index artifact, produce it.
            if _artifact_by_alias(artifacts, "vcf_tbi", "out.vcf.gz.tbi") is not None:
                built = _ensure_vcf_tbi(local_vcf)
                if built != local_vcf_tbi:
                    try:
                        shutil.copyfile(built, local_vcf_tbi)
                    except Exception:
                        local_vcf_tbi = built
    except Exception as ex:
        ok = False
        error = f"{type(ex).__name__}: {ex}"
        sys.stderr.write(f"[ogn-runner] error: {error}\n")
        traceback.print_exc(file=sys.stderr)

    finished_at = _utc_now_iso()

    provenance = _build_provenance(
        job=spec,
        engine_version=engine_version,
        outputs=artifacts,
        started_at=started_at,
        finished_at=finished_at,
        ok=ok,
        error=error,
        engine_exit_code=exit_code if exit_code != 0 or ok else None,
        local_artifacts=local_artifacts,
    )
    local_prov.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Upload artifacts (best-effort for optional entries; strict for required).
    upload_failures: list[str] = []
    local_by_id: dict[str, Path] = {
        "vcf": local_vcf,
        "out.vcf.gz": local_vcf,
        "vcf_tbi": local_vcf_tbi,
        "out.vcf.gz.tbi": local_vcf_tbi,
        "provenance": local_prov,
        "provenance.json": local_prov,
        "logs": local_logs,
        "logs.jsonl": local_logs,
    }
    for name, target in artifacts.items():
        src = local_by_id.get(name)
        if src is None:
            if target.optional:
                continue
            upload_failures.append(f"{name}: unsupported artifact id")
            continue

        if not src.exists():
            if target.optional:
                continue
            upload_failures.append(f"{name}: missing local artifact {src.name}")
            continue
        try:
            if target.put_url:
                _http_put_file(target.put_url, src, target.media_type)
            else:
                _upload_to_uri(target.uri, src, base_dir)
        except Exception as ex:
            if target.optional:
                continue
            upload_failures.append(f"{name}: upload failed: {ex}")

    if upload_failures:
        sys.stderr.write("[ogn-runner] upload failures:\n")
        for msg in upload_failures:
            sys.stderr.write(f"  - {msg}\n")
        ok = False

    if not args.keep_workdir and not args.workdir:
        shutil.rmtree(workdir, ignore_errors=True)
    else:
        sys.stderr.write(f"[ogn-runner] workdir: {workdir}\n")

    return 0 if ok else (exit_code if exit_code != 0 else 1)


if __name__ == "__main__":
    raise SystemExit(run_ogn_runner(sys.argv))


def run(argv: list[str] | None = None) -> int:
    return run_ogn_runner(sys.argv if argv is None else argv)


def main() -> None:
    raise SystemExit(run())
