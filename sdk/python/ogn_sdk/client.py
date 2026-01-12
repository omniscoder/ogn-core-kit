"""Simple helper to shell out to `ogn_run`."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Optional, Sequence


@dataclass
class RunRequest:
  """Declarative description of a local `ogn_run` invocation."""

  fastq: str
  reference: str
  output_vcf: str
  sample_id: str = "sample"
  fastq2: Optional[str] = None
  bundle: Optional[str] = None
  pipeline: Optional[str] = None
  profile_dir: Optional[str] = None
  extra_args: Sequence[str] = field(default_factory=tuple)

  def to_argv(self) -> list[str]:
    argv = [
        "--fastq",
        self.fastq,
        "--reference",
        self.reference,
        "--vcf",
        self.output_vcf,
        "--sample",
        self.sample_id,
    ]
    if self.fastq2:
      argv.extend(["--fastq2", self.fastq2])
    if self.bundle:
      argv.extend(["--bundle", self.bundle])
    if self.pipeline:
      argv.extend(["--pipeline", self.pipeline])
    if self.profile_dir:
      argv.extend(["--profile", "--profile-dir", self.profile_dir])
    if self.extra_args:
      argv.extend(self.extra_args)
    return argv


def run_local(
    request: RunRequest,
    *,
    binary: str = "ogn_run",
    env: Optional[Mapping[str, str]] = None,
    cwd: Optional[str | Path] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
  """Execute `ogn_run` with the supplied request and return the process result."""

  argv = [binary, *request.to_argv()]
  merged_env: MutableMapping[str, str] = dict(os.environ)
  if env:
    merged_env.update(env)
  return subprocess.run(
      argv,
      cwd=str(cwd) if cwd else None,
      env=merged_env,
      check=check,
      capture_output=False,
  )


def format_command(request: RunRequest, binary: str = "ogn_run") -> str:
  """Return a human-friendly string for logging."""

  return " ".join(shlex.quote(part) for part in [binary, *request.to_argv()])


def _grpcurl_bin() -> str:
  candidate = os.environ.get("GRPCURL_BIN", "grpcurl")
  if shutil.which(candidate) is None:
    raise RuntimeError(
        f"grpcurl not found (looked for '{candidate}'); set GRPCURL_BIN or install grpcurl."
    )
  return candidate


def submit_run_grpc(
    host: str,
    tenant_id: str,
    fastqs: Sequence[str],
    reference_uri: str,
    sample_id: str = "sample",
    pipeline_json: str = "{}",
    token: Optional[str] = None,
) -> dict:
  """Invoke the OgnControl.SubmitRun RPC via grpcurl and return the parsed JSON."""

  payload = {
      "tenant_id": tenant_id,
      "sample_id": sample_id,
      "fastq_uris": list(fastqs),
      "reference_uri": reference_uri,
      "pipeline_json": pipeline_json,
  }
  args = [_grpcurl_bin()]
  if token:
    args.extend(["-H", f"Authorization: Bearer {token}"])
  args.extend(["-d", json.dumps(payload), host, "ogn.OgnControl/SubmitRun"])
  output = subprocess.check_output(args, text=True)
  return json.loads(output)
