"""Minimal gRPC client using the canonical OGN control-plane proto.

This module avoids import-time failures if generated stubs are not present.
It imports the generated modules lazily when OGNClient is instantiated.
Generate stubs via: scripts/sdk/gen_python_grpc.sh
"""

from __future__ import annotations

import typing as _t

import grpc  # type: ignore


def _load_stubs():
  # Try common generated module names; fall back to repo-local names if present.
  candidates = [
      ("ogn_sdk.generated.ogn_control_pb2", "ogn_sdk.generated.ogn_control_pb2_grpc"),
      ("ogn_control_pb2", "ogn_control_pb2_grpc"),
  ]
  last_err: Exception | None = None
  for mod_pb2, mod_pb2g in candidates:
    try:
      pb2 = __import__(mod_pb2, fromlist=["*"])  # type: ignore
      pb2g = __import__(mod_pb2g, fromlist=["*"])  # type: ignore
      return pb2, pb2g
    except Exception as ex:  # pragma: no cover
      last_err = ex
      continue
  raise RuntimeError(
      "OGN gRPC stubs not found. Run scripts/sdk/gen_python_grpc.sh before packaging/using OGNClient."
  ) from last_err


class OGNClient:
  """Thin wrapper around the OGN control plane service."""

  def __init__(self, endpoint: str, *, insecure: bool = False, token: str | None = None, timeout_s: float = 30.0):
    self._pb2, self._pb2g = _load_stubs()
    self._timeout = timeout_s

    if insecure:
      channel = grpc.insecure_channel(endpoint)
    else:
      creds = grpc.ssl_channel_credentials()
      if token:
        call_creds = grpc.metadata_call_credentials(
            lambda _, cb: cb((("authorization", f"Bearer {token}"),), None)
        )
        channel = grpc.secure_channel(endpoint, grpc.composite_channel_credentials(creds, call_creds))
      else:
        channel = grpc.secure_channel(endpoint, creds)

    # Resolve stub class name across possible canonical names
    stub_cls: _t.Any = None
    for name in ("OgnControlStub", "ControlServiceStub", "ControlStub"):
      stub_cls = getattr(self._pb2g, name, None)
      if stub_cls:
        break
    if stub_cls is None:
      raise RuntimeError("Could not locate control-plane Stub class in generated module")

    self._stub = stub_cls(channel)

  # RPC helpers use canonical method names; adapt to available surface.
  def submit_run(self, req: _t.Any) -> _t.Any:
    return self._stub.SubmitRun(req, timeout=self._timeout)

  def get_run_status(self, req: _t.Any) -> _t.Any:
    # Canonical method is GetRunStatus
    method = getattr(self._stub, "GetRunStatus")
    return method(req, timeout=self._timeout)

  def list_artifacts(self, req: _t.Any) -> _t.Any:
    method = getattr(self._stub, "ListArtifacts")
    return method(req, timeout=self._timeout)

  def stream_logs(self, req: _t.Any) -> _t.Any:
    return self._stub.StreamLogs(req, timeout=self._timeout)

  def cancel_run(self, req: _t.Any) -> _t.Any:
    return self._stub.CancelRun(req, timeout=self._timeout)
