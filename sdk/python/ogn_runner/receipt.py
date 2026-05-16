from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_for_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_receipt_payload_sha256(receipt: Mapping[str, Any]) -> str:
    payload = _strip_receipt_payload_hash(receipt)
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def validate_receipt_payload_hash(receipt: Mapping[str, Any]) -> bool:
    hashes = _expect_mapping(receipt.get("hashes"), "receipt.hashes")
    expected = hashes.get("receipt_payload_sha256")
    if not isinstance(expected, str):
        return False
    return compute_receipt_payload_sha256(receipt) == expected


def _strip_receipt_payload_hash(receipt: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = dict(receipt)
    hashes = dict(_expect_mapping(payload.get("hashes"), "receipt.hashes"))
    hashes.pop("receipt_payload_sha256", None)
    payload["hashes"] = hashes
    return payload


@dataclasses.dataclass(frozen=True)
class ArtifactReceipt:
    id: str
    path: str
    sha256: str
    media_type: str | None = None
    optional: bool = False


def make_artifact_receipt(
    *, name: str, path: Path, media_type: str | None = None, optional: bool = False
) -> ArtifactReceipt:
    return ArtifactReceipt(
        id=name,
        path=str(path.resolve()),
        sha256=sha256_for_file(path),
        media_type=media_type,
        optional=optional,
    )


def build_receipt(
    *,
    run_id: str,
    tenant_id: str,
    created_at: str | None,
    engine: Mapping[str, Any],
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
    artifacts: Sequence[ArtifactReceipt],
    verification: Mapping[str, Any] | None = None,
    signature: Mapping[str, Any] | None = None,
    created_by: str | None = None,
) -> dict[str, Any]:
    if created_at is None:
        created_at = _utc_now_iso()

    receipt: dict[str, Any] = {
        "schema_version": "ogn.receipt.v1",
        "run_id": run_id,
        "tenant_id": tenant_id,
        "created_at": created_at,
        "engine": {
            "profile": engine.get("profile"),
            "requested_version": engine.get("version"),
            "additional": dict(engine),
        },
        "inputs": dict(inputs),
        "outputs": dict(outputs),
        "artifacts": [dataclasses.asdict(artifact) for artifact in artifacts],
        "hashes": {"receipt_payload_sha256": ""},
    }

    if signature is not None:
        receipt["signature"] = dict(signature)

    if created_by is not None:
        receipt["created_by"] = created_by

    if verification is not None:
        status = verification.get("status")
        if isinstance(status, str):
            receipt["verification"] = {"status": status}
    if "verification" not in receipt:
        receipt["verification"] = {"status": "unsigned"}
    else:
        # Keep existing keys if caller provides details.
        receipt["verification"].update(dict(verification or {}))

    receipt["hashes"]["receipt_payload_sha256"] = compute_receipt_payload_sha256(receipt)
    return receipt


def _expect_mapping(value: Any, path: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    raise ValueError(f"{path} must be an object")


def _normalize_verification_status(value: str) -> str:
    if value not in {"unsigned", "signed", "verified"}:
        raise ValueError(f"invalid receipt.verification.status: {value}")
    return value


def normalize_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(receipt)
    verification = _expect_mapping(payload.get("verification"), "receipt.verification")
    status = verification.get("status")
    if isinstance(status, str):
        verification["status"] = _normalize_verification_status(status)
    payload["verification"] = verification
    return payload
