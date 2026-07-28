from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

from .receipt import compute_receipt_payload_sha256, sha256_for_file


def _read_receipt(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as ex:
        raise ValueError(f"invalid JSON: {ex}") from ex
    if not isinstance(raw, dict):
        raise ValueError("receipt must be a JSON object")
    return raw


def _schema_version(receipt: Mapping[str, Any]) -> str:
    value = receipt.get("schema_version")
    if not isinstance(value, str) or not value:
        raise ValueError("missing or invalid receipt.schema_version")
    return value


def _required_fields(receipt: Mapping[str, Any]) -> None:
    required = ["schema_version", "run_id", "tenant_id", "created_at", "engine", "inputs", "outputs", "artifacts", "hashes", "verification"]
    for field in required:
        if field not in receipt:
            raise ValueError(f"missing required field: {field}")


def _verify_verification_block(receipt: Mapping[str, Any]) -> list[str]:
    messages: list[str] = []
    verification = receipt.get("verification")
    if not isinstance(verification, Mapping):
        messages.append("receipt.verification must be an object")
        return messages
    status = verification.get("status")
    if status not in {"unsigned", "signed", "verified"}:
        messages.append("receipt.verification.status must be one of unsigned|signed|verified")
        return messages
    if status in {"signed", "verified"} and "signature" not in receipt:
        messages.append("signature unsupported/not supplied")
        messages.append("signed receipts require signature data; verification currently stubbed")
    return messages


def _resolve_artifact_path(
    *,
    value: str,
    receipt_path: Path,
    base_dir: Path | None = None,
) -> Path:
    candidate = Path(value)
    root = (Path(base_dir) if base_dir is not None else receipt_path.parent).resolve()
    resolved = (
        candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    )
    try:
        resolved.relative_to(root)
    except ValueError as ex:
        raise ValueError(f"artifact path escapes verification root: {value}") from ex
    return resolved


def _verify_artifacts(
    receipt: Mapping[str, Any],
    *,
    receipt_path: Path,
    metadata_only: bool,
    base_dir: Path | None = None,
) -> list[str]:
    messages: list[str] = []
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list):
        return ["receipt.artifacts must be an array"]
    for idx, raw in enumerate(artifacts):
        if not isinstance(raw, Mapping):
            messages.append(f"receipt.artifacts[{idx}] must be an object")
            continue
        path = str(raw.get("path", ""))
        if not path:
            messages.append(f"receipt.artifacts[{idx}].path must be a non-empty string")
            continue
        if not metadata_only:
            try:
                artifact_path = _resolve_artifact_path(
                    value=path,
                    receipt_path=receipt_path,
                    base_dir=base_dir,
                )
            except ValueError as ex:
                messages.append(str(ex))
                continue
            if not artifact_path.exists():
                messages.append(f"missing artifact: {path}")
                continue
            expected = raw.get("sha256")
            if not isinstance(expected, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", expected or ""):
                messages.append(f"receipt.artifacts[{idx}].sha256 must be a 64-hex string")
                continue
            actual = sha256_for_file(artifact_path)
            if actual.lower() != expected.lower():
                messages.append(f"hash mismatch for {path}")
    return messages


def _verify_hash(receipt: Mapping[str, Any]) -> list[str]:
    messages: list[str] = []
    hashes = receipt.get("hashes")
    if not isinstance(hashes, Mapping):
        messages.append("receipt.hashes must be an object")
        return messages
    expected = hashes.get("receipt_payload_sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", expected):
        messages.append("receipt.hashes.receipt_payload_sha256 must be a 64-hex string")
        return messages
    actual = compute_receipt_payload_sha256(receipt)
    if actual.lower() != expected.lower():
        messages.append(f"receipt payload hash mismatch: {expected} != {actual}")
    return messages


def verify_receipt(
    path: Path,
    metadata_only: bool = False,
    base_dir: Path | None = None,
) -> list[str]:
    receipt = _read_receipt(path)
    messages: list[str] = []
    if _schema_version(receipt) != "ogn.receipt.v1":
        messages.append("receipt.schema_version must be ogn.receipt.v1")
    try:
        _required_fields(receipt)
    except ValueError as ex:
        messages.append(str(ex))
        return messages

    messages.extend(_verify_hash(receipt))
    messages.extend(
        _verify_artifacts(
            receipt,
            receipt_path=path,
            metadata_only=metadata_only,
            base_dir=base_dir,
        )
    )
    messages.extend(_verify_verification_block(receipt))
    return messages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ogn-verify-receipt")
    parser.add_argument("receipt", help="Path to receipt.json")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--base-dir", help="Override root directory for relative artifact paths")
    parsed = parser.parse_args(argv)

    receipt_path = Path(parsed.receipt)
    if not receipt_path.exists():
        return 2

    try:
        override_base_dir = Path(parsed.base_dir).resolve() if parsed.base_dir else None
        messages = verify_receipt(
            receipt_path,
            metadata_only=parsed.metadata_only,
            base_dir=override_base_dir,
        )
    except ValueError as ex:
        print(f"ERROR: {ex}")
        return 2
    if messages:
        for msg in messages:
            if msg:
                print(f"ERROR: {msg}")
        return 1
    print("receipt valid")
    return 0


def run(argv: list[str] | None = None) -> int:
    return main(argv)


def _main(argv: list[str] | None = None) -> None:
    raise SystemExit(main(argv))


if __name__ == "__main__":
    _main()
