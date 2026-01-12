#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Case:
    path: Path
    kind: str
    name: str
    jobspec: dict[str, Any] | None
    want_ok: bool | None
    want_code: str | None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_cases() -> list[Case]:
    conformance_dir = _repo_root() / "spec" / "v1" / "conformance"
    cases: list[Case] = []
    for path in sorted(conformance_dir.glob("*.json")):
        raw = _load_json(path)
        cases.append(
            Case(
                path=path,
                kind=str(raw.get("kind", "")),
                name=str(raw.get("name", path.name)),
                jobspec=raw.get("jobspec") if isinstance(raw.get("jobspec"), dict) else None,
                want_ok=raw.get("want_ok") if isinstance(raw.get("want_ok"), bool) else None,
                want_code=str(raw.get("want_code")) if raw.get("want_code") is not None else None,
            )
        )
    return cases


def _validate_instance(schema: dict[str, Any], instance: dict[str, Any]) -> None:
    try:
        import jsonschema  # type: ignore[import-not-found]
    except Exception as ex:  # pragma: no cover
        raise RuntimeError(
            f"jsonschema is required to validate conformance fixtures (pip install jsonschema): {ex}"
        ) from ex

    jsonschema.validate(instance=instance, schema=schema)


def main(argv: list[str]) -> int:
    if "--help" in argv or "-h" in argv:
        sys.stdout.write(
            "validate_conformance.py: validates spec/v1/conformance/*.json jobspec cases against jobspec.schema.json\n"
        )
        return 0

    root = _repo_root()
    schema_path = root / "spec" / "v1" / "jobspec.schema.json"
    schema = _load_json(schema_path)

    ok = True
    for case in _load_cases():
        if case.kind != "jobspec":
            continue
        if case.jobspec is None:
            sys.stderr.write(f"[conformance] {case.path}: missing jobspec\n")
            ok = False
            continue

        want_invalid = case.want_code == "INVALID_JOBSPEC"
        try:
            _validate_instance(schema, case.jobspec)
            if want_invalid:
                sys.stderr.write(f"[conformance] {case.name}: expected INVALID_JOBSPEC but schema accepted\n")
                ok = False
            else:
                sys.stderr.write(f"[conformance] {case.name}: OK\n")
        except Exception as ex:
            if want_invalid:
                sys.stderr.write(f"[conformance] {case.name}: rejected as expected\n")
            else:
                sys.stderr.write(f"[conformance] {case.name}: schema rejected: {ex}\n")
                ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

