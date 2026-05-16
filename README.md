# OGN Core Kit

OGN Core Kit is the open-source adoption surface for OGN.

It contains:

- Job Spec JSON v1 contract (`spec/v1/jobspec.md`, `spec/v1/jobspec.schema.json`, fixtures)
- `ogn-runner` worker entrypoint that executes Job Spec workflows and uploads artifacts
- mock mode (`ogn-runner --mock`) for public adoption without proprietary engine access
- receipt contract (`receipt.json`) and verifier (`ogn-verify-receipt`)
- Python package `ogn-sdk` (CLI `ogn`, SDK `ogn_sdk`, runner `ogn_runner`)
- workflow adapters (for example, Nextflow)

It does **not** include the proprietary OGN Engine implementation, CUDA kernels, engine container images, or hosted OGN Cloud access.

## Install

From source (dev / editable):

```bash
python -m pip install -e sdk/python[dev]
```

## Public adoption flow

```bash
ogn-runner --validate examples/minimal-job/job_spec.json
ogn-runner --mock examples/minimal-job/job_spec.json
ogn-verify-receipt examples/minimal-job/receipt.json
```

## Quickstart: `ogn-runner` (Job Spec v1)

`ogn-runner` accepts a jobspec path or `-` for stdin.

Run from a file:

```bash
ogn-runner job_spec.json
```

Run from stdin:

```bash
cat job_spec.json | ogn-runner -
```

## Upload semantics

For each `outputs.*` artifact:

- If `put_url` is present, `ogn-runner` uploads via HTTP PUT to the presigned URL.
- Otherwise it uploads to `uri` directly when supported:
  - `s3://...` via `aws s3 cp` (AWS CLI required)
  - `file://...` or local path via copy

If an output is marked `optional: true`, upload failures are non-fatal.

## Contracts

Canonical artifacts:

- `spec/v1/jobspec.md`
- `spec/v1/jobspec.schema.json`
- `spec/v1/conformance/`
- `spec/v1/receipt.schema.json`

Unknown fields are ignored (additive-only). Breaking changes require a new schema version (`v2`+).

## Documentation

- [`docs/public-adoption-flow.md`](docs/public-adoption-flow.md)
- [`docs/receipts.md`](docs/receipts.md)
- [`docs/what-this-repo-does-not-include.md`](docs/what-this-repo-does-not-include.md)

## Validation

```bash
python tools/validate_conformance.py
pytest -q
./public_audit.sh
```

## Development

```bash
python -m pip install -e sdk/python[dev]
pytest -q
```

## License

Apache 2.0; see `LICENSE` and `NOTICE`.
