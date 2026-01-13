# OGN Core Kit

OGN Core Kit is the open adoption surface for Omnis Genome Nexus (OGN).

It contains:

- Job Spec JSON v1 contract (`spec/v1/jobspec.md` + `spec/v1/jobspec.schema.json` + fixtures)
- `ogn-runner`: stable worker entrypoint that executes an OGN engine and uploads artifacts
- Python package `ogn-sdk` (CLI `ogn`, SDK `ogn_sdk`, runner `ogn_runner`)
- Workflow adapters (e.g. `pipelines/ogn_variant.nf`)

It does **not** include the proprietary OGN engine implementation, GPU kernels, or engine container images. The engine
must be provided separately as a binary (e.g. `ogn_run`) or container image.

## Install

### Python (CLI + runner + SDK)

Requires Python 3.9+.

From PyPI:

```bash
python -m pip install ogn-sdk
```

From source (dev / editable):

```bash
python -m pip install -e sdk/python[dev]
```

This installs:

- `ogn` (CLI wrapper; requires access to an engine binary such as `ogn_variant_runner`)
- `ogn-runner` (worker entrypoint for Job Spec v1)
- `ogn_sdk` / `ogn_runner` Python modules

### Rust (SDK)

The Rust crate lives at `sdk/rust/ogn-sdk`.

- If/when published, install via crates.io.
- Until then, consume it via a path dependency in a clone:

```toml
ogn-sdk = { path = "sdk/rust/ogn-sdk" }
```

## Quickstart: `ogn-runner` (Job Spec v1)

`ogn-runner` accepts a jobspec path or `-` for stdin (the gateway worker contract is `ogn-runner -`).

Engine selection:

```bash
export OGN_ENGINE_BIN=/path/to/ogn_run
# or: ogn-runner -E /path/to/ogn_run job_spec.json
```

Run from a file:

```bash
ogn-runner job_spec.json
```

Run from stdin:

```bash
cat job_spec.json | ogn-runner -
```

### Upload semantics

For each `outputs.*` artifact:

- If `put_url` is present, `ogn-runner` uploads via HTTP PUT to the presigned URL.
- Otherwise it uploads to `uri` directly when supported:
  - `s3://...` via `aws s3 cp` (AWS CLI required)
  - `file://...` or a plain path via a local copy

If an output is marked `optional: true`, upload failures are non-fatal.

## Contracts

Canonical artifacts:

- `spec/v1/jobspec.md`
- `spec/v1/jobspec.schema.json`
- `spec/v1/conformance/`

Schema-level validation:

```bash
python tools/validate_conformance.py
```

Unknown fields are ignored (additive-only). Breaking changes require a new schema version (v2+).

## Development

Run tests:

```bash
python -m pip install -e sdk/python[dev]
pytest -q
```

Public repo hygiene:

```bash
./public_audit.sh
```

## License

Apache 2.0; see `LICENSE` and `NOTICE`.

## Security

See `SECURITY.md`.

## Contributing

See `CONTRIBUTING.md`.
