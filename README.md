# OGN Core Kit

OGN Core Kit is the open adoption surface for OGN:

- Job Spec JSON v1 contract (docs + fixtures)
- `ogn-runner`: a stable worker entrypoint for executing an OGN engine and uploading artifacts
- SDKs and workflow adapters (Nextflow, etc.)

This repository does **not** include the proprietary OGN engine implementation or GPU kernels.
The engine is expected to be provided separately as a binary (e.g. `ogn_run`) or container image.

## Quickstart (local runner)

1) Install the Python package:

```bash
pip install ./sdk/python
```

2) Point the runner at your engine binary (optional; defaults to `ogn_run` on PATH):

```bash
export OGN_ENGINE_BIN=/path/to/ogn_run
```

3) Run a Job Spec v1:

```bash
ogn-runner job_spec.json
```

## Contracts

The canonical pilot contract lives in:

- `spec/v1/jobspec.md`
- `spec/v1/conformance/`

Unknown fields are ignored (additive-only). Breaking changes require a new schema version (v2+).

