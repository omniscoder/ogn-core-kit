# Job Spec JSON v1 (frozen pilot contract)

Job Spec v1 is the payload published on `ogn.run.submit.v1` (gateway → worker). It is validated by `internal/jobspec/v1.go` and used end-to-end in the pilot.

## Versioning policy

- `schema_version` **MUST** be the exact string `"v1"`.
- v1 is **additive-only**:
  - Adding new optional fields is allowed.
  - Unknown fields **MUST** be ignored by consumers.
- v1 is **frozen against breaking changes**:
  - You MUST NOT change the meaning of existing fields.
  - You MUST NOT make optional fields required.
  - You MUST NOT remove fields.
- Any breaking change requires a new `schema_version` (v2+) and continued support for v1 during the pilot.

Conformance fixtures live in `spec/v1/conformance/` and are enforced by `internal/jobspec/conformance_test.go`.

## Schema (high level)

Required fields:

- `schema_version`: `"v1"`
- `run_id`: non-empty string
- `tenant_id`: non-empty string
- `engine.version`: non-empty string
- `engine.profile`: non-empty string
- `inputs.fastq_uris`: array length >= 1
- `inputs.reference_uri`: non-empty string
- `outputs`: object with >= 1 entry; each output requires `uri`

Optional fields:

- `submitted_at`: RFC3339 timestamp (recommended)
- `engine.parameters`: free-form object
- `inputs.extra`: free-form object
- `outputs.*.put_url`: presigned PUT URL for workers to upload to
- `outputs.*.optional`: bool
- `outputs.*.media_type`: string
- `control.cancel_subject`, `control.logs_subject`: NATS subjects (recommended)

## Output contract

- `outputs.*.uri` is the **stable artifact identity** (e.g. `s3://bucket/key`).
- `outputs.*.put_url` is a short-lived presigned URL used only for upload.
- Finalization events MUST reference stable URIs (never presigned URLs).

