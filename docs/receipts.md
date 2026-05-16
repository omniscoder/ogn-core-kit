# OGN Receipts and Proof Bundles

This repo adds a public verification contract that is independent of any proprietary engine image.

A run emits:

- `provenance.json`: execution detail for the local run path (including outputs and engine metadata)
- `receipt.json`: verification contract for portable validation
- `metrics.json` (optional, when produced by the execution environment)
- logs and result artifacts declared by Job Spec v1

`receipt.json` is the user-facing verification contract. It contains:

- stable identity (`run_id`, `tenant_id`, `created_at`)
- contract context (`engine`, `inputs`, `outputs`)
- artifact list + SHA-256 for emitted files
- canonical `hashes.receipt_payload_sha256`
- `verification` status (`unsigned`, `signed`, or `verified`)

## Verification status semantics

- `unsigned` (default): no signature expectations
- `signed`: signature is present and expected (signature verification is currently stubbed)
- `verified`: signature and policy checks are expected (currently stubbed)

Use `ogn-verify-receipt` for public checks.

## Recommended public command

```bash
ogn-verify-receipt <path>/receipt.json
```
