# Minimal public adoption example

This example is intentionally minimal and runs with `ogn-runner --mock`.

## Run

```bash
python -m pip install -e sdk/python[dev]
cd examples/minimal-job
ogn-runner --validate job_spec.json
ogn-runner --mock job_spec.json
```

Expected output files appear next to the job spec path by default:

- `out.vcf.gz`
- `provenance.json`
- `logs.jsonl`
- `receipt.json`

Verify the generated receipt:

```bash
ogn-verify-receipt receipt.json
```
