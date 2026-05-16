# Public Adoption Flow for OGN Core Kit

OGN Core Kit is the public entrypoint for anyone wiring contract-compliant workers or pilots.

## 1) Validate the job contract

```bash
ogn-runner --validate examples/minimal-job/job_spec.json
```

## 2) Execute without proprietary compute (mock mode)

```bash
ogn-runner --mock examples/minimal-job/job_spec.json
```

This writes deterministic mock outputs and a receipt so teams can inspect the full contract flow without a private engine dependency.

## 3) Verify the receipt

```bash
ogn-verify-receipt examples/minimal-job/receipt.json
```

## 4) Wire with hosted execution later

Once the same job spec is executable in your environment, the hosted flow is the same:

- produce `receipt.json`
- collect `provenance.json`
- emit logs and artifacts to your destination URIs

The contract keeps these surfaces stable while engine implementations can evolve separately.
