# Job Spec v1 conformance fixtures

These fixtures exist to keep `schema_version: "v1"` stable and enforceable across the OGN ecosystem.

Each `*.json` file is a single conformance case with this shape:

```json
{
  "kind": "jobspec|timeline",
  "name": "human-readable",
  "jobspec": { "...": "..." },
  "timeline": ["QUEUED", "RUNNING", "SUCCEEDED"],
  "want_ok": true,
  "want_code": "OK|INVALID_JOBSPEC|INVALID_TIMELINE",
  "want_error": "exact error string (when want_ok=false)"
}
```

In this repo, `tools/validate_conformance.py` performs schema-level validation for `kind: "jobspec"` cases.
Timeline/state-machine semantics (`kind: "timeline"`) are enforced by the control plane implementation.
