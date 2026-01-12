# Job Spec v1 conformance fixtures

These fixtures exist to keep `schema_version: "v1"` stable and enforceable.

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

The runner lives in `internal/jobspec/conformance_test.go`.

