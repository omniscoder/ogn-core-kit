# Contributing to OGN Core Kit

Thanks for helping improve the OGN Core Kit adoption surface (runner, contracts, SDKs, workflow glue).

## Quick dev setup (Python)

```bash
python -m pip install -e sdk/python[dev]
pytest -q
```

## Contract / conformance changes

Job Spec v1 is **additive-only**. Do not make optional fields required or change meanings in-place.

When editing anything under `spec/v1/`:

```bash
python tools/validate_conformance.py
```

If you add or modify fixtures, keep them minimal and deterministic.

## Public repo hygiene

Before opening a PR:

```bash
./public_audit.sh
git ls-files -s | awk '$1==120000{print}'
```

## Pull requests

- Keep PRs small and contract-focused.
- If you change runner behavior, add or update a focused test in `tests/`.
- Avoid adding proprietary engine references, hard-coded machine paths, or secrets.
