# ogn-sdk (Python)

`ogn-sdk` is a tiny convenience layer around the `ogn_run` binary. Install it with:

```bash
python3 -m pip install ogn-sdk
```

Example usage:

```python
from ogn_sdk import RunRequest, run_local

req = RunRequest(
    fastq="/data/sample_R1.fastq.gz",
    reference="/data/ref.fa",
    output_vcf="out/sample.g.vcf.gz",
    sample_id="hg002_chr20",
    profile_dir="profile_hg002"
)

result = run_local(req)
print("exit status", result.returncode)
```

`RunRequest` maps directly to CLI flags, so you can pass `fastq2`, `bundle`, or
any additional CLI switches via `extra_args`. The helper simply shells out to
`ogn_run` (defaulting to the binary on `PATH`).
