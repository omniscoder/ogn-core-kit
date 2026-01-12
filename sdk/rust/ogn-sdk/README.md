# ogn-sdk (Rust)

Add the crate to your project:

```bash
cargo add ogn-sdk --git https://github.com/your-org/OGN --branch main
```

Example:

```rust
use ogn_sdk::{RunRequest, run_local};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let req = RunRequest::builder()
        .fastq("data/sample.fastq")
        .reference("data/ref.fa")
        .output_vcf("out/sample.g.vcf.gz")
        .sample_id("hg002_chr20")
        .build();
    run_local(&req, None)?;
    Ok(0.into())
}
```
