use std::fmt;
use std::process::{Command, Output};

use anyhow::{anyhow, ensure};
use serde_json::json;

/// Request describing an `ogn_run` invocation.
#[derive(Debug, Clone)]
pub struct RunRequest {
    pub fastq: String,
    pub reference: String,
    pub output_vcf: String,
    pub sample_id: String,
    pub fastq2: Option<String>,
    pub bundle: Option<String>,
    pub pipeline: Option<String>,
    pub profile_dir: Option<String>,
    pub extra_args: Vec<String>,
}

impl RunRequest {
    pub fn builder() -> RunRequestBuilder {
        RunRequestBuilder::default()
    }

    fn argv(&self) -> Vec<String> {
        let mut args = vec![
            "--fastq".to_string(),
            self.fastq.clone(),
            "--reference".to_string(),
            self.reference.clone(),
            "--vcf".to_string(),
            self.output_vcf.clone(),
            "--sample".to_string(),
            self.sample_id.clone(),
        ];
        if let Some(fq2) = &self.fastq2 {
            args.push("--fastq2".into());
            args.push(fq2.clone());
        }
        if let Some(bundle) = &self.bundle {
            args.push("--bundle".into());
            args.push(bundle.clone());
        }
        if let Some(pipeline) = &self.pipeline {
            args.push("--pipeline".into());
            args.push(pipeline.clone());
        }
        if let Some(profile_dir) = &self.profile_dir {
            args.push("--profile".into());
            args.push("--profile-dir".into());
            args.push(profile_dir.clone());
        }
        args.extend(self.extra_args.clone());
        args
    }
}

#[derive(Debug, Default)]
pub struct RunRequestBuilder {
    fastq: Option<String>,
    reference: Option<String>,
    output_vcf: Option<String>,
    sample_id: Option<String>,
    fastq2: Option<String>,
    bundle: Option<String>,
    pipeline: Option<String>,
    profile_dir: Option<String>,
    extra_args: Vec<String>,
}

impl RunRequestBuilder {
    pub fn fastq(mut self, value: impl Into<String>) -> Self {
        self.fastq = Some(value.into());
        self
    }

    pub fn reference(mut self, value: impl Into<String>) -> Self {
        self.reference = Some(value.into());
        self
    }

    pub fn output_vcf(mut self, value: impl Into<String>) -> Self {
        self.output_vcf = Some(value.into());
        self
    }

    pub fn sample_id(mut self, value: impl Into<String>) -> Self {
        self.sample_id = Some(value.into());
        self
    }

    pub fn fastq2(mut self, value: impl Into<String>) -> Self {
        self.fastq2 = Some(value.into());
        self
    }

    pub fn bundle(mut self, value: impl Into<String>) -> Self {
        self.bundle = Some(value.into());
        self
    }

    pub fn pipeline(mut self, value: impl Into<String>) -> Self {
        self.pipeline = Some(value.into());
        self
    }

    pub fn profile_dir(mut self, value: impl Into<String>) -> Self {
        self.profile_dir = Some(value.into());
        self
    }

    pub fn extra_args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for arg in args {
            self.extra_args.push(arg.into());
        }
        self
    }

    pub fn build(self) -> Result<RunRequest, BuildError> {
        Ok(RunRequest {
            fastq: self
                .fastq
                .ok_or(BuildError::MissingField("fastq"))?,
            reference: self
                .reference
                .ok_or(BuildError::MissingField("reference"))?,
            output_vcf: self
                .output_vcf
                .ok_or(BuildError::MissingField("output_vcf"))?,
            sample_id: self.sample_id.unwrap_or_else(|| "sample".into()),
            fastq2: self.fastq2,
            bundle: self.bundle,
            pipeline: self.pipeline,
            profile_dir: self.profile_dir,
            extra_args: self.extra_args,
        })
    }
}

#[derive(Debug)]
pub enum BuildError {
    MissingField(&'static str),
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuildError::MissingField(field) => write!(f, "missing field: {}", field),
        }
    }
}

impl std::error::Error for BuildError {}

#[derive(Debug)]
pub enum RunError {
    Io(std::io::Error),
    Failed(i32, Vec<String>),
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RunError::Io(err) => write!(f, "io error: {}", err),
            RunError::Failed(code, argv) => write!(f, "command {:?} exited with {}", argv, code),
        }
    }
}

impl std::error::Error for RunError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RunError::Io(err) => Some(err),
            RunError::Failed(_, _) => None,
        }
    }
}

pub type RunResult = Result<Output, RunError>;

/// Execute `ogn_run` with the provided request and return the captured output.
pub fn run_local(request: &RunRequest, binary: Option<&str>) -> RunResult {
    let program = binary.unwrap_or("ogn_run");
    let mut cmd = Command::new(program);
    cmd.args(request.argv());
    let output = cmd.output().map_err(RunError::Io)?;
    if output.status.success() {
        Ok(output)
    } else {
        Err(RunError::Failed(
            output.status.code().unwrap_or(-1),
            request.argv(),
        ))
    }
}

/// Pretty command-line string (for logging or prompts).
pub fn format_command(request: &RunRequest, binary: Option<&str>) -> String {
    let program = binary.unwrap_or("ogn_run");
    let mut parts = vec![program.to_string()];
    parts.extend(request.argv());
    parts
        .into_iter()
        .map(|part| shell_words::quote(&part))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Submit a run to the OgnControl service via `grpcurl`.
pub fn submit_run_grpc(
    host: &str,
    tenant: &str,
    fastqs: &[&str],
    reference: &str,
    sample: &str,
) -> anyhow::Result<String> {
    let grpcurl = std::env::var("GRPCURL_BIN").unwrap_or_else(|_| "grpcurl".to_string());
    if which::which(&grpcurl).is_err() {
        return Err(anyhow!(
            "grpcurl not found (looked for '{}'). Set GRPCURL_BIN or install grpcurl.",
            grpcurl
        ));
    }

    let payload = json!({
        "tenant_id": tenant,
        "sample_id": sample,
        "fastq_uris": fastqs,
        "reference_uri": reference,
        "pipeline_json": "{}"
    })
    .to_string();

    let output = Command::new(&grpcurl)
        .args(["-d", &payload, host, "ogn.OgnControl/SubmitRun"])
        .output()?;
    ensure!(
        output.status.success(),
        "grpcurl failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn submit_helper_handles_missing_grpcurl() {
        let result = submit_run_grpc(
            "localhost:50051",
            "demo",
            &["s3://demo/R1.fq"],
            "s3://refs/ref.fa",
            "sample",
        );
        if let Err(err) = result {
            let msg = err.to_string();
            assert!(
                msg.contains("grpcurl not found") || msg.contains("connect"),
                "unexpected error: {msg}"
            );
        }
    }
}

/// Ensure shell escaping without taking a dependency at runtime.
mod shell_words {
    pub fn quote(input: &str) -> String {
        if input.is_empty() {
            return "''".to_string();
        }
        if input
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "-_.@/".contains(c))
        {
            return input.to_string();
        }
        format!("'{}'", input.replace('\'', "'\\''"))
    }
}
