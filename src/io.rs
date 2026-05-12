use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RunConfig {
    pub run: RunSection,
    pub parameters: ParametersSection,
    pub decoder: DecoderSection,
    pub profile: ProfileSection,
    pub state: StateSection,
    pub binning: BinningSection,
    pub sampling: SamplingSection,
    pub metrics: MetricsSection,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RunSection {
    pub run_id: String,
    pub seed: u64,
    pub mode: String,
    pub output_dir: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ParametersSection {
    pub r: usize,
    pub d: usize,
    pub t: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DecoderSection {
    pub iterations: usize,
    pub thresholds: Vec<usize>,
    pub wrong_zero_is_failure: bool,
    pub flip_policy: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ProfileSection {
    pub u_values: Vec<usize>,
    pub gathering_m_max: usize,
    pub gathering_l_max: usize,
    pub fixedpoint_m_max: usize,
    pub fixedpoint_thresholds: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StateSection {
    pub models: Vec<String>,
    pub u_bins: Vec<usize>,
    pub omega_bins: Vec<usize>,
    pub strict_absorbing: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BinningSection {
    pub bin_budgets: Vec<usize>,
    pub quantile_ties: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SamplingSection {
    pub key_count: usize,
    pub error_sample_count: usize,
    pub train_fraction: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MetricsSection {
    pub q_values: Vec<usize>,
    pub top_tail_fractions: Vec<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Manifest {
    pub run_id: String,
    pub binary_version: String,
    pub docker_image_tag: String,
    pub command: String,
    pub config_hash: String,
    pub git_commit: Option<String>,
    pub start_timestamp_utc: String,
    pub end_timestamp_utc: Option<String>,
    pub seed: u64,
    pub keys_processed: u64,
    pub errors_processed: u64,
}

pub fn load_config(path: &Path) -> Result<RunConfig> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read config {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("failed to parse config {}", path.display()))
}

pub fn file_sha256(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    Ok(hex_sha256(&bytes))
}

pub fn hex_sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

pub fn ensure_artifact_layout(out_dir: &Path) -> Result<()> {
    for subdir in [
        out_dir.to_path_buf(),
        out_dir.join("configs"),
        out_dir.join("csv"),
        out_dir.join("json"),
        out_dir.join("plots"),
    ] {
        fs::create_dir_all(&subdir)
            .with_context(|| format!("failed to create {}", subdir.display()))?;
    }
    Ok(())
}

pub fn copy_resolved_config(config_path: &Path, out_dir: &Path) -> Result<PathBuf> {
    let dest = out_dir.join("configs").join("config.resolved.toml");
    fs::copy(config_path, &dest).with_context(|| {
        format!(
            "failed to copy resolved config from {} to {}",
            config_path.display(),
            dest.display()
        )
    })?;
    Ok(dest)
}

pub fn manifest_path(out_dir: &Path) -> PathBuf {
    out_dir.join("manifest.json")
}

pub fn write_manifest(path: &Path, manifest: &Manifest) -> Result<()> {
    let payload = serde_json::to_vec_pretty(manifest)?;
    fs::write(path, payload).with_context(|| format!("failed to write {}", path.display()))
}

pub fn write_string(path: &Path, contents: &str) -> Result<()> {
    fs::write(path, contents).with_context(|| format!("failed to write {}", path.display()))
}

pub fn detect_git_commit() -> Option<String> {
    let head = fs::read_to_string(".git/HEAD").ok()?;
    if let Some(ref_name) = head.strip_prefix("ref: ").map(str::trim) {
        fs::read_to_string(Path::new(".git").join(ref_name))
            .ok()
            .map(|s| s.trim().to_string())
    } else {
        Some(head.trim().to_string())
    }
}

pub fn now_utc() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true)
}

pub fn default_image_tag() -> String {
    env::var("IMAGE_TAG").unwrap_or_else(|_| "dependency-aware-dfrcert:v01".to_string())
}
