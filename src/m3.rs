use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::binning::{build_bins_for_model, model_score};
use crate::bitset::Mask256;
use crate::combinatorics::{binom_u128, iter_combinations};
use crate::decoder::{decode_with_logs, syndrome_for_error, DecodeResult, Outcome};
use crate::io::{default_image_tag, detect_git_commit, file_sha256, now_utc, write_manifest, DecoderSection, Manifest, ProfileSection};
use crate::key::{build_key_runtime, enumerate_all_keys, sample_keys_uniform, Key};
use crate::profile::{compute_profiles, ProfileRecord};
use crate::progress::ProgressReporter;

const TAU_GRID: [f64; 8] = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.99];
const FAIL_Q_GRID: [usize; 7] = [1, 2, 5, 10, 25, 50, 100];

#[derive(Clone, Debug, Deserialize)]
struct M3ExperimentConfig {
    run: M3RunSection,
    parameters: M3ExperimentParameters,
}

#[derive(Clone, Debug, Deserialize)]
struct M3ThresholdSweepConfig {
    run: M3RunSection,
    parameters: M3ThresholdSweepParameters,
}

#[derive(Clone, Debug, Deserialize)]
struct M3RunSection {
    run_id: String,
    seed: u64,
}

#[derive(Clone, Debug, Deserialize)]
struct M3ExperimentParameters {
    r: usize,
    d: usize,
    t: usize,
    key_mode: String,
    key_count: usize,
    thresholds: Vec<usize>,
    wrong_zero_is_failure: bool,
    exact_errors: bool,
    error_sample_count: Option<usize>,
    divergence_flip_threshold: usize,
    divergence_residual_threshold: usize,
    baseline_bin_budget: Option<usize>,
    quantile_bin_count: Option<usize>,
    top_fraction: Option<f64>,
    notes: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct M3ThresholdSweepParameters {
    r: usize,
    d: usize,
    t: usize,
    key_mode: String,
    key_count: usize,
    schedules: Vec<Vec<usize>>,
    wrong_zero_is_failure: bool,
    exact_errors: bool,
    error_sample_count: Option<usize>,
    divergence_flip_threshold: usize,
    divergence_residual_threshold: usize,
    preferred_mean_min: Option<f64>,
    preferred_mean_max: Option<f64>,
    acceptable_mean_min: Option<f64>,
    acceptable_mean_max: Option<f64>,
}

#[derive(Clone, Debug)]
struct KeyAggregate {
    key: Key,
    profile: ProfileRecord,
    p_h: f64,
    failure_count: usize,
    error_count: usize,
    timeout_count: usize,
    wrong_zero_count: usize,
    nonzero_halt_count: usize,
    divergence_like_count: usize,
    fp0_exact_or_proxy: f64,
    fn0_exact_or_proxy: f64,
    max_clean_counter: usize,
    min_true_counter: usize,
    mean_clean_counter: f64,
    mean_true_counter: f64,
    fp0_bin: usize,
    fn0_bin: usize,
    reach_state_count: Option<usize>,
    reach_successor_count: Option<usize>,
    dominant_failure_type: String,
}

#[derive(Clone, Debug)]
struct ModelArtifacts {
    score_map: BTreeMap<u64, f64>,
    bin_map: BTreeMap<u64, usize>,
    bin_label_map: BTreeMap<usize, String>,
}

#[derive(Clone, Debug)]
struct TupleAssignment {
    tuple_text: String,
    tuple_score: f64,
    tuple_bin: usize,
}

#[derive(Clone, Debug, Serialize)]
struct CertificateSummaryRow {
    run_id: String,
    model: String,
    empirical_delta_avg: f64,
    direct_profile_bound: f64,
    direct_profile_looseness: String,
    number_of_bins: usize,
    nonempty_bins: usize,
    max_bin_mass: f64,
    average_bin_mass: f64,
    max_p_phi_max: f64,
    mean_p_phi_max: f64,
    mean_p_phi_mean: f64,
}

#[derive(Clone, Debug)]
struct BinSummaryRow {
    run_id: String,
    model: String,
    bin_id: usize,
    bin_label: String,
    key_count: usize,
    bin_mass: f64,
    p_phi_mean: f64,
    p_phi_max: f64,
}

#[derive(Clone, Debug, Serialize)]
struct RankingSummaryRow {
    run_id: String,
    model: String,
    key_count: usize,
    spearman_with_p_h: f64,
    kendall_with_p_h: f64,
    auc_bad_any: f64,
    auc_top10_label: f64,
    tail_l1_error: f64,
    bad_top10_recall: f64,
    bad_top10_precision: f64,
    direct_profile_bound: f64,
    direct_profile_looseness: String,
    comments: String,
}

#[derive(Clone, Debug, Serialize)]
struct BadKeySummaryRow {
    run_id: String,
    model: String,
    top_k: usize,
    highest_bin: usize,
    bad_any_recall_at_highest_bin: f64,
    bad_high_recall_at_highest_bin: f64,
    bad_any_precision_at_highest_bin: f64,
    bad_high_precision_at_highest_bin: f64,
    bad_top10_recall: f64,
    bad_top10_precision: f64,
}

#[derive(Clone, Debug, Serialize)]
struct StaticKeyBoundsRow {
    run_id: String,
    model: String,
    q: usize,
    empirical_fail_q: f64,
    profile_bound_fail_q: f64,
    average_union_bound_min_1: f64,
    profile_bound_looseness: String,
    improvement_vs_average_union_bound: f64,
}

#[derive(Clone, Debug, Serialize)]
struct RunMetadata {
    run_id: String,
    r: usize,
    d: usize,
    t: usize,
    threshold_schedule: String,
    key_count: usize,
    error_count_per_key: usize,
    exact_errors: bool,
    quantile_bin_count: usize,
    baseline_bin_budget: usize,
    m3_cert_primary_feature: String,
    notes: String,
}

#[derive(Clone, Debug, Serialize)]
struct ThresholdSweepRow {
    run_id: String,
    r: usize,
    d: usize,
    t: usize,
    key_count: usize,
    error_count_per_key: usize,
    threshold_schedule: String,
    zero_error_pass_rate: f64,
    single_bit_geometry_pass_rate: f64,
    syndrome_update_pass_rate: f64,
    p_h_mean: f64,
    p_h_min: f64,
    p_h_median: f64,
    p_h_max: f64,
    success_rate: f64,
    timeout_rate: f64,
    wrong_zero_rate: f64,
    nonzero_halt_rate: f64,
    selected: bool,
}

pub fn run_m3_experiment_command(config_path: &PathBuf, out_dir: &PathBuf) -> Result<()> {
    let config = load_m3_experiment_config(config_path)?;
    prepare_layout(out_dir)?;
    fs::copy(config_path, out_dir.join("configs").join("config.resolved.toml"))
        .with_context(|| format!("failed to copy {}", config_path.display()))?;

    eprintln!("[progress] m3 stage 1/5 selecting keys");
    let keys = select_keys(&config.parameters.key_mode, config.parameters.r, config.parameters.d, config.parameters.key_count, config.run.seed)?;
    eprintln!("[progress] m3 stage 1/5 complete: {} keys selected", keys.len());
    eprintln!("[progress] m3 stage 2/5 generating errors");
    let errors = generate_error_masks(
        config.parameters.r,
        config.parameters.t,
        config.parameters.exact_errors,
        config.parameters.error_sample_count.unwrap_or(5_000),
        config.run.seed,
    )?;
    eprintln!("[progress] m3 stage 2/5 complete: {} errors per key", errors.len());

    let decoder = DecoderSection {
        iterations: config.parameters.thresholds.len(),
        thresholds: config.parameters.thresholds.clone(),
        wrong_zero_is_failure: config.parameters.wrong_zero_is_failure,
        flip_policy: "all_ge_threshold".to_string(),
    };
    let threshold_label = schedule_to_string(&config.parameters.thresholds);
    let profile_config = default_profile_config(config.parameters.d, config.parameters.t, &config.parameters.thresholds);

    eprintln!("[progress] m3 stage 3/5 computing profiles");
    let profiles = compute_profiles(&keys, config.parameters.t, &profile_config)?;
    eprintln!("[progress] m3 stage 3/5 complete");

    eprintln!("[progress] m3 stage 4/5 exact/sample decoding with streamed taxonomy");
    let aggregates = compute_key_aggregates(
        &config.run.run_id,
        &keys,
        &profiles,
        &decoder,
        &errors,
        config.parameters.t,
        config.parameters.divergence_flip_threshold,
        config.parameters.divergence_residual_threshold,
        out_dir,
    )?;
    eprintln!("[progress] m3 stage 4/5 complete");

    eprintln!("[progress] m3 stage 5/5 building M3 models and summaries");
    let summaries = build_and_write_summaries(
        &config.run.run_id,
        &aggregates,
        &threshold_label,
        config.parameters.t,
        config.parameters.baseline_bin_budget.unwrap_or(8),
        config.parameters.quantile_bin_count.unwrap_or(4),
        config.parameters.top_fraction.unwrap_or(0.10),
        out_dir,
        config.parameters.notes.clone().unwrap_or_default(),
        config.parameters.exact_errors,
    )?;
    eprintln!("[progress] m3 stage 5/5 complete");

    let manifest = Manifest {
        run_id: config.run.run_id.clone(),
        binary_version: env!("CARGO_PKG_VERSION").to_string(),
        docker_image_tag: default_image_tag(),
        command: format!("dfrcert m3-experiment --config {}", config_path.display()),
        config_hash: file_sha256(config_path)?,
        git_commit: detect_git_commit(),
        start_timestamp_utc: now_utc(),
        end_timestamp_utc: None,
        seed: config.run.seed,
        keys_processed: keys.len() as u64,
        errors_processed: (keys.len() * errors.len()) as u64,
    };
    write_manifest(&out_dir.join("manifest.json"), &manifest)?;
    write_run_metadata_json(&summaries.run_metadata, out_dir)?;
    println!("wrote new M3 artifacts into {}", out_dir.display());
    Ok(())
}

pub fn run_m3_threshold_sweep_command(config_path: &PathBuf, out_dir: &PathBuf) -> Result<()> {
    let config = load_m3_threshold_sweep_config(config_path)?;
    prepare_layout(out_dir)?;
    fs::copy(config_path, out_dir.join("configs").join("config.resolved.toml"))
        .with_context(|| format!("failed to copy {}", config_path.display()))?;

    let keys = select_keys(&config.parameters.key_mode, config.parameters.r, config.parameters.d, config.parameters.key_count, config.run.seed)?;
    let errors = generate_error_masks(
        config.parameters.r,
        config.parameters.t,
        config.parameters.exact_errors,
        config.parameters.error_sample_count.unwrap_or(5_000),
        config.run.seed,
    )?;
    let zero_error_pass_rate = zero_error_pass_rate(&keys, &config.parameters)?;
    let single_bit_geometry_pass_rate = single_bit_geometry_pass_rate(&keys, &config.parameters)?;
    let syndrome_update_pass_rate = syndrome_update_pass_rate(&keys, &config.parameters)?;
    let progress = ProgressReporter::new("m3 threshold sweep schedules", config.parameters.schedules.len());
    let mut rows = Vec::new();
    for (index, schedule) in config.parameters.schedules.iter().enumerate() {
        let decoder = DecoderSection {
            iterations: schedule.len(),
            thresholds: schedule.clone(),
            wrong_zero_is_failure: config.parameters.wrong_zero_is_failure,
            flip_policy: "all_ge_threshold".to_string(),
        };
        let stats = compute_schedule_stats(
            &keys,
            &decoder,
            &errors,
            config.parameters.t,
            config.parameters.divergence_flip_threshold,
            config.parameters.divergence_residual_threshold,
        )?;
        rows.push(ThresholdSweepRow {
            run_id: config.run.run_id.clone(),
            r: config.parameters.r,
            d: config.parameters.d,
            t: config.parameters.t,
            key_count: keys.len(),
            error_count_per_key: errors.len(),
            threshold_schedule: schedule_to_string(schedule),
            zero_error_pass_rate,
            single_bit_geometry_pass_rate,
            syndrome_update_pass_rate,
            p_h_mean: stats.p_h_mean,
            p_h_min: stats.p_h_min,
            p_h_median: stats.p_h_median,
            p_h_max: stats.p_h_max,
            success_rate: stats.success_rate,
            timeout_rate: stats.timeout_rate,
            wrong_zero_rate: stats.wrong_zero_rate,
            nonzero_halt_rate: stats.nonzero_halt_rate,
            selected: false,
        });
        let _ = index;
        progress.tick();
    }
    progress.finish();
    let selected_index = select_sweep_row(
        &rows,
        config.parameters.preferred_mean_min.unwrap_or(0.02),
        config.parameters.preferred_mean_max.unwrap_or(0.5),
        config.parameters.acceptable_mean_min.unwrap_or(0.005),
        config.parameters.acceptable_mean_max.unwrap_or(0.7),
    );
    if let Some(index) = selected_index {
        rows[index].selected = true;
    }
    write_threshold_sweep_csv(&rows, out_dir)?;
    write_threshold_sweep_summary(&rows, out_dir)?;
    let manifest = Manifest {
        run_id: config.run.run_id.clone(),
        binary_version: env!("CARGO_PKG_VERSION").to_string(),
        docker_image_tag: default_image_tag(),
        command: format!("dfrcert m3-threshold-sweep --config {}", config_path.display()),
        config_hash: file_sha256(config_path)?,
        git_commit: detect_git_commit(),
        start_timestamp_utc: now_utc(),
        end_timestamp_utc: None,
        seed: config.run.seed,
        keys_processed: keys.len() as u64,
        errors_processed: (keys.len() * errors.len() * config.parameters.schedules.len()) as u64,
    };
    write_manifest(&out_dir.join("manifest.json"), &manifest)?;
    println!("wrote threshold sweep artifacts into {}", out_dir.display());
    Ok(())
}

#[derive(Clone, Debug)]
struct ScheduleStats {
    p_h_mean: f64,
    p_h_min: f64,
    p_h_median: f64,
    p_h_max: f64,
    success_rate: f64,
    timeout_rate: f64,
    wrong_zero_rate: f64,
    nonzero_halt_rate: f64,
}

#[derive(Clone, Debug)]
struct SummaryOutputs {
    run_metadata: RunMetadata,
}

fn load_m3_experiment_config(path: &Path) -> Result<M3ExperimentConfig> {
    let text = fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))
}

fn load_m3_threshold_sweep_config(path: &Path) -> Result<M3ThresholdSweepConfig> {
    let text = fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))
}

fn prepare_layout(out_dir: &Path) -> Result<()> {
    fs::create_dir_all(out_dir).with_context(|| format!("failed to create {}", out_dir.display()))?;
    fs::create_dir_all(out_dir.join("configs")).with_context(|| format!("failed to create {}", out_dir.join("configs").display()))?;
    Ok(())
}

fn select_keys(key_mode: &str, r: usize, d: usize, key_count: usize, seed: u64) -> Result<Vec<Key>> {
    let mut keys = match key_mode {
        "exact_full_keyspace" => enumerate_all_keys(r, d)?,
        "sampled_keys" => sample_keys_uniform(r, d, key_count.max(1), seed),
        other => bail!("unsupported key_mode '{other}'"),
    };
    if keys.len() > key_count && key_count > 0 {
        keys.truncate(key_count);
    }
    Ok(keys)
}

fn generate_error_masks(
    r: usize,
    t: usize,
    exact_errors: bool,
    error_sample_count: usize,
    seed: u64,
) -> Result<Vec<Mask256>> {
    let bit_count = 2 * r;
    if t == 0 {
        return Ok(vec![Mask256::empty()]);
    }
    if exact_errors {
        return iter_combinations(bit_count, t)?
            .map(mask_from_combo)
            .collect::<Result<Vec<_>>>();
    }
    sample_error_masks(bit_count, t, error_sample_count.max(1), seed)
}

fn mask_from_combo(combo: Vec<usize>) -> Result<Mask256> {
    let mut mask = Mask256::empty();
    for bit in combo {
        mask.set(bit)?;
    }
    Ok(mask)
}

fn sample_error_masks(bit_count: usize, t: usize, target: usize, seed: u64) -> Result<Vec<Mask256>> {
    let total = binom_u128(bit_count as u32, t as u32);
    if total <= target as u128 {
        return iter_combinations(bit_count, t)?
            .map(mask_from_combo)
            .collect::<Result<Vec<_>>>();
    }
    let mut rng = XorShift64::new(seed);
    let mut seen = HashSet::new();
    let mut out = Vec::with_capacity(target);
    let max_attempts = target.saturating_mul(50).max(1000);
    for _ in 0..max_attempts {
        if out.len() >= target {
            break;
        }
        let mut values = (0..bit_count).collect::<Vec<_>>();
        for index in 0..t {
            let swap_index = index + rng.gen_range(bit_count - index);
            values.swap(index, swap_index);
        }
        values[..t].sort_unstable();
        let key = values[..t]
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("-");
        if !seen.insert(key) {
            continue;
        }
        out.push(mask_from_combo(values[..t].to_vec())?);
    }
    if out.is_empty() {
        bail!("failed to sample any error masks");
    }
    Ok(out)
}

fn default_profile_config(d: usize, t: usize, thresholds: &[usize]) -> ProfileSection {
    let mut fixed_thresholds = thresholds.to_vec();
    fixed_thresholds.sort_unstable();
    fixed_thresholds.dedup();
    ProfileSection {
        u_values: (1..=t.max(1).min(4)).collect(),
        gathering_m_max: d.min(3),
        gathering_l_max: d.min(3),
        fixedpoint_m_max: d.min(3),
        fixedpoint_thresholds: fixed_thresholds,
    }
}

fn compute_key_aggregates(
    run_id: &str,
    keys: &[Key],
    profiles: &[ProfileRecord],
    decoder: &DecoderSection,
    errors: &[Mask256],
    t: usize,
    divergence_flip_threshold: usize,
    divergence_residual_threshold: usize,
    out_dir: &Path,
) -> Result<Vec<KeyAggregate>> {
    let path = out_dir.join("failure_taxonomy.csv");
    let file = File::create(&path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "run_id,key_id,error_id,threshold_schedule,outcome,iterations_used,initial_syndrome_weight,final_syndrome_weight,final_residual_weight,flipped_total,wrong_zero,fp0_event,fn0_event,max_clean_counter0,min_true_counter0"
    )?;
    let mut profile_map = profiles
        .iter()
        .map(|profile| (profile.key_id, profile.clone()))
        .collect::<BTreeMap<_, _>>();
    let progress = ProgressReporter::new(format!("m3 exact {}", run_id), keys.len());
    let threshold0 = decoder.thresholds.first().copied().unwrap_or(0);
    let threshold_label = schedule_to_string(&decoder.thresholds);
    let bit_count = keys.first().map(|key| 2 * key.r).unwrap_or(0);
    let mut out = Vec::with_capacity(keys.len());
    for key in keys {
        let runtime = build_key_runtime(key.clone())?;
        let profile = profile_map
            .remove(&key.key_id)
            .with_context(|| format!("missing profile for key {}", key.key_id))?;
        let mut failures = 0usize;
        let mut timeout_count = 0usize;
        let mut wrong_zero_count = 0usize;
        let mut nonzero_halt_count = 0usize;
        let mut divergence_like_count = 0usize;
        let mut fp0_events = 0usize;
        let mut fn0_events = 0usize;
        let mut max_clean_counter = 0usize;
        let mut min_true_counter = usize::MAX;
        let mut sum_clean_counter = 0u128;
        let mut sum_true_counter = 0u128;
        let mut clean_counter_obs = 0u128;
        let mut true_counter_obs = 0u128;
        let mut outcome_counts = BTreeMap::<String, usize>::new();

        for (error_id, error) in errors.iter().copied().enumerate() {
            let initial_syndrome = syndrome_for_error(&runtime, error)?;
            let mut local_max_clean = 0usize;
            let mut local_min_true = usize::MAX;
            let mut fp0_event = false;
            let mut fn0_event = false;
            for bit in 0..bit_count {
                let counter = runtime.column_masks[bit].and(initial_syndrome).popcount() as usize;
                if error.contains(bit) {
                    local_min_true = local_min_true.min(counter);
                    true_counter_obs += 1;
                    sum_true_counter += counter as u128;
                    if counter < threshold0 {
                        fn0_event = true;
                    }
                } else {
                    local_max_clean = local_max_clean.max(counter);
                    clean_counter_obs += 1;
                    sum_clean_counter += counter as u128;
                    if counter >= threshold0 {
                        fp0_event = true;
                    }
                }
            }
            if fp0_event {
                fp0_events += 1;
            }
            if fn0_event {
                fn0_events += 1;
            }
            max_clean_counter = max_clean_counter.max(local_max_clean);
            if local_min_true != usize::MAX {
                min_true_counter = min_true_counter.min(local_min_true);
            }

            let result = decode_with_logs(&runtime, error, decoder)?;
            let outcome = classify_outcome(&result, divergence_flip_threshold, divergence_residual_threshold, t);
            *outcome_counts.entry(outcome.clone()).or_insert(0) += 1;
            if outcome == "divergence_like" {
                divergence_like_count += 1;
            }
            match result.outcome {
                Outcome::Success => {}
                Outcome::Timeout => {
                    failures += 1;
                    timeout_count += 1;
                }
                Outcome::WrongZero => {
                    failures += 1;
                    wrong_zero_count += 1;
                }
                Outcome::NonzeroHalt => {
                    failures += 1;
                    nonzero_halt_count += 1;
                }
            }
            writeln!(
                writer,
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                run_id,
                key.key_id,
                error_id,
                threshold_label,
                outcome,
                result.iterations,
                initial_syndrome.popcount(),
                result.final_syndrome_weight,
                result.final_residual_weight,
                result.logs.iter().map(|log| u32::from(log.flip_count)).sum::<u32>(),
                result.outcome == Outcome::WrongZero,
                fp0_event,
                fn0_event,
                local_max_clean,
                if local_min_true == usize::MAX { 0 } else { local_min_true }
            )?;
        }
        let error_count = errors.len().max(1);
        let p_h = failures as f64 / error_count as f64;
        let dominant_failure_type = outcome_counts
            .into_iter()
            .filter(|(label, _)| label != "success")
            .max_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)))
            .map(|(label, _)| label)
            .unwrap_or_else(|| "success".to_string());
        out.push(KeyAggregate {
            key: key.clone(),
            profile,
            p_h,
            failure_count: failures,
            error_count,
            timeout_count,
            wrong_zero_count,
            nonzero_halt_count,
            divergence_like_count,
            fp0_exact_or_proxy: fp0_events as f64 / error_count as f64,
            fn0_exact_or_proxy: fn0_events as f64 / error_count as f64,
            max_clean_counter,
            min_true_counter: if min_true_counter == usize::MAX { 0 } else { min_true_counter },
            mean_clean_counter: if clean_counter_obs == 0 { 0.0 } else { sum_clean_counter as f64 / clean_counter_obs as f64 },
            mean_true_counter: if true_counter_obs == 0 { 0.0 } else { sum_true_counter as f64 / true_counter_obs as f64 },
            fp0_bin: 0,
            fn0_bin: 0,
            reach_state_count: None,
            reach_successor_count: None,
            dominant_failure_type,
        });
        progress.tick();
    }
    progress.finish();
    writer.flush()?;
    Ok(out)
}

fn build_and_write_summaries(
    run_id: &str,
    aggregates: &[KeyAggregate],
    threshold_label: &str,
    t: usize,
    baseline_bin_budget: usize,
    quantile_bin_count: usize,
    top_fraction: f64,
    out_dir: &Path,
    notes: String,
    exact_errors: bool,
) -> Result<SummaryOutputs> {
    let mut rows = aggregates.to_vec();
    let fp0_bins = quantize_rows(&rows, quantile_bin_count, |row| row.fp0_exact_or_proxy, false);
    let fn0_bins = quantize_rows(&rows, quantile_bin_count, |row| row.fn0_exact_or_proxy, false);
    let max_clean_bins = quantize_rows(&rows, quantile_bin_count, |row| row.max_clean_counter as f64, false);
    let min_true_bins = quantize_rows(&rows, quantile_bin_count, |row| row.min_true_counter as f64, true);
    let mean_clean_bins = quantize_rows(&rows, quantile_bin_count, |row| row.mean_clean_counter, false);
    let mean_true_bins = quantize_rows(&rows, quantile_bin_count, |row| row.mean_true_counter, true);
    let c4_bins = quantize_rows(&rows, quantile_bin_count, |row| row.profile.c4_loc as f64, false);
    let lambda3_bins = quantize_rows(&rows, quantile_bin_count, |row| lambda_ge_count(&row.profile.lambda_json, 3) as f64, false);
    let omega_bins = quantize_rows(&rows, quantile_bin_count, |row| row.profile.omega_pair_max as f64, false);
    let b_fixed_bins = quantize_rows(&rows, quantile_bin_count, |row| row.profile.b_fixed as f64, false);
    let near_tail_high_bins = quantize_rows(&rows, quantile_bin_count, |row| near_tail_high(&row.profile.an_u_json), false);

    for row in &mut rows {
        row.fp0_bin = *fp0_bins.get(&row.key.key_id).unwrap_or(&0);
        row.fn0_bin = *fn0_bins.get(&row.key.key_id).unwrap_or(&0);
    }

    let profiles = rows.iter().map(|row| row.profile.clone()).collect::<Vec<_>>();
    let m0 = build_baseline_model("M0", baseline_bin_budget, &profiles)?;
    let m1 = build_baseline_model("M1", baseline_bin_budget, &profiles)?;
    let m2 = build_baseline_model("M2", baseline_bin_budget, &profiles)?;
    let m3_old = build_baseline_model("M3_old", baseline_bin_budget, &profiles)?;
    let exact_map = rows.iter().map(|row| (row.key.key_id, row.p_h)).collect::<BTreeMap<_, _>>();

    let m3_rank = build_tuple_model(
        &rows,
        "M3_rank",
        rows.iter().map(|row| {
            let tuple = if t == 1 {
                vec![
                    row.profile.d_max as usize,
                    *omega_bins.get(&row.key.key_id).unwrap_or(&0),
                    *max_clean_bins.get(&row.key.key_id).unwrap_or(&0),
                    *lambda3_bins.get(&row.key.key_id).unwrap_or(&0),
                    *c4_bins.get(&row.key.key_id).unwrap_or(&0),
                    *b_fixed_bins.get(&row.key.key_id).unwrap_or(&0),
                ]
            } else if t == 2 {
                vec![
                    *c4_bins.get(&row.key.key_id).unwrap_or(&0),
                    *lambda3_bins.get(&row.key.key_id).unwrap_or(&0),
                    row.profile.d_max as usize,
                    *omega_bins.get(&row.key.key_id).unwrap_or(&0),
                    *fp0_bins.get(&row.key.key_id).unwrap_or(&0),
                    *fn0_bins.get(&row.key.key_id).unwrap_or(&0),
                    *b_fixed_bins.get(&row.key.key_id).unwrap_or(&0),
                ]
            } else {
                vec![
                    *c4_bins.get(&row.key.key_id).unwrap_or(&0),
                    *fp0_bins.get(&row.key.key_id).unwrap_or(&0),
                    *fn0_bins.get(&row.key.key_id).unwrap_or(&0),
                    *omega_bins.get(&row.key.key_id).unwrap_or(&0),
                    row.profile.d_max as usize,
                    *near_tail_high_bins.get(&row.key.key_id).unwrap_or(&0),
                    *b_fixed_bins.get(&row.key.key_id).unwrap_or(&0),
                ]
            };
            (row.key.key_id, tuple)
        }),
    );

    let primary_tail_source = select_primary_tail_source(t, &exact_map, &m1, &m2, &fp0_bins);
    let m3_cert = build_tuple_model(
        &rows,
        "M3_cert",
        rows.iter().map(|row| {
            let primary_tail_bin = match primary_tail_source.as_str() {
                "M1" => *m1.bin_map.get(&row.key.key_id).unwrap_or(&0),
                "M2" => *m2.bin_map.get(&row.key.key_id).unwrap_or(&0),
                "fp0_proxy" => *fp0_bins.get(&row.key.key_id).unwrap_or(&0),
                _ => *m2.bin_map.get(&row.key.key_id).unwrap_or(&0),
            };
            (
                row.key.key_id,
                vec![
                    primary_tail_bin,
                    *c4_bins.get(&row.key.key_id).unwrap_or(&0),
                    row.profile.d_max as usize,
                    *omega_bins.get(&row.key.key_id).unwrap_or(&0),
                    *fp0_bins.get(&row.key.key_id).unwrap_or(&0),
                    *fn0_bins.get(&row.key.key_id).unwrap_or(&0),
                    *b_fixed_bins.get(&row.key.key_id).unwrap_or(&0),
                ],
            )
        }),
    );

    let m3_counter = build_tuple_model(
        &rows,
        "M3_counter",
        rows.iter().map(|row| {
            (
                row.key.key_id,
                vec![
                    *fp0_bins.get(&row.key.key_id).unwrap_or(&0),
                    *fn0_bins.get(&row.key.key_id).unwrap_or(&0),
                    *max_clean_bins.get(&row.key.key_id).unwrap_or(&0),
                    *min_true_bins.get(&row.key.key_id).unwrap_or(&0),
                    *mean_clean_bins.get(&row.key.key_id).unwrap_or(&0),
                    *mean_true_bins.get(&row.key.key_id).unwrap_or(&0),
                ],
            )
        }),
    );

    let models = vec![
        ("M0".to_string(), m0),
        ("M1".to_string(), m1),
        ("M2".to_string(), m2.clone()),
        ("M3_old".to_string(), m3_old),
        ("M3_rank".to_string(), m3_rank),
        ("M3_cert".to_string(), m3_cert),
        ("M3_counter".to_string(), m3_counter),
    ];
    write_per_key_profiles_csv(run_id, &rows, out_dir)?;
    write_per_key_m3_csv(
        run_id,
        &rows,
        &models,
        &fp0_bins,
        &fn0_bins,
        &primary_tail_source,
        out_dir,
    )?;

    let mut ranking_rows = Vec::new();
    let mut certificate_rows = Vec::new();
    let mut bad_key_rows = Vec::new();
    let mut static_key_rows = Vec::new();
    let mut bin_summary_rows = Vec::new();
    for (model_name, artifacts) in &models {
        let bin_rows = build_bin_summary_rows(run_id, model_name, &artifacts.bin_map, &artifacts.bin_label_map, &exact_map);
        let cert_row = build_certificate_row(run_id, model_name, &bin_rows, &exact_map);
        let ranking_row = build_ranking_row(run_id, model_name, artifacts, &bin_rows, &exact_map, top_fraction, threshold_label);
        let bad_key_row = build_bad_key_row(run_id, model_name, artifacts, &exact_map, top_fraction);
        ranking_rows.push(ranking_row);
        certificate_rows.push(cert_row);
        bad_key_rows.push(bad_key_row);
        static_key_rows.extend(build_static_key_rows(run_id, model_name, &bin_rows, &exact_map));
        bin_summary_rows.extend(bin_rows);
    }
    write_model_ranking_summary_csv(&ranking_rows, out_dir)?;
    write_certificate_bound_summary_csv(&certificate_rows, out_dir)?;
    write_static_key_bounds_csv(&static_key_rows, out_dir)?;
    write_bad_key_summary_csv(&bad_key_rows, out_dir)?;
    write_bin_summary_csv(&bin_summary_rows, out_dir)?;
    write_worst_key_mechanisms_csv(run_id, &rows, &m2, &models, t, out_dir)?;
    write_experiment_summary_md(
        run_id,
        &ranking_rows,
        &certificate_rows,
        &bad_key_rows,
        &rows,
        &primary_tail_source,
        out_dir,
    )?;
    Ok(SummaryOutputs {
        run_metadata: RunMetadata {
            run_id: run_id.to_string(),
            r: rows.first().map(|row| row.key.r).unwrap_or(0),
            d: rows.first().map(|row| row.key.d).unwrap_or(0),
            t,
            threshold_schedule: threshold_label.to_string(),
            key_count: rows.len(),
            error_count_per_key: rows.first().map(|row| row.error_count).unwrap_or(0),
            exact_errors,
            quantile_bin_count,
            baseline_bin_budget,
            m3_cert_primary_feature: primary_tail_source,
            notes,
        },
    })
}

fn build_baseline_model(model: &str, bin_budget: usize, profiles: &[ProfileRecord]) -> Result<ModelArtifacts> {
    let bins = build_bins_for_model("m3_new", profiles, bin_budget.max(1), model)?;
    let score_map = profiles
        .iter()
        .map(|profile| (profile.key_id, model_score(model, profile)))
        .collect::<BTreeMap<_, _>>();
    let mut bin_map = BTreeMap::new();
    let mut bin_label_map = BTreeMap::new();
    for bin in bins {
        bin_label_map.insert(bin.bin_id, bin.descriptor_json);
        for key_id in bin.key_ids {
            bin_map.insert(key_id, bin.bin_id);
        }
    }
    Ok(ModelArtifacts {
        score_map,
        bin_map,
        bin_label_map,
    })
}

fn build_tuple_model<'a>(
    rows: &[KeyAggregate],
    model: &str,
    tuples: impl Iterator<Item = (u64, Vec<usize>)>,
) -> ModelArtifacts {
    let _ = rows;
    let mut keyed = tuples.collect::<Vec<_>>();
    keyed.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    let mut tuple_to_bin = BTreeMap::<Vec<usize>, usize>::new();
    let mut bin_label_map = BTreeMap::<usize, String>::new();
    let mut score_map = BTreeMap::new();
    let mut bin_map = BTreeMap::new();
    for (key_id, tuple) in keyed {
        let next_bin_id = tuple_to_bin.len();
        let bin_id = *tuple_to_bin.entry(tuple.clone()).or_insert_with(|| {
            bin_label_map.insert(next_bin_id, format_tuple_label(model, &tuple));
            next_bin_id
        });
        score_map.insert(key_id, bin_id as f64);
        bin_map.insert(key_id, bin_id);
    }
    ModelArtifacts {
        score_map,
        bin_map,
        bin_label_map,
    }
}

fn format_tuple_label(model: &str, tuple: &[usize]) -> String {
    let labels = match model {
        "M3_rank" => vec!["a", "b", "c", "d", "e", "f", "g"],
        "M3_cert" => vec!["tail", "cycle", "dmax", "omega", "fp", "fn", "b"],
        "M3_counter" => vec!["fp", "fn", "mc", "mt", "mean_c", "mean_t"],
        _ => vec!["x", "y", "z", "u", "v", "w", "q"],
    };
    tuple
        .iter()
        .enumerate()
        .map(|(index, value)| format!("{}={}", labels.get(index).copied().unwrap_or("x"), value))
        .collect::<Vec<_>>()
        .join("|")
}

fn quantize_rows(
    rows: &[KeyAggregate],
    bin_count: usize,
    value_fn: impl Fn(&KeyAggregate) -> f64,
    low_is_risk: bool,
) -> BTreeMap<u64, usize> {
    let mut keyed = rows
        .iter()
        .map(|row| {
            let value = value_fn(row);
            let risk_value = if low_is_risk { -value } else { value };
            (row.key.key_id, risk_value)
        })
        .collect::<Vec<_>>();
    keyed.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    let n = keyed.len().max(1);
    let bins = bin_count.max(1).min(n);
    keyed
        .into_iter()
        .enumerate()
        .map(|(index, (key_id, _))| (key_id, index * bins / n))
        .collect()
}

fn select_primary_tail_source(
    t: usize,
    exact_map: &BTreeMap<u64, f64>,
    m1: &ModelArtifacts,
    m2: &ModelArtifacts,
    fp0_bins: &BTreeMap<u64, usize>,
) -> String {
    if t == 1 {
        return "M2".to_string();
    }
    if t >= 3 {
        return if fp0_bins.is_empty() { "M1".to_string() } else { "fp0_proxy".to_string() };
    }
    let m1_scores = aligned_scores(&m1.score_map, exact_map);
    let m2_scores = aligned_scores(&m2.score_map, exact_map);
    if spearman_rank_corr(&m1_scores.0, &m1_scores.1) >= spearman_rank_corr(&m2_scores.0, &m2_scores.1) {
        "M1".to_string()
    } else {
        "M2".to_string()
    }
}

fn build_bin_summary_rows(
    run_id: &str,
    model: &str,
    bin_map: &BTreeMap<u64, usize>,
    bin_label_map: &BTreeMap<usize, String>,
    exact_map: &BTreeMap<u64, f64>,
) -> Vec<BinSummaryRow> {
    let mut grouped = BTreeMap::<usize, Vec<f64>>::new();
    for (key_id, p_h) in exact_map {
        grouped.entry(*bin_map.get(key_id).unwrap_or(&0)).or_default().push(*p_h);
    }
    let total = exact_map.len().max(1) as f64;
    grouped
        .into_iter()
        .map(|(bin_id, values)| BinSummaryRow {
            run_id: run_id.to_string(),
            model: model.to_string(),
            bin_id,
            bin_label: bin_label_map.get(&bin_id).cloned().unwrap_or_default(),
            key_count: values.len(),
            bin_mass: values.len() as f64 / total,
            p_phi_mean: mean_f64(&values),
            p_phi_max: values.iter().copied().max_by(|a, b| a.total_cmp(b)).unwrap_or(0.0),
        })
        .collect()
}

fn build_certificate_row(
    run_id: &str,
    model: &str,
    bin_rows: &[BinSummaryRow],
    exact_map: &BTreeMap<u64, f64>,
) -> CertificateSummaryRow {
    let empirical_delta_avg = mean_f64(&exact_map.values().copied().collect::<Vec<_>>());
    let direct_profile_bound = bin_rows.iter().map(|row| row.bin_mass * row.p_phi_max).sum::<f64>();
    let p_phi_max_values = bin_rows.iter().map(|row| row.p_phi_max).collect::<Vec<_>>();
    let p_phi_mean_values = bin_rows.iter().map(|row| row.p_phi_mean).collect::<Vec<_>>();
    CertificateSummaryRow {
        run_id: run_id.to_string(),
        model: model.to_string(),
        empirical_delta_avg,
        direct_profile_bound,
        direct_profile_looseness: format_optional_f64(safe_ratio(direct_profile_bound, empirical_delta_avg)),
        number_of_bins: bin_rows.len(),
        nonempty_bins: bin_rows.len(),
        max_bin_mass: bin_rows.iter().map(|row| row.bin_mass).fold(0.0_f64, f64::max),
        average_bin_mass: mean_f64(&bin_rows.iter().map(|row| row.bin_mass).collect::<Vec<_>>()),
        max_p_phi_max: p_phi_max_values.iter().copied().fold(0.0_f64, f64::max),
        mean_p_phi_max: mean_f64(&p_phi_max_values),
        mean_p_phi_mean: mean_f64(&p_phi_mean_values),
    }
}

fn build_ranking_row(
    run_id: &str,
    model: &str,
    artifacts: &ModelArtifacts,
    bin_rows: &[BinSummaryRow],
    exact_map: &BTreeMap<u64, f64>,
    top_fraction: f64,
    threshold_label: &str,
) -> RankingSummaryRow {
    // Metric convention: larger model scores mean higher predicted failure risk.
    // Keep all score/label vectors aligned by key id. A previous implementation
    // accidentally paired score-sorted scores with labels ordered by score rank,
    // which inverted or scrambled auc_top10_label for several M3 runs.
    let aligned = aligned_score_records(&artifacts.score_map, exact_map);
    let score_values = aligned.iter().map(|(_, _, score)| *score).collect::<Vec<_>>();
    let exact_values = aligned.iter().map(|(_, p_h, _)| *p_h).collect::<Vec<_>>();
    let bad_any_labels = aligned.iter().map(|(_, p_h, _)| *p_h > 0.0).collect::<Vec<_>>();
    let ranked = score_ranked_pairs(&artifacts.score_map, exact_map);
    let top_k_labels = top_k_labels(exact_map, top_fraction);
    let top10_labels = aligned
        .iter()
        .map(|(key_id, _, _)| top_k_labels.contains(key_id))
        .collect::<Vec<_>>();
    let direct_profile_bound = bin_rows.iter().map(|row| row.bin_mass * row.p_phi_max).sum::<f64>();
    RankingSummaryRow {
        run_id: run_id.to_string(),
        model: model.to_string(),
        key_count: exact_map.len(),
        spearman_with_p_h: spearman_rank_corr(&score_values, &exact_values),
        kendall_with_p_h: kendall_tau_b(&score_values, &exact_values),
        auc_bad_any: auc_binary(&score_values, &bad_any_labels),
        auc_top10_label: auc_binary(&score_values, &top10_labels),
        tail_l1_error: tail_l1_error(bin_rows, exact_map),
        bad_top10_recall: top_fraction_recall(&ranked, &top_k_labels, top_fraction),
        bad_top10_precision: top_fraction_precision(&ranked, &top_k_labels, top_fraction),
        direct_profile_bound,
        direct_profile_looseness: format_optional_f64(safe_ratio(
            direct_profile_bound,
            mean_f64(&exact_values),
        )),
        comments: format!(
            "threshold_schedule={threshold_label}; highest_bin={}; risk_direction=larger_score_higher_risk",
            artifacts.bin_map.values().copied().max().unwrap_or(0)
        ),
    }
}

fn build_bad_key_row(
    run_id: &str,
    model: &str,
    artifacts: &ModelArtifacts,
    exact_map: &BTreeMap<u64, f64>,
    top_fraction: f64,
) -> BadKeySummaryRow {
    let ranked = score_ranked_pairs(&artifacts.score_map, exact_map);
    let top_k = ((ranked.len() as f64 * top_fraction).ceil() as usize).max(1).min(ranked.len());
    let highest_bin = artifacts.bin_map.values().copied().max().unwrap_or(0);
    let highest_bin_keys = artifacts
        .bin_map
        .iter()
        .filter_map(|(key_id, bin_id)| (*bin_id == highest_bin).then_some(*key_id))
        .collect::<HashSet<_>>();
    let top_k_keys = ranked.iter().take(top_k).map(|(key_id, _, _)| *key_id).collect::<HashSet<_>>();
    let bad_any = exact_map.iter().filter_map(|(key_id, p_h)| (*p_h > 0.0).then_some(*key_id)).collect::<HashSet<_>>();
    let bad_high = exact_map.iter().filter_map(|(key_id, p_h)| (*p_h >= 0.5).then_some(*key_id)).collect::<HashSet<_>>();
    let exact_top = top_k_labels(exact_map, top_fraction);
    BadKeySummaryRow {
        run_id: run_id.to_string(),
        model: model.to_string(),
        top_k,
        highest_bin,
        bad_any_recall_at_highest_bin: set_recall(&highest_bin_keys, &bad_any),
        bad_high_recall_at_highest_bin: set_recall(&highest_bin_keys, &bad_high),
        bad_any_precision_at_highest_bin: set_precision(&highest_bin_keys, &bad_any),
        bad_high_precision_at_highest_bin: set_precision(&highest_bin_keys, &bad_high),
        bad_top10_recall: set_recall(&top_k_keys, &exact_top),
        bad_top10_precision: set_precision(&top_k_keys, &exact_top),
    }
}

fn build_static_key_rows(
    run_id: &str,
    model: &str,
    bin_rows: &[BinSummaryRow],
    exact_map: &BTreeMap<u64, f64>,
) -> Vec<StaticKeyBoundsRow> {
    let empirical_delta_avg = mean_f64(&exact_map.values().copied().collect::<Vec<_>>());
    FAIL_Q_GRID
        .into_iter()
        .map(|q| {
            let empirical_fail_q = exact_map
                .values()
                .map(|p_h| 1.0 - (1.0 - p_h).powf(q as f64))
                .sum::<f64>()
                / exact_map.len().max(1) as f64;
            let profile_bound_fail_q = bin_rows
                .iter()
                .map(|row| row.bin_mass * (1.0 - (1.0 - row.p_phi_max).powf(q as f64)))
                .sum::<f64>();
            let average_union_bound_min_1 = (q as f64 * empirical_delta_avg).min(1.0);
            StaticKeyBoundsRow {
                run_id: run_id.to_string(),
                model: model.to_string(),
                q,
                empirical_fail_q,
                profile_bound_fail_q,
                average_union_bound_min_1,
                profile_bound_looseness: format_optional_f64(safe_ratio(profile_bound_fail_q, empirical_fail_q)),
                improvement_vs_average_union_bound: average_union_bound_min_1 - profile_bound_fail_q,
            }
        })
        .collect()
}

fn write_per_key_profiles_csv(run_id: &str, rows: &[KeyAggregate], out_dir: &Path) -> Result<()> {
    let mut writer = csv_writer(out_dir.join("per_key_profiles.csv"))?;
    writeln!(
        writer,
        "run_id,key_id,h0_support,h1_support,p_h,failure_count,error_count,Dmax,C4_loc,Lambda_ge_2_count,Lambda_ge_3_count,Lambda_ge_4_count,Lambda_max_count,Omega_max,near_tail_u1,near_tail_u2,near_tail_u3,near_tail_u4,G_2_min_syndrome,G_2_low_count,G_3_min_syndrome,G_3_low_count,B_fixed_count,max_clean_counter_t1,fp0_proxy,fn0_proxy,reach_state_count,reach_successor_count,min_true_counter,mean_clean_counter,mean_true_counter,notes"
    )?;
    for row in rows {
        let lambda_map = parse_json_u64_map(&row.profile.lambda_json);
        let gather_map = parse_json_u64_map(&row.profile.gather_json);
        let near_map = parse_json_f64_map(&row.profile.an_u_json);
        writeln!(
            writer,
            "{},{},{},{},{:.17},{},{},{},{},{},{},{},{},{},{:.17},{:.17},{:.17},{:.17},{},{},{},{},{},{},{:.17},{:.17},{},{},{},{:.17},{:.17},{}",
            run_id,
            row.key.key_id,
            csv_escape(&join_support(&row.key.h0)),
            csv_escape(&join_support(&row.key.h1)),
            row.p_h,
            row.failure_count,
            row.error_count,
            row.profile.d_max,
            row.profile.c4_loc,
            lambda_ge_count_map(&lambda_map, 2),
            lambda_ge_count_map(&lambda_map, 3),
            lambda_ge_count_map(&lambda_map, 4),
            lambda_map.get(&row.profile.d_max.to_string()).copied().unwrap_or(0),
            row.profile.omega_pair_max,
            near_map.get("1").copied().unwrap_or(0.0),
            near_map.get("2").copied().unwrap_or(0.0),
            near_map.get("3").copied().unwrap_or(0.0),
            near_map.get("4").copied().unwrap_or(0.0),
            gather_min_syndrome(&gather_map, 2),
            gather_low_count(&gather_map, 2),
            gather_min_syndrome(&gather_map, 3),
            gather_low_count(&gather_map, 3),
            row.profile.b_fixed,
            row.max_clean_counter,
            row.fp0_exact_or_proxy,
            row.fn0_exact_or_proxy,
            format_optional_usize(row.reach_state_count),
            format_optional_usize(row.reach_successor_count),
            row.min_true_counter,
            row.mean_clean_counter,
            row.mean_true_counter,
            csv_escape("reach_state_count/reach_successor_count unavailable in this run and left NA.")
        )?;
    }
    Ok(())
}

fn write_per_key_m3_csv(
    run_id: &str,
    rows: &[KeyAggregate],
    models: &[(String, ModelArtifacts)],
    fp0_bins: &BTreeMap<u64, usize>,
    fn0_bins: &BTreeMap<u64, usize>,
    primary_tail_source: &str,
    out_dir: &Path,
) -> Result<()> {
    let model_map = models.iter().map(|(name, artifacts)| (name.clone(), artifacts)).collect::<BTreeMap<_, _>>();
    let mut writer = csv_writer(out_dir.join("per_key_m3.csv"))?;
    writeln!(
        writer,
        "run_id,key_id,M0_score,M0_bin,M1_score,M1_bin,M2_score,M2_bin,M3_old_score,M3_old_bin,M3_rank_score,M3_rank_bin,M3_rank_tuple,M3_cert_score,M3_cert_bin,M3_cert_tuple,M3_counter_score,M3_counter_bin,M3_counter_tuple,fp0_exact_or_proxy,fn0_exact_or_proxy,fp0_bin,fn0_bin,max_clean_counter,min_true_counter,mean_clean_counter,mean_true_counter,primary_tail_source,reach_state_count,reach_successor_count,notes"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{:.17},{},{:.17},{},{:.17},{},{:.17},{},{:.17},{},{},{:.17},{},{},{:.17},{},{},{:.17},{:.17},{},{},{},{},{:.17},{:.17},{},{},{},{}",
            run_id,
            row.key.key_id,
            model_map["M0"].score_map[&row.key.key_id],
            model_map["M0"].bin_map[&row.key.key_id],
            model_map["M1"].score_map[&row.key.key_id],
            model_map["M1"].bin_map[&row.key.key_id],
            model_map["M2"].score_map[&row.key.key_id],
            model_map["M2"].bin_map[&row.key.key_id],
            model_map["M3_old"].score_map[&row.key.key_id],
            model_map["M3_old"].bin_map[&row.key.key_id],
            model_map["M3_rank"].score_map[&row.key.key_id],
            model_map["M3_rank"].bin_map[&row.key.key_id],
            csv_escape(&model_map["M3_rank"].bin_label_map[&model_map["M3_rank"].bin_map[&row.key.key_id]]),
            model_map["M3_cert"].score_map[&row.key.key_id],
            model_map["M3_cert"].bin_map[&row.key.key_id],
            csv_escape(&model_map["M3_cert"].bin_label_map[&model_map["M3_cert"].bin_map[&row.key.key_id]]),
            model_map["M3_counter"].score_map[&row.key.key_id],
            model_map["M3_counter"].bin_map[&row.key.key_id],
            csv_escape(&model_map["M3_counter"].bin_label_map[&model_map["M3_counter"].bin_map[&row.key.key_id]]),
            row.fp0_exact_or_proxy,
            row.fn0_exact_or_proxy,
            fp0_bins.get(&row.key.key_id).copied().unwrap_or(0),
            fn0_bins.get(&row.key.key_id).copied().unwrap_or(0),
            row.max_clean_counter,
            row.min_true_counter,
            row.mean_clean_counter,
            row.mean_true_counter,
            csv_escape(primary_tail_source),
            format_optional_usize(row.reach_state_count),
            format_optional_usize(row.reach_successor_count),
            csv_escape("M3_counter uses counter-derived tuple bins; reach_* remains NA unless a reachable-state pass is added.")
        )?;
    }
    Ok(())
}

fn write_model_ranking_summary_csv(rows: &[RankingSummaryRow], out_dir: &Path) -> Result<()> {
    let mut writer = csv_writer(out_dir.join("model_ranking_summary.csv"))?;
    writeln!(
        writer,
        "run_id,model,key_count,spearman_with_p_h,kendall_with_p_h,auc_bad_any,auc_top10_label,tail_l1_error,bad_top10_recall,bad_top10_precision,direct_profile_bound,direct_profile_looseness,comments"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{},{}",
            row.run_id,
            row.model,
            row.key_count,
            row.spearman_with_p_h,
            row.kendall_with_p_h,
            row.auc_bad_any,
            row.auc_top10_label,
            row.tail_l1_error,
            row.bad_top10_recall,
            row.bad_top10_precision,
            row.direct_profile_bound,
            row.direct_profile_looseness,
            csv_escape(&row.comments)
        )?;
    }
    Ok(())
}

fn write_certificate_bound_summary_csv(rows: &[CertificateSummaryRow], out_dir: &Path) -> Result<()> {
    let mut writer = csv_writer(out_dir.join("certificate_bound_summary.csv"))?;
    writeln!(
        writer,
        "run_id,model,empirical_delta_avg,direct_profile_bound,direct_profile_looseness,number_of_bins,nonempty_bins,max_bin_mass,average_bin_mass,max_p_phi_max,mean_p_phi_max,mean_p_phi_mean"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{:.17},{:.17},{},{},{},{:.17},{:.17},{:.17},{:.17},{:.17}",
            row.run_id,
            row.model,
            row.empirical_delta_avg,
            row.direct_profile_bound,
            row.direct_profile_looseness,
            row.number_of_bins,
            row.nonempty_bins,
            row.max_bin_mass,
            row.average_bin_mass,
            row.max_p_phi_max,
            row.mean_p_phi_max,
            row.mean_p_phi_mean
        )?;
    }
    Ok(())
}

fn write_static_key_bounds_csv(rows: &[StaticKeyBoundsRow], out_dir: &Path) -> Result<()> {
    let mut writer = csv_writer(out_dir.join("static_key_bounds.csv"))?;
    writeln!(
        writer,
        "run_id,model,q,empirical_fail_q,profile_bound_fail_q,average_union_bound_min_1,profile_bound_looseness,improvement_vs_average_union_bound"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{:.17},{:.17},{:.17},{},{:.17}",
            row.run_id,
            row.model,
            row.q,
            row.empirical_fail_q,
            row.profile_bound_fail_q,
            row.average_union_bound_min_1,
            row.profile_bound_looseness,
            row.improvement_vs_average_union_bound
        )?;
    }
    Ok(())
}

fn write_bad_key_summary_csv(rows: &[BadKeySummaryRow], out_dir: &Path) -> Result<()> {
    let mut writer = csv_writer(out_dir.join("bad_key_recall_summary.csv"))?;
    writeln!(
        writer,
        "run_id,model,top_k,highest_bin,bad_any_recall_at_highest_bin,bad_high_recall_at_highest_bin,bad_any_precision_at_highest_bin,bad_high_precision_at_highest_bin,bad_top10_recall,bad_top10_precision"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17}",
            row.run_id,
            row.model,
            row.top_k,
            row.highest_bin,
            row.bad_any_recall_at_highest_bin,
            row.bad_high_recall_at_highest_bin,
            row.bad_any_precision_at_highest_bin,
            row.bad_high_precision_at_highest_bin,
            row.bad_top10_recall,
            row.bad_top10_precision
        )?;
    }
    Ok(())
}

fn write_bin_summary_csv(rows: &[BinSummaryRow], out_dir: &Path) -> Result<()> {
    let mut writer = csv_writer(out_dir.join("bin_summary.csv"))?;
    writeln!(writer, "run_id,model,bin_id,bin_label,key_count,bin_mass,p_phi_mean,p_phi_max")?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{},{},{:.17},{:.17},{:.17}",
            row.run_id,
            row.model,
            row.bin_id,
            csv_escape(&row.bin_label),
            row.key_count,
            row.bin_mass,
            row.p_phi_mean,
            row.p_phi_max
        )?;
    }
    Ok(())
}

fn write_worst_key_mechanisms_csv(
    run_id: &str,
    rows: &[KeyAggregate],
    m2: &ModelArtifacts,
    models: &[(String, ModelArtifacts)],
    t: usize,
    out_dir: &Path,
) -> Result<()> {
    let model_map = models.iter().map(|(name, artifacts)| (name.clone(), artifacts)).collect::<BTreeMap<_, _>>();
    let mut selected = rows.to_vec();
    selected.sort_by(|a, b| b.p_h.total_cmp(&a.p_h).then_with(|| a.key.key_id.cmp(&b.key.key_id)));
    if t == 1 {
        selected.retain(|row| row.p_h > 0.0);
    } else if selected.len() > 10 {
        selected.truncate(10);
    }
    let mut writer = csv_writer(out_dir.join("worst_key_mechanisms.csv"))?;
    writeln!(
        writer,
        "run_id,key_id,p_h,failure_count,error_count,M2_bin,M3_cert_tuple,Dmax,C4_loc,near_tail_u1,near_tail_u2,near_tail_u3,near_tail_u4,B_fixed_count,Omega_max,dominant_failure_type,timeout_count,wrong_zero_count,divergence_like_count,max_clean_counter,explanation_hint"
    )?;
    for row in selected {
        let near_map = parse_json_f64_map(&row.profile.an_u_json);
        writeln!(
            writer,
            "{},{},{:.17},{},{},{},{},{},{},{:.17},{:.17},{:.17},{:.17},{},{},{},{},{},{},{},{}",
            run_id,
            row.key.key_id,
            row.p_h,
            row.failure_count,
            row.error_count,
            m2.bin_map[&row.key.key_id],
            csv_escape(&model_map["M3_cert"].bin_label_map[&model_map["M3_cert"].bin_map[&row.key.key_id]]),
            row.profile.d_max,
            row.profile.c4_loc,
            near_map.get("1").copied().unwrap_or(0.0),
            near_map.get("2").copied().unwrap_or(0.0),
            near_map.get("3").copied().unwrap_or(0.0),
            near_map.get("4").copied().unwrap_or(0.0),
            row.profile.b_fixed,
            row.profile.omega_pair_max,
            row.dominant_failure_type,
            row.timeout_count,
            row.wrong_zero_count,
            row.divergence_like_count,
            row.max_clean_counter,
            csv_escape(&explanation_hint(&row))
        )?;
    }
    Ok(())
}

fn write_experiment_summary_md(
    run_id: &str,
    ranking_rows: &[RankingSummaryRow],
    certificate_rows: &[CertificateSummaryRow],
    bad_key_rows: &[BadKeySummaryRow],
    rows: &[KeyAggregate],
    primary_tail_source: &str,
    out_dir: &Path,
) -> Result<()> {
    let best_rank = ranking_rows
        .iter()
        .max_by(|a, b| a.spearman_with_p_h.total_cmp(&b.spearman_with_p_h))
        .map(|row| row.model.clone())
        .unwrap_or_else(|| "NA".to_string());
    let best_cert = certificate_rows
        .iter()
        .filter_map(|row| parse_optional_f64(&row.direct_profile_looseness).map(|value| (row.model.clone(), value)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(model, _)| model)
        .unwrap_or_else(|| "NA".to_string());
    let m1_rank = ranking_rows.iter().find(|row| row.model == "M1");
    let m2_rank = ranking_rows.iter().find(|row| row.model == "M2");
    let m3_rank = ranking_rows.iter().find(|row| row.model == "M3_rank");
    let m3_cert = certificate_rows.iter().find(|row| row.model == "M3_cert");
    let m2_cert = certificate_rows.iter().find(|row| row.model == "M2");
    let m3_static_q10 = find_static_looseness(out_dir, "M3_cert", 10)?;
    let m2_static_q10 = find_static_looseness(out_dir, "M2", 10)?;
    let worst = rows
        .iter()
        .filter(|row| row.p_h > 0.0)
        .take(3)
        .map(|row| format!("key {}: p_H={:.6}, Dmax={}, C4_loc={}, Omega_max={}, fp0={:.6}, fn0={:.6}", row.key.key_id, row.p_h, row.profile.d_max, row.profile.c4_loc, row.profile.omega_pair_max, row.fp0_exact_or_proxy, row.fn0_exact_or_proxy))
        .collect::<Vec<_>>();
    let mut lines = Vec::new();
    lines.push("# New M3 Experiment Summary".to_string());
    lines.push(String::new());
    lines.push(format!("- Run id: `{run_id}`"));
    lines.push(format!("- Best rank predictor by Spearman: `{}`", best_rank));
    lines.push(format!("- Best certificate refiner by looseness: `{}`", best_cert));
    lines.push(format!("- M3_cert primary tail source: `{}`", primary_tail_source));
    lines.push(String::new());
    lines.push("## Questions".to_string());
    lines.push(format!(
        "- Which feature/model was the best rank predictor in this regime? `{}`",
        best_rank
    ));
    if let (Some(m3_rank), Some(m1_rank), Some(m2_rank)) = (m3_rank, m1_rank, m2_rank) {
        lines.push(format!(
            "- Did M3_rank improve ranking over M1/M2? Spearman: M3_rank={:.6}, M1={:.6}, M2={:.6}.",
            m3_rank.spearman_with_p_h,
            m1_rank.spearman_with_p_h,
            m2_rank.spearman_with_p_h
        ));
    }
    if let (Some(m3_cert), Some(m2_cert)) = (m3_cert, m2_cert) {
        lines.push(format!(
            "- Did M3_cert improve direct profile-bound looseness? M3_cert={}, M2={}.",
            m3_cert.direct_profile_looseness,
            m2_cert.direct_profile_looseness
        ));
    }
    lines.push(format!(
        "- Did M3_cert improve static-key Fail_q bounds at q=10? M3_cert={}, M2={}.",
        m3_static_q10,
        m2_static_q10
    ));
    let structural_consistency = bad_key_rows
        .iter()
        .find(|row| row.model == "M3_cert")
        .map(|row| row.bad_any_recall_at_highest_bin > 0.0)
        .unwrap_or(false);
    lines.push(format!(
        "- Are worst keys structurally consistent across this regime? {}",
        if structural_consistency { "yes, the bad tail aligns with structural bins." } else { "partially; see worst_key_mechanisms.csv." }
    ));
    lines.push(format!(
        "- Does the evidence support the current paper claims? {}",
        if best_cert == "M3_cert" {
            "Yes for certificate refinement; ranking claims should remain qualified."
        } else {
            "Partially; M3 is more defensible as a certificate refiner than as a universal predictor."
        }
    ));
    lines.push(format!(
        "- Should the paper claim M3 as predictor, certificate refiner, or both? {}",
        if best_rank == "M3_rank" && best_cert == "M3_cert" {
            "Both, but with ranking phrased as competitive rather than universally best."
        } else {
            "Primarily a certificate refiner, with ranking utility reported separately."
        }
    ));
    lines.push(String::new());
    lines.push("## Worst Keys".to_string());
    if worst.is_empty() {
        lines.push("- No bad keys observed in this run.".to_string());
    } else {
        lines.extend(worst.into_iter().map(|line| format!("- {}", line)));
    }
    fs::write(out_dir.join("experiment_summary.md"), lines.join("\n") + "\n")
        .with_context(|| format!("failed to write {}", out_dir.join("experiment_summary.md").display()))
}

fn write_run_metadata_json(metadata: &RunMetadata, out_dir: &Path) -> Result<()> {
    let payload = serde_json::to_vec_pretty(metadata)?;
    fs::write(out_dir.join("run_metadata.json"), payload)
        .with_context(|| format!("failed to write {}", out_dir.join("run_metadata.json").display()))
}

fn compute_schedule_stats(
    keys: &[Key],
    decoder: &DecoderSection,
    errors: &[Mask256],
    t: usize,
    divergence_flip_threshold: usize,
    divergence_residual_threshold: usize,
) -> Result<ScheduleStats> {
    let mut key_failure_rates = Vec::with_capacity(keys.len());
    let mut success = 0usize;
    let mut timeout = 0usize;
    let mut wrong_zero = 0usize;
    let mut nonzero_halt = 0usize;
    for key in keys {
        let runtime = build_key_runtime(key.clone())?;
        let mut key_failures = 0usize;
        for error in errors {
            let result = decode_with_logs(&runtime, *error, decoder)?;
            let _ = classify_outcome(&result, divergence_flip_threshold, divergence_residual_threshold, t);
            match result.outcome {
                Outcome::Success => success += 1,
                Outcome::Timeout => {
                    timeout += 1;
                    key_failures += 1;
                }
                Outcome::WrongZero => {
                    wrong_zero += 1;
                    key_failures += 1;
                }
                Outcome::NonzeroHalt => {
                    nonzero_halt += 1;
                    key_failures += 1;
                }
            }
        }
        key_failure_rates.push(key_failures as f64 / errors.len().max(1) as f64);
    }
    key_failure_rates.sort_by(|a, b| a.total_cmp(b));
    let total = (keys.len() * errors.len()).max(1) as f64;
    Ok(ScheduleStats {
        p_h_mean: mean_f64(&key_failure_rates),
        p_h_min: key_failure_rates.first().copied().unwrap_or(0.0),
        p_h_median: median_sorted(&key_failure_rates),
        p_h_max: key_failure_rates.last().copied().unwrap_or(0.0),
        success_rate: success as f64 / total,
        timeout_rate: timeout as f64 / total,
        wrong_zero_rate: wrong_zero as f64 / total,
        nonzero_halt_rate: nonzero_halt as f64 / total,
    })
}

fn zero_error_pass_rate(keys: &[Key], params: &M3ThresholdSweepParameters) -> Result<f64> {
    let decoder = DecoderSection {
        iterations: params.schedules.first().map(|s| s.len()).unwrap_or(1),
        thresholds: params.schedules.first().cloned().unwrap_or_else(|| vec![1]),
        wrong_zero_is_failure: params.wrong_zero_is_failure,
        flip_policy: "all_ge_threshold".to_string(),
    };
    let mut passes = 0usize;
    for key in keys {
        let runtime = build_key_runtime(key.clone())?;
        let result = decode_with_logs(&runtime, Mask256::empty(), &decoder)?;
        if result.outcome == Outcome::Success && result.final_syndrome_weight == 0 {
            passes += 1;
        }
    }
    Ok(passes as f64 / keys.len().max(1) as f64)
}

fn single_bit_geometry_pass_rate(keys: &[Key], params: &M3ThresholdSweepParameters) -> Result<f64> {
    let mut total = 0usize;
    let mut passes = 0usize;
    for key in keys.iter().take(4.min(keys.len())) {
        let runtime = build_key_runtime(key.clone())?;
        for bit_index in 0..(2 * params.r) {
            let mut error = Mask256::empty();
            error.set(bit_index)?;
            let syndrome = syndrome_for_error(&runtime, error)?;
            let observed = runtime.column_masks[bit_index].and(syndrome).popcount() as usize;
            total += 1;
            if observed == params.d {
                passes += 1;
            }
        }
    }
    Ok(passes as f64 / total.max(1) as f64)
}

fn syndrome_update_pass_rate(keys: &[Key], params: &M3ThresholdSweepParameters) -> Result<f64> {
    let decoder = DecoderSection {
        iterations: params.schedules.first().map(|s| s.len()).unwrap_or(1),
        thresholds: params.schedules.first().cloned().unwrap_or_else(|| vec![1]),
        wrong_zero_is_failure: params.wrong_zero_is_failure,
        flip_policy: "all_ge_threshold".to_string(),
    };
    let errors = generate_error_masks(params.r, params.t.max(1), params.exact_errors, params.error_sample_count.unwrap_or(32), 17)?;
    let mut total = 0usize;
    let mut passes = 0usize;
    for key in keys.iter().take(4.min(keys.len())) {
        let runtime = build_key_runtime(key.clone())?;
        for error in &errors {
            let result = decode_with_logs(&runtime, *error, &decoder)?;
            for log in &result.logs {
                let recomputed = syndrome_for_error(&runtime, log.e_mask)?;
                total += 1;
                if recomputed == log.syndrome {
                    passes += 1;
                }
            }
        }
    }
    Ok(passes as f64 / total.max(1) as f64)
}

fn write_threshold_sweep_csv(rows: &[ThresholdSweepRow], out_dir: &Path) -> Result<()> {
    let mut writer = csv_writer(out_dir.join("threshold_sweep.csv"))?;
    writeln!(
        writer,
        "run_id,r,d,t,key_count,error_count_per_key,threshold_schedule,zero_error_pass_rate,single_bit_geometry_pass_rate,syndrome_update_pass_rate,p_h_mean,p_h_min,p_h_median,p_h_max,success_rate,timeout_rate,wrong_zero_rate,nonzero_halt_rate,selected"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{}",
            row.run_id,
            row.r,
            row.d,
            row.t,
            row.key_count,
            row.error_count_per_key,
            row.threshold_schedule,
            row.zero_error_pass_rate,
            row.single_bit_geometry_pass_rate,
            row.syndrome_update_pass_rate,
            row.p_h_mean,
            row.p_h_min,
            row.p_h_median,
            row.p_h_max,
            row.success_rate,
            row.timeout_rate,
            row.wrong_zero_rate,
            row.nonzero_halt_rate,
            row.selected
        )?;
    }
    Ok(())
}

fn write_threshold_sweep_summary(rows: &[ThresholdSweepRow], out_dir: &Path) -> Result<()> {
    let selected = rows.iter().find(|row| row.selected);
    let mut lines = Vec::new();
    lines.push("# Threshold Sweep Summary".to_string());
    lines.push(String::new());
    if let Some(row) = selected {
        lines.push(format!("- Selected schedule: `{}`", row.threshold_schedule));
        lines.push(format!("- mean p_H: {:.6}", row.p_h_mean));
        lines.push(format!("- success rate: {:.6}", row.success_rate));
    } else {
        lines.push("- No acceptable non-saturated schedule found.".to_string());
    }
    fs::write(out_dir.join("threshold_sweep_summary.md"), lines.join("\n") + "\n")
        .with_context(|| format!("failed to write {}", out_dir.join("threshold_sweep_summary.md").display()))
}

fn select_sweep_row(
    rows: &[ThresholdSweepRow],
    preferred_min: f64,
    preferred_max: f64,
    acceptable_min: f64,
    acceptable_max: f64,
) -> Option<usize> {
    let mut candidates = rows
        .iter()
        .enumerate()
        .filter(|(_, row)| row.zero_error_pass_rate == 1.0 && row.single_bit_geometry_pass_rate == 1.0 && row.syndrome_update_pass_rate == 1.0)
        .filter(|(_, row)| (acceptable_min..=acceptable_max).contains(&row.p_h_mean))
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| {
        let a_preferred = (preferred_min..=preferred_max).contains(&a.1.p_h_mean);
        let b_preferred = (preferred_min..=preferred_max).contains(&b.1.p_h_mean);
        b_preferred
            .cmp(&a_preferred)
            .then_with(|| (a.1.p_h_mean - 0.15).abs().total_cmp(&(b.1.p_h_mean - 0.15).abs()))
            .then_with(|| a.1.threshold_schedule.cmp(&b.1.threshold_schedule))
    });
    candidates.into_iter().next().map(|(index, _)| index)
}

fn aligned_scores(score_map: &BTreeMap<u64, f64>, exact_map: &BTreeMap<u64, f64>) -> (Vec<f64>, Vec<f64>) {
    let mut aligned = exact_map
        .iter()
        .map(|(key_id, p_h)| (*score_map.get(key_id).unwrap_or(&0.0), *p_h))
        .collect::<Vec<_>>();
    aligned.sort_by(|a, b| a.0.total_cmp(&b.0));
    (
        aligned.iter().map(|(score, _)| *score).collect(),
        aligned.iter().map(|(_, p_h)| *p_h).collect(),
    )
}

fn aligned_score_records(score_map: &BTreeMap<u64, f64>, exact_map: &BTreeMap<u64, f64>) -> Vec<(u64, f64, f64)> {
    exact_map
        .iter()
        .map(|(key_id, p_h)| (*key_id, *p_h, *score_map.get(key_id).unwrap_or(&0.0)))
        .collect::<Vec<_>>()
}


fn score_ranked_pairs(score_map: &BTreeMap<u64, f64>, exact_map: &BTreeMap<u64, f64>) -> Vec<(u64, f64, f64)> {
    let mut ranked = exact_map
        .iter()
        .map(|(key_id, p_h)| (*key_id, *p_h, *score_map.get(key_id).unwrap_or(&0.0)))
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
    ranked
}

fn top_k_labels(exact_map: &BTreeMap<u64, f64>, top_fraction: f64) -> HashSet<u64> {
    let mut exact_ranked = exact_map.iter().map(|(key_id, p_h)| (*key_id, *p_h)).collect::<Vec<_>>();
    exact_ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let k = ((exact_ranked.len() as f64 * top_fraction).ceil() as usize).max(1).min(exact_ranked.len());
    exact_ranked.into_iter().take(k).map(|(key_id, _)| key_id).collect()
}

fn top_fraction_recall(score_ranked: &[(u64, f64, f64)], truth: &HashSet<u64>, fraction: f64) -> f64 {
    let k = ((score_ranked.len() as f64 * fraction).ceil() as usize).max(1).min(score_ranked.len());
    let predicted = score_ranked.iter().take(k).map(|(key_id, _, _)| *key_id).collect::<HashSet<_>>();
    set_recall(&predicted, truth)
}

fn top_fraction_precision(score_ranked: &[(u64, f64, f64)], truth: &HashSet<u64>, fraction: f64) -> f64 {
    let k = ((score_ranked.len() as f64 * fraction).ceil() as usize).max(1).min(score_ranked.len());
    let predicted = score_ranked.iter().take(k).map(|(key_id, _, _)| *key_id).collect::<HashSet<_>>();
    set_precision(&predicted, truth)
}

fn set_recall(predicted: &HashSet<u64>, truth: &HashSet<u64>) -> f64 {
    if truth.is_empty() {
        0.0
    } else {
        predicted.intersection(truth).count() as f64 / truth.len() as f64
    }
}

fn set_precision(predicted: &HashSet<u64>, truth: &HashSet<u64>) -> f64 {
    if predicted.is_empty() {
        0.0
    } else {
        predicted.intersection(truth).count() as f64 / predicted.len() as f64
    }
}

fn tail_l1_error(bin_rows: &[BinSummaryRow], exact_map: &BTreeMap<u64, f64>) -> f64 {
    TAU_GRID
        .iter()
        .map(|tau| {
            let empirical = exact_map.values().filter(|value| **value > *tau).count() as f64 / exact_map.len().max(1) as f64;
            let model_tail = bin_rows.iter().filter(|row| row.p_phi_max > *tau).map(|row| row.bin_mass).sum::<f64>();
            (empirical - model_tail).abs()
        })
        .sum::<f64>()
        / TAU_GRID.len() as f64
}

fn mean_f64(values: &[f64]) -> f64 {
    if values.is_empty() { 0.0 } else { values.iter().sum::<f64>() / values.len() as f64 }
}

fn median_sorted(values: &[f64]) -> f64 {
    if values.is_empty() { return 0.0; }
    let mid = values.len() / 2;
    if values.len() % 2 == 1 { values[mid] } else { (values[mid - 1] + values[mid]) / 2.0 }
}

fn spearman_rank_corr(xs: &[f64], ys: &[f64]) -> f64 {
    if xs.len() != ys.len() || xs.is_empty() || all_equal(xs) || all_equal(ys) { return 0.0; }
    let rx = rank_values(xs);
    let ry = rank_values(ys);
    pearson(&rx, &ry)
}

fn rank_values(values: &[f64]) -> Vec<f64> {
    let mut indexed = values.iter().copied().enumerate().collect::<Vec<_>>();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    let mut ranks = vec![0.0; values.len()];
    for (rank, (index, _)) in indexed.into_iter().enumerate() {
        ranks[index] = rank as f64 + 1.0;
    }
    ranks
}

fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    if xs.len() != ys.len() || xs.is_empty() { return 0.0; }
    let mean_x = mean_f64(xs);
    let mean_y = mean_f64(ys);
    let mut num = 0.0_f64;
    let mut den_x = 0.0_f64;
    let mut den_y = 0.0_f64;
    for (x, y) in xs.iter().zip(ys) {
        let dx = *x - mean_x;
        let dy = *y - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }
    if den_x <= 1e-15 || den_y <= 1e-15 { 0.0 } else { num / (den_x.sqrt() * den_y.sqrt()) }
}

fn kendall_tau_b(xs: &[f64], ys: &[f64]) -> f64 {
    if xs.len() != ys.len() || xs.len() < 2 { return 0.0; }
    let mut concordant = 0.0_f64;
    let mut discordant = 0.0_f64;
    let mut ties_x = 0.0_f64;
    let mut ties_y = 0.0_f64;
    for i in 0..xs.len() {
        for j in (i + 1)..xs.len() {
            let dx = xs[i].total_cmp(&xs[j]);
            let dy = ys[i].total_cmp(&ys[j]);
            match (dx, dy) {
                (Ordering::Equal, Ordering::Equal) => {}
                (Ordering::Equal, _) => ties_x += 1.0,
                (_, Ordering::Equal) => ties_y += 1.0,
                (Ordering::Less, Ordering::Less) | (Ordering::Greater, Ordering::Greater) => concordant += 1.0,
                _ => discordant += 1.0,
            }
        }
    }
    let denom = ((concordant + discordant + ties_x) * (concordant + discordant + ties_y)).sqrt();
    if denom <= 1e-15 { 0.0 } else { (concordant - discordant) / denom }
}

fn auc_binary(scores: &[f64], labels: &[bool]) -> f64 {
    if scores.len() != labels.len() || scores.is_empty() {
        return 0.0;
    }
    let positives = scores.iter().zip(labels).filter_map(|(score, label)| (*label).then_some(*score)).collect::<Vec<_>>();
    let negatives = scores.iter().zip(labels).filter_map(|(score, label)| (!*label).then_some(*score)).collect::<Vec<_>>();
    if positives.is_empty() || negatives.is_empty() {
        return 0.5;
    }
    let mut wins = 0.0_f64;
    for pos in &positives {
        for neg in &negatives {
            wins += match pos.total_cmp(neg) {
                Ordering::Greater => 1.0,
                Ordering::Equal => 0.5,
                Ordering::Less => 0.0,
            };
        }
    }
    wins / (positives.len() * negatives.len()) as f64
}

fn all_equal(values: &[f64]) -> bool {
    values.first().map(|first| values.iter().all(|value| (*value - *first).abs() <= 1e-15)).unwrap_or(true)
}

fn safe_ratio(numer: f64, denom: f64) -> Option<f64> {
    (denom.abs() > 1e-15).then_some(numer / denom)
}

fn format_optional_f64(value: Option<f64>) -> String {
    value.map(|inner| format!("{inner:.17}")).unwrap_or_else(|| "NA".to_string())
}

fn format_optional_usize(value: Option<usize>) -> String {
    value.map(|inner| inner.to_string()).unwrap_or_else(|| "NA".to_string())
}

fn parse_optional_f64(value: &str) -> Option<f64> {
    value.parse::<f64>().ok()
}

fn join_support(values: &[u16]) -> String {
    values.iter().map(u16::to_string).collect::<Vec<_>>().join(";")
}

fn parse_json_u64_map(payload: &str) -> BTreeMap<String, u64> {
    serde_json::from_str(payload).unwrap_or_default()
}

fn parse_json_f64_map(payload: &str) -> BTreeMap<String, f64> {
    serde_json::from_str(payload).unwrap_or_default()
}

fn lambda_ge_count(payload: &str, min_k: u32) -> u64 {
    let map = parse_json_u64_map(payload);
    lambda_ge_count_map(&map, min_k)
}

fn lambda_ge_count_map(map: &BTreeMap<String, u64>, min_k: u32) -> u64 {
    map.iter()
        .filter_map(|(key, value)| key.parse::<u32>().ok().map(|parsed| (parsed, *value)))
        .filter(|(parsed, _)| *parsed >= min_k)
        .map(|(_, value)| value)
        .sum()
}

fn near_tail_high(payload: &str) -> f64 {
    parse_json_f64_map(payload).values().copied().fold(0.0_f64, f64::max)
}

fn gather_min_syndrome(map: &BTreeMap<String, u64>, m: usize) -> String {
    let mut values = map
        .keys()
        .filter_map(|key| {
            let mut parts = key.split(':');
            let left = parts.next()?.parse::<usize>().ok()?;
            let right = parts.next()?.parse::<usize>().ok()?;
            (left == m).then_some(right)
        })
        .collect::<Vec<_>>();
    values.sort_unstable();
    values.first().map(|value| value.to_string()).unwrap_or_else(|| "NA".to_string())
}

fn gather_low_count(map: &BTreeMap<String, u64>, m: usize) -> u64 {
    map.iter()
        .filter_map(|(key, value)| {
            let mut parts = key.split(':');
            let left = parts.next()?.parse::<usize>().ok()?;
            let right = parts.next()?.parse::<usize>().ok()?;
            (left == m && right <= 3).then_some(*value)
        })
        .sum()
}

fn explanation_hint(row: &KeyAggregate) -> String {
    let mut signals = 0usize;
    if row.profile.d_max >= 3 { signals += 1; }
    if row.profile.omega_pair_max >= 3 { signals += 1; }
    if row.fp0_exact_or_proxy > 0.0 { signals += 1; }
    if row.profile.b_fixed > 0 { signals += 1; }
    if signals >= 2 {
        "mixed".to_string()
    } else if row.fp0_exact_or_proxy > 0.0 {
        "counter-fp dominant".to_string()
    } else if row.profile.d_max >= 3 {
        "high Dmax".to_string()
    } else if row.profile.b_fixed > 0 {
        "fixed-point indicator".to_string()
    } else {
        "unknown".to_string()
    }
}

fn classify_outcome(result: &DecodeResult, divergence_flip_threshold: usize, divergence_residual_threshold: usize, t: usize) -> String {
    let divergence_like = result.logs.iter().any(|log| {
        usize::from(log.flip_count) >= divergence_flip_threshold || usize::from(log.residual_weight) >= divergence_residual_threshold
    }) || usize::from(result.final_residual_weight) >= divergence_residual_threshold.max(t);
    if divergence_like && result.outcome != Outcome::Success {
        "divergence_like".to_string()
    } else {
        match result.outcome {
            Outcome::Success => "success".to_string(),
            Outcome::Timeout => "timeout".to_string(),
            Outcome::WrongZero => "wrong_zero".to_string(),
            Outcome::NonzeroHalt => "nonzero_halt".to_string(),
        }
    }
}

fn schedule_to_string(thresholds: &[usize]) -> String {
    thresholds.iter().map(usize::to_string).collect::<Vec<_>>().join(";")
}

fn find_static_looseness(out_dir: &Path, model: &str, q: usize) -> Result<String> {
    let text = fs::read_to_string(out_dir.join("static_key_bounds.csv"))
        .with_context(|| format!("failed to read {}", out_dir.join("static_key_bounds.csv").display()))?;
    for line in text.lines().skip(1) {
        let parts = line.split(',').collect::<Vec<_>>();
        if parts.len() >= 7 && parts[1] == model && parts[2] == q.to_string() {
            return Ok(parts[6].to_string());
        }
    }
    Ok("NA".to_string())
}

fn csv_escape(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

fn csv_writer(path: PathBuf) -> Result<BufWriter<File>> {
    let file = File::create(&path).with_context(|| format!("failed to create {}", path.display()))?;
    Ok(BufWriter::new(file))
}

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn gen_range(&mut self, upper: usize) -> usize {
        if upper <= 1 { 0 } else { (self.next_u64() % upper as u64) as usize }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{build_ranking_row, format_tuple_label, schedule_to_string, ModelArtifacts};

    #[test]
    fn tuple_labels_are_stable() {
        assert_eq!(format_tuple_label("M3_rank", &[1, 2, 3]), "a=1|b=2|c=3");
    }

    #[test]
    fn schedules_are_stable() {
        assert_eq!(schedule_to_string(&[4, 4, 3, 3]), "4;4;3;3");
    }

    #[test]
    fn top10_auc_uses_key_aligned_labels_and_high_score_as_high_risk() {
        let exact_map = BTreeMap::from([
            (1_u64, 0.0_f64),
            (2_u64, 0.1_f64),
            (3_u64, 0.5_f64),
            (4_u64, 1.0_f64),
        ]);
        let artifacts = ModelArtifacts {
            score_map: BTreeMap::from([
                (1_u64, 0.0_f64),
                (2_u64, 1.0_f64),
                (3_u64, 2.0_f64),
                (4_u64, 3.0_f64),
            ]),
            bin_map: BTreeMap::from([
                (1_u64, 0_usize),
                (2_u64, 1_usize),
                (3_u64, 2_usize),
                (4_u64, 3_usize),
            ]),
            bin_label_map: BTreeMap::new(),
        };
        let row = build_ranking_row(
            "test",
            "M_test",
            &artifacts,
            &[],
            &exact_map,
            0.5,
            "4",
        );
        assert!((row.auc_top10_label - 1.0).abs() <= 1e-12);
        assert!((row.auc_bad_any - 1.0).abs() <= 1e-12);
    }
}
