use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::bitset::Mask256;
use crate::binning::{build_bins_for_model, model_score};
use crate::combinatorics::{binom_u128, iter_combinations};
use crate::decoder::{decode_with_logs, syndrome_for_error, DecodeResult, Outcome};
use crate::io::{default_image_tag, detect_git_commit, file_sha256, now_utc, write_manifest, DecoderSection, Manifest};
use crate::key::{build_key_runtime, enumerate_all_keys, sample_keys_uniform, Key, KeyRuntime};
use crate::profile::{compute_profiles, ProfileRecord};
use crate::progress::ProgressReporter;

#[derive(Clone, Debug, Deserialize)]
struct DiagnosticsConfig {
    run: DiagnosticsRunSection,
    cases: Vec<DiagnosticsCase>,
}

#[derive(Clone, Debug, Deserialize)]
struct DiagnosticsRunSection {
    run_id: String,
    seed: u64,
}

#[derive(Clone, Debug, Deserialize)]
struct DiagnosticsCase {
    name: String,
    r: usize,
    d: usize,
    key_mode: String,
    key_count: usize,
    geometry_key_count: usize,
    update_key_count: usize,
    distribution_key_count: usize,
    baseline_thresholds: Vec<usize>,
    wrong_zero_is_failure: bool,
    t_values: Vec<usize>,
    distribution_t_values: Vec<usize>,
    sweep_max_iterations: Vec<usize>,
    sweep_constant_thresholds: Vec<usize>,
    sweep_schedules: Vec<Vec<usize>>,
    error_sample_count: usize,
    exact_error_weight_max: usize,
    divergence_flip_threshold: usize,
    divergence_residual_threshold: usize,
}

#[derive(Clone, Debug, Serialize)]
struct ZeroErrorSanityRow {
    r: usize,
    d: usize,
    key_id: u64,
    success: bool,
    final_syndrome_weight: u16,
    failure_type: String,
}

#[derive(Clone, Debug, Serialize)]
struct SingleBitGeometryRow {
    r: usize,
    d: usize,
    key_id: u64,
    bit_index: usize,
    expected_counter: usize,
    observed_true_counter: usize,
    syndrome_weight: u32,
    geometry_ok: bool,
    block: usize,
    position: usize,
}

#[derive(Clone, Debug, Serialize)]
struct SyndromeUpdateRow {
    r: usize,
    d: usize,
    t: usize,
    key_id: u64,
    error_id: usize,
    iteration: usize,
    incremental_syndrome_weight: u32,
    recomputed_syndrome_weight: u32,
    syndromes_equal: bool,
    flipped_count: u16,
}

#[derive(Clone, Debug, Serialize)]
struct CounterDistributionRow {
    r: usize,
    d: usize,
    t: usize,
    key_id: u64,
    error_id: usize,
    iteration: usize,
    bit_type: String,
    counter: usize,
}

#[derive(Clone, Debug, Serialize)]
struct ThresholdSweepRow {
    r: usize,
    d: usize,
    t: usize,
    key_count: usize,
    error_count_per_key: usize,
    threshold_schedule: String,
    max_iterations: usize,
    p_h_mean: f64,
    p_h_min: f64,
    p_h_median: f64,
    p_h_max: f64,
    success_rate: f64,
    timeout_rate: f64,
    wrong_zero_rate: f64,
    nonzero_halt_rate: f64,
    mean_final_residual_weight: f64,
    mean_final_syndrome_weight: f64,
}

#[derive(Clone, Debug, Serialize)]
struct FailureTaxonomyRow {
    r: usize,
    d: usize,
    t: usize,
    key_id: u64,
    error_id: usize,
    threshold_schedule: String,
    outcome: String,
    iterations_used: u16,
    initial_syndrome_weight: u32,
    final_syndrome_weight: u16,
    final_residual_weight: u16,
    flipped_total: u32,
    wrong_zero: bool,
}

#[derive(Clone, Debug, Serialize)]
struct UnifiedModelComparisonRow {
    run_id: String,
    r: usize,
    d: usize,
    t: usize,
    key_id: u64,
    p_h: f64,
    M0_score: f64,
    M1_score: f64,
    M2_score: f64,
    M3_old_score: f64,
    M3_hier_rank_score: f64,
    profile_bin_M0: usize,
    profile_bin_M1: usize,
    profile_bin_M2: usize,
    profile_bin_M3_old: usize,
    profile_bin_M3_hier: usize,
    M3_hier_tuple: String,
}

#[derive(Clone, Debug, Serialize)]
struct ModelRankingSummaryRow {
    run_id: String,
    r: usize,
    d: usize,
    t: usize,
    model: String,
    key_count: usize,
    p_h_mean: f64,
    p_h_min: f64,
    p_h_median: f64,
    p_h_max: f64,
    spearman_with_p_h: f64,
    pearson_with_p_h: f64,
    kendall_with_p_h: f64,
    auc_bad_any: f64,
    tail_l1_error: f64,
    top_10_percent_recall: f64,
    direct_average_bound: f64,
    empirical_average: f64,
    direct_bound_looseness: Option<f64>,
    comments: String,
}

#[derive(Clone, Debug, Serialize)]
struct PerKeyProfileRow {
    run_id: String,
    key_id: u64,
    h0_support: String,
    h1_support: String,
    p_h: f64,
    failure_count: usize,
    error_count: usize,
    M0_score: f64,
    M0_bin: usize,
    M1_score: f64,
    M1_bin: usize,
    M2_score: f64,
    M2_bin: usize,
    M3_old_score: f64,
    M3_old_bin: usize,
    Dmax: u32,
    C4_loc: u64,
    Lambda_ge_2_count: u64,
    Lambda_ge_3_count: u64,
    Lambda_max_count: u64,
    near_tail_u1: f64,
    near_tail_u2: f64,
    near_tail_u3: f64,
    near_tail_u4: f64,
    G_2_ell_min_or_count: String,
    G_3_ell_min_or_count: String,
    B_fixed_count: u64,
    Omega_max: u32,
    max_clean_counter_t1: String,
    notes: String,
}

#[derive(Clone, Debug, Serialize)]
struct PerKeyProfileWithHierRow {
    #[serde(flatten)]
    base: PerKeyProfileRow,
    M3_hier_tuple: String,
    M3_hier_bin: usize,
    M3_hier_rank_score: f64,
    M3_hier_primary_bin: usize,
    M3_hier_refinement_bin: usize,
}

#[derive(Clone, Debug, Serialize)]
struct BinSummaryRow {
    run_id: String,
    r: usize,
    d: usize,
    t: usize,
    model: String,
    bin_id: usize,
    bin_label: String,
    key_count: usize,
    bin_mass: f64,
    p_bin_mean: f64,
    p_bin_max: f64,
}

#[derive(Clone, Debug, Serialize)]
struct TailL1SummaryRow {
    run_id: String,
    r: usize,
    d: usize,
    t: usize,
    model: String,
    key_count: usize,
    empirical_average: f64,
    direct_average_bound: f64,
    direct_bound_looseness: String,
    tail_l1_error: f64,
    spearman_with_p_h: f64,
    pearson_with_p_h: f64,
    kendall_with_p_h: f64,
    auc_bad_any: f64,
    empirical_tail_tau_0: f64,
    empirical_tail_tau_0_01: f64,
    empirical_tail_tau_0_05: f64,
    empirical_tail_tau_0_1: f64,
    empirical_tail_tau_0_25: f64,
    empirical_tail_tau_0_5: f64,
    empirical_tail_tau_0_75: f64,
    empirical_tail_tau_0_99: f64,
    model_tail_tau_0: f64,
    model_tail_tau_0_01: f64,
    model_tail_tau_0_05: f64,
    model_tail_tau_0_1: f64,
    model_tail_tau_0_25: f64,
    model_tail_tau_0_5: f64,
    model_tail_tau_0_75: f64,
    model_tail_tau_0_99: f64,
}

#[derive(Clone, Debug, Serialize)]
struct BadKeyRecallSummaryRow {
    run_id: String,
    r: usize,
    d: usize,
    t: usize,
    model: String,
    top_k: usize,
    highest_bin: usize,
    bad_any_recall_at_highest_bin: f64,
    bad_high_recall_at_highest_bin: f64,
    bad_catastrophic_recall_at_highest_bin: f64,
    bad_any_precision_at_highest_bin: f64,
    bad_high_precision_at_highest_bin: f64,
    bad_catastrophic_precision_at_highest_bin: f64,
    bad_any_recall_at_top_k: f64,
    bad_high_recall_at_top_k: f64,
    bad_catastrophic_recall_at_top_k: f64,
}

#[derive(Clone, Debug, Serialize)]
struct StaticKeyBoundsRow {
    run_id: String,
    model: String,
    q: usize,
    empirical_fail_q: f64,
    profile_bound_fail_q: f64,
    average_union_bound: f64,
    profile_bound_looseness: String,
}

#[derive(Clone, Debug, Serialize)]
struct WorstKeyMechanismRow {
    run_id: String,
    key_id: u64,
    p_h: f64,
    failure_count: usize,
    error_count: usize,
    M2_bin: usize,
    M3_hier_tuple: String,
    Dmax: u32,
    C4_loc: u64,
    near_tail_u1: f64,
    near_tail_u2: f64,
    near_tail_u3: f64,
    near_tail_u4: f64,
    B_fixed_count: u64,
    Omega_max: u32,
    dominant_failure_type: String,
    timeout_count: usize,
    wrong_zero_count: usize,
    divergence_like_count: usize,
    max_clean_counter: String,
    explanation_hint: String,
}

#[derive(Clone, Debug, Serialize)]
struct RunMetadata {
    run_id: String,
    case_name: String,
    r: usize,
    d: usize,
    t: usize,
    threshold_schedule: String,
    key_count: usize,
    error_count_per_key: usize,
    models: Vec<String>,
    selected_from_sweep: bool,
}

#[derive(Clone, Debug)]
struct HierKeyAssignment {
    tuple_text: String,
    global_bin_id: usize,
    primary_bin: usize,
    refinement_bin: usize,
    rank_score: f64,
}

#[derive(Clone, Debug)]
struct ModelArtifacts {
    score_map: BTreeMap<u64, f64>,
    bin_map: BTreeMap<u64, usize>,
    bin_label_map: BTreeMap<usize, String>,
}

#[derive(Clone, Debug)]
struct ComparisonArtifacts {
    unified_rows: Vec<UnifiedModelComparisonRow>,
    ranking_rows: Vec<ModelRankingSummaryRow>,
    per_key_rows: Vec<PerKeyProfileRow>,
    per_key_hier_rows: Vec<PerKeyProfileWithHierRow>,
    bin_summary_rows: Vec<BinSummaryRow>,
    tail_summary_rows: Vec<TailL1SummaryRow>,
    bad_key_rows: Vec<BadKeyRecallSummaryRow>,
    static_key_rows: Vec<StaticKeyBoundsRow>,
    worst_key_rows: Vec<WorstKeyMechanismRow>,
    run_metadata: Option<RunMetadata>,
    shared_keys_payloads: Vec<String>,
}

#[derive(Clone, Debug)]
struct KeyExactStats {
    key_id: u64,
    p_h: f64,
    failure_count: usize,
    error_count: usize,
}

#[derive(Clone, Debug, Serialize)]
struct DiagnosticsSummaryJson {
    verdict: String,
    zero_error_pass_rate: f64,
    single_bit_geometry_pass_rate: f64,
    syndrome_update_pass_rate: f64,
    non_saturated_setting_count: usize,
    failure_counts: BTreeMap<String, usize>,
    recommendation: String,
}

#[derive(Clone, Debug)]
struct RunCaseDiagnostics {
    zero_error_rows: Vec<ZeroErrorSanityRow>,
    geometry_rows: Vec<SingleBitGeometryRow>,
    update_rows: Vec<SyndromeUpdateRow>,
    counter_rows: Vec<CounterDistributionRow>,
    sweep_rows: Vec<ThresholdSweepRow>,
    failure_rows: Vec<FailureTaxonomyRow>,
    unified_rows: Vec<UnifiedModelComparisonRow>,
    ranking_rows: Vec<ModelRankingSummaryRow>,
    per_key_rows: Vec<PerKeyProfileRow>,
    per_key_hier_rows: Vec<PerKeyProfileWithHierRow>,
    bin_summary_rows: Vec<BinSummaryRow>,
    tail_summary_rows: Vec<TailL1SummaryRow>,
    bad_key_rows: Vec<BadKeyRecallSummaryRow>,
    static_key_rows: Vec<StaticKeyBoundsRow>,
    worst_key_rows: Vec<WorstKeyMechanismRow>,
    run_metadata: Option<RunMetadata>,
    shared_keys_payloads: Vec<String>,
}

pub fn run_diagnostics_command(config_path: &PathBuf, out_dir: &PathBuf) -> Result<()> {
    let config = load_diagnostics_config(config_path)?;
    prepare_diagnostics_layout(out_dir)?;
    fs::copy(config_path, out_dir.join("configs").join("config.resolved.toml")).with_context(|| {
        format!(
            "failed to copy diagnostics config from {} into {}",
            config_path.display(),
            out_dir.display()
        )
    })?;

    eprintln!("[progress] diagnostics stage 1/4 loading cases");
    let mut all_zero_error = Vec::new();
    let mut all_geometry = Vec::new();
    let mut all_updates = Vec::new();
    let mut all_counters = Vec::new();
    let mut all_sweeps = Vec::new();
    let mut all_failures = Vec::new();
    let mut all_unified = Vec::new();
    let mut all_rankings = Vec::new();
    let mut all_per_key = Vec::new();
    let mut all_per_key_hier = Vec::new();
    let mut all_bin_summaries = Vec::new();
    let mut all_tail_summaries = Vec::new();
    let mut all_bad_key_summaries = Vec::new();
    let mut all_static_key_bounds = Vec::new();
    let mut all_worst_keys = Vec::new();
    let mut run_metadata = None;
    let mut all_shared_keys = Vec::new();

    let case_progress = ProgressReporter::new("diagnostics cases", config.cases.len());
    for (case_index, case) in config.cases.iter().enumerate() {
        let diagnostics = run_case_diagnostics(case_index, case, config.run.seed)?;
        all_zero_error.extend(diagnostics.zero_error_rows);
        all_geometry.extend(diagnostics.geometry_rows);
        all_updates.extend(diagnostics.update_rows);
        all_counters.extend(diagnostics.counter_rows);
        all_sweeps.extend(diagnostics.sweep_rows);
        all_failures.extend(diagnostics.failure_rows);
        all_unified.extend(diagnostics.unified_rows);
        all_rankings.extend(diagnostics.ranking_rows);
        all_per_key.extend(diagnostics.per_key_rows);
        all_per_key_hier.extend(diagnostics.per_key_hier_rows);
        all_bin_summaries.extend(diagnostics.bin_summary_rows);
        all_tail_summaries.extend(diagnostics.tail_summary_rows);
        all_bad_key_summaries.extend(diagnostics.bad_key_rows);
        all_static_key_bounds.extend(diagnostics.static_key_rows);
        all_worst_keys.extend(diagnostics.worst_key_rows);
        run_metadata = run_metadata.or(diagnostics.run_metadata);
        all_shared_keys.extend(diagnostics.shared_keys_payloads);
        case_progress.tick();
    }
    case_progress.finish();
    eprintln!("[progress] diagnostics stage 1/4 complete");

    eprintln!("[progress] diagnostics stage 2/4 writing csv outputs");
    write_zero_error_csv(&all_zero_error, out_dir)?;
    write_single_bit_geometry_csv(&all_geometry, out_dir)?;
    write_syndrome_update_csv(&all_updates, out_dir)?;
    write_counter_distributions_csv(&all_counters, out_dir)?;
    write_threshold_sweep_csv(&all_sweeps, out_dir)?;
    write_failure_taxonomy_csv(&all_failures, out_dir)?;
    write_unified_model_comparison_csv(&all_unified, out_dir)?;
    write_model_ranking_summary_csv(&all_rankings, out_dir)?;
    write_per_key_profiles_csv(&all_per_key, out_dir)?;
    write_per_key_profiles_with_hier_csv(&all_per_key_hier, out_dir)?;
    write_bin_summary_csv(&all_bin_summaries, out_dir)?;
    write_tail_l1_summary_csv(&all_tail_summaries, out_dir)?;
    write_bad_key_recall_summary_csv(&all_bad_key_summaries, out_dir)?;
    write_static_key_bounds_csv(&all_static_key_bounds, out_dir)?;
    write_worst_key_mechanisms_csv(&all_worst_keys, out_dir)?;
    if let Some(metadata) = run_metadata.as_ref() {
        write_run_metadata_json(metadata, out_dir)?;
    }
    write_shared_keys_jsonl(&all_shared_keys, out_dir)?;
    eprintln!("[progress] diagnostics stage 2/4 complete");

    eprintln!("[progress] diagnostics stage 3/4 generating summaries");
    let summary = build_summary(
        &all_zero_error,
        &all_geometry,
        &all_updates,
        &all_sweeps,
        &all_failures,
        &all_rankings,
    );
    write_summary_markdown(out_dir, &summary, &all_sweeps, &all_failures, &all_rankings)?;
    write_summary_json(out_dir, &summary)?;
    eprintln!("[progress] diagnostics stage 3/4 complete");

    eprintln!("[progress] diagnostics stage 4/4 writing manifest");
    let manifest = Manifest {
        run_id: config.run.run_id,
        binary_version: env!("CARGO_PKG_VERSION").to_string(),
        docker_image_tag: default_image_tag(),
        command: format!("dfrcert diagnostics --config {}", config_path.display()),
        config_hash: file_sha256(config_path)?,
        git_commit: detect_git_commit(),
        start_timestamp_utc: now_utc(),
        end_timestamp_utc: None,
        seed: config.run.seed,
        keys_processed: all_zero_error.len() as u64,
        errors_processed: all_failures.len() as u64,
    };
    write_manifest(&out_dir.join("manifest.json"), &manifest)?;
    eprintln!("[progress] diagnostics stage 4/4 complete");

    println!("wrote diagnostics artifacts into {}", out_dir.display());
    Ok(())
}

fn load_diagnostics_config(path: &Path) -> Result<DiagnosticsConfig> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read diagnostics config {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("failed to parse diagnostics config {}", path.display()))
}

fn prepare_diagnostics_layout(out_dir: &Path) -> Result<()> {
    for dir in [out_dir.to_path_buf(), out_dir.join("configs")] {
        fs::create_dir_all(&dir).with_context(|| format!("failed to create {}", dir.display()))?;
    }
    Ok(())
}

fn run_case_diagnostics(case_index: usize, case: &DiagnosticsCase, seed: u64) -> Result<RunCaseDiagnostics> {
    let keys = select_case_keys(case, seed.wrapping_add(case_index as u64 * 10_000))?;
    if keys.is_empty() {
        bail!("diagnostics case '{}' selected no keys", case.name);
    }

    let baseline_decoder = DecoderSection {
        iterations: case.baseline_thresholds.len(),
        thresholds: case.baseline_thresholds.clone(),
        wrong_zero_is_failure: case.wrong_zero_is_failure,
        flip_policy: "all_ge_threshold".to_string(),
    };

    let zero_error_rows = run_zero_error_sanity(case, &keys, &baseline_decoder)?;
    let geometry_rows = run_single_bit_geometry(case, &keys)?;
    let update_rows = run_syndrome_update_consistency(
        case,
        &keys,
        &baseline_decoder,
        seed.wrapping_add(case_index as u64 * 100_000),
    )?;
    let counter_rows = run_counter_distributions(
        case,
        &keys,
        seed.wrapping_add(case_index as u64 * 1_000_000),
    )?;
    let (sweep_rows, failure_rows) = run_threshold_sweeps(
        case,
        &keys,
        seed.wrapping_add(case_index as u64 * 10_000_000),
    )?;
    let comparison = run_unified_model_comparison(
        case,
        &keys,
        &sweep_rows,
        &failure_rows,
        seed.wrapping_add(case_index as u64 * 100_000_000),
    )?;

    Ok(RunCaseDiagnostics {
        zero_error_rows,
        geometry_rows,
        update_rows,
        counter_rows,
        sweep_rows,
        failure_rows,
        unified_rows: comparison.unified_rows,
        ranking_rows: comparison.ranking_rows,
        per_key_rows: comparison.per_key_rows,
        per_key_hier_rows: comparison.per_key_hier_rows,
        bin_summary_rows: comparison.bin_summary_rows,
        tail_summary_rows: comparison.tail_summary_rows,
        bad_key_rows: comparison.bad_key_rows,
        static_key_rows: comparison.static_key_rows,
        worst_key_rows: comparison.worst_key_rows,
        run_metadata: comparison.run_metadata,
        shared_keys_payloads: comparison.shared_keys_payloads,
    })
}

fn select_case_keys(case: &DiagnosticsCase, seed: u64) -> Result<Vec<Key>> {
    let mut keys = match case.key_mode.as_str() {
        "exact_full_keyspace" => enumerate_all_keys(case.r, case.d)?,
        "sampled_keys" => sample_keys_uniform(case.r, case.d, case.key_count.max(1), seed),
        other => bail!("unsupported diagnostics key_mode '{other}'"),
    };
    if case.key_count > 0 && keys.len() > case.key_count {
        keys.truncate(case.key_count);
    }
    Ok(keys)
}

fn run_zero_error_sanity(
    case: &DiagnosticsCase,
    keys: &[Key],
    decoder: &DecoderSection,
) -> Result<Vec<ZeroErrorSanityRow>> {
    let progress = ProgressReporter::new(
        format!("zero-error {}", case.name),
        keys.len(),
    );
    let mut rows = Vec::with_capacity(keys.len());
    for key in keys {
        let runtime = build_key_runtime(key.clone())?;
        let result = decode_with_logs(&runtime, Mask256::empty(), decoder)?;
        rows.push(ZeroErrorSanityRow {
            r: case.r,
            d: case.d,
            key_id: key.key_id,
            success: result.outcome == Outcome::Success && result.final_syndrome_weight == 0,
            final_syndrome_weight: result.final_syndrome_weight,
            failure_type: outcome_label(result.outcome).to_string(),
        });
        progress.tick();
    }
    progress.finish();
    Ok(rows)
}

fn run_single_bit_geometry(case: &DiagnosticsCase, keys: &[Key]) -> Result<Vec<SingleBitGeometryRow>> {
    let take = case.geometry_key_count.max(1).min(keys.len());
    let total = take * (2 * case.r);
    let progress = ProgressReporter::new(format!("single-bit geometry {}", case.name), total);
    let mut rows = Vec::with_capacity(total);
    for key in keys.iter().take(take) {
        let runtime = build_key_runtime(key.clone())?;
        for bit_index in 0..(2 * case.r) {
            let mut error = Mask256::empty();
            error.set(bit_index)?;
            let syndrome = syndrome_for_error(&runtime, error)?;
            let observed_true_counter = runtime.column_masks[bit_index].and(syndrome).popcount() as usize;
            rows.push(SingleBitGeometryRow {
                r: case.r,
                d: case.d,
                key_id: key.key_id,
                bit_index,
                expected_counter: case.d,
                observed_true_counter,
                syndrome_weight: syndrome.popcount(),
                geometry_ok: observed_true_counter == case.d,
                block: bit_index / case.r,
                position: bit_index % case.r,
            });
            progress.tick();
        }
    }
    progress.finish();
    Ok(rows)
}

fn run_syndrome_update_consistency(
    case: &DiagnosticsCase,
    keys: &[Key],
    decoder: &DecoderSection,
    seed: u64,
) -> Result<Vec<SyndromeUpdateRow>> {
    let t_values = case.t_values.iter().copied().filter(|t| *t > 0).collect::<Vec<_>>();
    let take = case.update_key_count.max(1).min(keys.len());
    let mut rows = Vec::new();
    for (t_offset, t) in t_values.iter().copied().enumerate() {
        let errors = generate_error_masks(case, t, seed.wrapping_add(t_offset as u64 * 13))?;
        let total = take * errors.len();
        let progress = ProgressReporter::new(
            format!("syndrome consistency {} t={t}", case.name),
            total,
        );
        for key in keys.iter().take(take) {
            let runtime = build_key_runtime(key.clone())?;
            for (error_id, error) in errors.iter().copied().enumerate() {
                let result = decode_with_logs(&runtime, error, decoder)?;
                rows.extend(build_syndrome_update_rows(case, key.key_id, t, error_id, &runtime, &result)?);
                progress.tick();
            }
        }
        progress.finish();
    }
    Ok(rows)
}

fn build_syndrome_update_rows(
    case: &DiagnosticsCase,
    key_id: u64,
    t: usize,
    error_id: usize,
    runtime: &KeyRuntime,
    result: &DecodeResult,
) -> Result<Vec<SyndromeUpdateRow>> {
    let mut rows = Vec::with_capacity(result.logs.len());
    for log in &result.logs {
        let recomputed = syndrome_for_error(runtime, log.e_mask)?;
        rows.push(SyndromeUpdateRow {
            r: case.r,
            d: case.d,
            t,
            key_id,
            error_id,
            iteration: log.i,
            incremental_syndrome_weight: log.syndrome_weight as u32,
            recomputed_syndrome_weight: recomputed.popcount(),
            syndromes_equal: recomputed == log.syndrome,
            flipped_count: log.flip_count,
        });
    }
    Ok(rows)
}

fn run_counter_distributions(case: &DiagnosticsCase, keys: &[Key], seed: u64) -> Result<Vec<CounterDistributionRow>> {
    let take = case.distribution_key_count.max(1).min(keys.len());
    let mut rows = Vec::new();
    for (t_offset, t) in case.distribution_t_values.iter().copied().enumerate() {
        if t == 0 {
            continue;
        }
        let errors = generate_error_masks(case, t, seed.wrapping_add(t_offset as u64 * 17))?;
        let total = take * errors.len();
        let progress = ProgressReporter::new(
            format!("counter distributions {} t={t}", case.name),
            total,
        );
        for key in keys.iter().take(take) {
            let runtime = build_key_runtime(key.clone())?;
            for (error_id, error) in errors.iter().copied().enumerate() {
                let syndrome = syndrome_for_error(&runtime, error)?;
                for bit_index in 0..(2 * case.r) {
                    let counter = runtime.column_masks[bit_index].and(syndrome).popcount() as usize;
                    rows.push(CounterDistributionRow {
                        r: case.r,
                        d: case.d,
                        t,
                        key_id: key.key_id,
                        error_id,
                        iteration: 0,
                        bit_type: if error.contains(bit_index) {
                            "true".to_string()
                        } else {
                            "clean".to_string()
                        },
                        counter,
                    });
                }
                progress.tick();
            }
        }
        progress.finish();
    }
    Ok(rows)
}

fn run_threshold_sweeps(
    case: &DiagnosticsCase,
    keys: &[Key],
    seed: u64,
) -> Result<(Vec<ThresholdSweepRow>, Vec<FailureTaxonomyRow>)> {
    let mut sweep_rows = Vec::new();
    let mut failure_rows = Vec::new();
    let settings = build_sweep_settings(case);
    let progress = ProgressReporter::new(
        format!("threshold sweeps {}", case.name),
        settings.len() * case.t_values.iter().filter(|t| **t > 0).count(),
    );
    for (setting_index, thresholds) in settings.iter().enumerate() {
        let decoder = DecoderSection {
            iterations: thresholds.len(),
            thresholds: thresholds.clone(),
            wrong_zero_is_failure: case.wrong_zero_is_failure,
            flip_policy: "all_ge_threshold".to_string(),
        };
        let schedule_label = schedule_to_string(thresholds);
        for &t in &case.t_values {
            if t == 0 {
                continue;
            }
            let errors = generate_error_masks(case, t, seed.wrapping_add(setting_index as u64 * 31 + t as u64))?;
            let (summary_row, mut taxonomy_rows) = run_single_threshold_setting(case, keys, &decoder, t, &schedule_label, &errors)?;
            sweep_rows.push(summary_row);
            failure_rows.append(&mut taxonomy_rows);
            progress.tick();
        }
    }
    progress.finish();
    Ok((sweep_rows, failure_rows))
}

fn run_unified_model_comparison(
    case: &DiagnosticsCase,
    keys: &[Key],
    sweep_rows: &[ThresholdSweepRow],
    failure_rows: &[FailureTaxonomyRow],
    seed: u64,
) -> Result<ComparisonArtifacts> {
    let Some(selected) = select_non_saturated_setting(case, sweep_rows) else {
        return Ok(ComparisonArtifacts {
            unified_rows: Vec::new(),
            ranking_rows: Vec::new(),
            per_key_rows: Vec::new(),
            per_key_hier_rows: Vec::new(),
            bin_summary_rows: Vec::new(),
            tail_summary_rows: Vec::new(),
            bad_key_rows: Vec::new(),
            static_key_rows: Vec::new(),
            worst_key_rows: Vec::new(),
            run_metadata: None,
            shared_keys_payloads: Vec::new(),
        });
    };

    let run_id = case.name.clone();
    let decoder = DecoderSection {
        iterations: selected.max_iterations,
        thresholds: parse_schedule(&selected.threshold_schedule)?,
        wrong_zero_is_failure: case.wrong_zero_is_failure,
        flip_policy: "all_ge_threshold".to_string(),
    };
    let errors = generate_error_masks(case, selected.t, seed)?;
    let exact_rows = compute_exact_ph_for_keys(case, keys, &decoder, selected.t, &errors)?;
    let profile_config = diagnostics_profile_config(case, selected.t, &decoder);
    let profiles = compute_profiles(keys, selected.t, &profile_config)?;
    let models = ["M0", "M1", "M2", "M3_old"];
    let bin_budget = keys.len().clamp(1, 8);
    let coord_budget = keys.len().clamp(1, 4);

    let mut model_artifacts = BTreeMap::new();
    for model in models {
        model_artifacts.insert(
            model.to_string(),
            build_model_artifacts(case, model, bin_budget, &profiles)?,
        );
    }
    let (m3_hier_artifacts, hier_assignments) = build_hierarchical_model_artifacts(
        case,
        &profiles,
        &model_artifacts["M2"].bin_map,
        coord_budget,
    )?;
    model_artifacts.insert("M3_hier".to_string(), m3_hier_artifacts);

    let exact_map = exact_rows
        .iter()
        .map(|row| (row.key_id, row.p_h))
        .collect::<BTreeMap<_, _>>();
    let exact_stats_map = exact_rows
        .iter()
        .map(|row| (row.key_id, row.clone()))
        .collect::<BTreeMap<_, _>>();
    let failure_count_map = exact_rows
        .iter()
        .map(|row| (row.key_id, row.failure_count))
        .collect::<BTreeMap<_, _>>();
    let max_clean_counter_map = compute_max_clean_counter_map(case, keys, &errors)?;

    let mut unified_rows = Vec::with_capacity(keys.len());
    let mut per_key_rows = Vec::with_capacity(keys.len());
    let mut per_key_hier_rows = Vec::with_capacity(keys.len());
    for key in keys {
        let profile = profiles
            .iter()
            .find(|profile| profile.key_id == key.key_id)
            .with_context(|| format!("missing profile for key {}", key.key_id))?;
        let hier = hier_assignments
            .get(&key.key_id)
            .with_context(|| format!("missing M3_hier assignment for key {}", key.key_id))?;
        let lambda_map = parse_json_u64_map(&profile.lambda_json);
        let an_u_map = parse_json_f64_map(&profile.an_u_json);
        let gather_map = parse_json_u64_map(&profile.gather_json);
        let lambda_max_count = lambda_map.get(&profile.d_max.to_string()).copied().unwrap_or(0);
        let per_key_row = PerKeyProfileRow {
            run_id: run_id.clone(),
            key_id: key.key_id,
            h0_support: join_support(&key.h0),
            h1_support: join_support(&key.h1),
            p_h: exact_map[&key.key_id],
            failure_count: failure_count_map[&key.key_id],
            error_count: errors.len(),
            M0_score: model_artifacts["M0"].score_map[&key.key_id],
            M0_bin: model_artifacts["M0"].bin_map[&key.key_id],
            M1_score: model_artifacts["M1"].score_map[&key.key_id],
            M1_bin: model_artifacts["M1"].bin_map[&key.key_id],
            M2_score: model_artifacts["M2"].score_map[&key.key_id],
            M2_bin: model_artifacts["M2"].bin_map[&key.key_id],
            M3_old_score: model_artifacts["M3_old"].score_map[&key.key_id],
            M3_old_bin: model_artifacts["M3_old"].bin_map[&key.key_id],
            Dmax: profile.d_max,
            C4_loc: profile.c4_loc,
            Lambda_ge_2_count: sum_lambda_ge(&lambda_map, 2),
            Lambda_ge_3_count: sum_lambda_ge(&lambda_map, 3),
            Lambda_max_count: lambda_max_count,
            near_tail_u1: an_u_map.get("1").copied().unwrap_or(0.0),
            near_tail_u2: an_u_map.get("2").copied().unwrap_or(0.0),
            near_tail_u3: an_u_map.get("3").copied().unwrap_or(0.0),
            near_tail_u4: an_u_map.get("4").copied().unwrap_or(0.0),
            G_2_ell_min_or_count: gather_min_ell_or_count(&gather_map, 2),
            G_3_ell_min_or_count: gather_min_ell_or_count(&gather_map, 3),
            B_fixed_count: profile.b_fixed,
            Omega_max: profile.omega_pair_max,
            max_clean_counter_t1: max_clean_counter_map
                .get(&key.key_id)
                .map(|value| value.to_string())
                .unwrap_or_default(),
            notes: "G_m columns store minimum observed syndrome weight for gathering size m when available.".to_string(),
        };
        unified_rows.push(UnifiedModelComparisonRow {
            run_id: run_id.clone(),
            r: case.r,
            d: case.d,
            t: selected.t,
            key_id: key.key_id,
            p_h: exact_map[&key.key_id],
            M0_score: model_artifacts["M0"].score_map[&key.key_id],
            M1_score: model_artifacts["M1"].score_map[&key.key_id],
            M2_score: model_artifacts["M2"].score_map[&key.key_id],
            M3_old_score: model_artifacts["M3_old"].score_map[&key.key_id],
            M3_hier_rank_score: model_artifacts["M3_hier"].score_map[&key.key_id],
            profile_bin_M0: model_artifacts["M0"].bin_map[&key.key_id],
            profile_bin_M1: model_artifacts["M1"].bin_map[&key.key_id],
            profile_bin_M2: model_artifacts["M2"].bin_map[&key.key_id],
            profile_bin_M3_old: model_artifacts["M3_old"].bin_map[&key.key_id],
            profile_bin_M3_hier: model_artifacts["M3_hier"].bin_map[&key.key_id],
            M3_hier_tuple: hier.tuple_text.clone(),
        });
        per_key_hier_rows.push(PerKeyProfileWithHierRow {
            base: per_key_row.clone(),
            M3_hier_tuple: hier.tuple_text.clone(),
            M3_hier_bin: hier.global_bin_id,
            M3_hier_rank_score: hier.rank_score,
            M3_hier_primary_bin: hier.primary_bin,
            M3_hier_refinement_bin: hier.refinement_bin,
        });
        per_key_rows.push(per_key_row);
    }

    let comparison_models = ["M0", "M1", "M2", "M3_old", "M3_hier"];
    let mut ranking_rows = Vec::new();
    let mut bin_summary_rows = Vec::new();
    let mut tail_summary_rows = Vec::new();
    let mut bad_key_rows = Vec::new();
    let mut static_key_rows = Vec::new();
    for model in comparison_models {
        let artifacts = &model_artifacts[model];
        let bin_rows = build_bin_summary_rows(
            &run_id,
            case,
            selected.t,
            model,
            &artifacts.bin_map,
            &artifacts.bin_label_map,
            &exact_map,
        );
        let ranking_row = build_model_ranking_summary_row(
            &run_id,
            case,
            selected.t,
            model,
            &artifacts.score_map,
            &artifacts.bin_map,
            &bin_rows,
            &exact_map,
            &selected.threshold_schedule,
            errors.len(),
        );
        let tail_row = build_tail_l1_summary_row(
            &run_id,
            case,
            selected.t,
            model,
            &artifacts.score_map,
            &artifacts.bin_map,
            &bin_rows,
            &exact_map,
        );
        ranking_rows.push(ranking_row);
        bad_key_rows.push(build_bad_key_recall_summary_row(
            &run_id,
            case,
            selected.t,
            model,
            &artifacts.score_map,
            &artifacts.bin_map,
            &exact_map,
        ));
        static_key_rows.extend(build_static_key_bounds_rows(
            &run_id,
            model,
            &bin_rows,
            &exact_map,
        ));
        bin_summary_rows.extend(bin_rows);
        tail_summary_rows.push(tail_row);
    }

    let worst_key_rows = build_worst_key_mechanism_rows(
        &run_id,
        selected.t,
        &per_key_hier_rows,
        failure_rows,
        &selected.threshold_schedule,
        &exact_stats_map,
    );

    let shared_keys_payloads = keys
        .iter()
        .map(|key| {
            serde_json::json!({
                "case": case.name,
                "r": case.r,
                "d": case.d,
                "t": selected.t,
                "threshold_schedule": selected.threshold_schedule,
                "key_id": key.key_id,
                "h0": key.h0,
                "h1": key.h1,
            })
            .to_string()
        })
        .collect::<Vec<_>>();

    Ok(ComparisonArtifacts {
        unified_rows,
        ranking_rows,
        per_key_rows,
        per_key_hier_rows,
        bin_summary_rows,
        tail_summary_rows,
        bad_key_rows,
        static_key_rows,
        worst_key_rows,
        run_metadata: Some(RunMetadata {
            run_id: run_id.clone(),
            case_name: case.name.clone(),
            r: case.r,
            d: case.d,
            t: selected.t,
            threshold_schedule: selected.threshold_schedule.clone(),
            key_count: keys.len(),
            error_count_per_key: errors.len(),
            models: comparison_models.iter().map(|model| (*model).to_string()).collect(),
            selected_from_sweep: true,
        }),
        shared_keys_payloads,
    })
}

fn select_non_saturated_setting<'a>(
    case: &DiagnosticsCase,
    sweep_rows: &'a [ThresholdSweepRow],
) -> Option<&'a ThresholdSweepRow> {
    let mut candidates = sweep_rows
        .iter()
        .filter(|row| row.r == case.r && row.d == case.d)
        .filter(|row| (0.001..=0.5).contains(&row.p_h_mean))
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| {
        a.t.cmp(&b.t)
            .then_with(|| (a.p_h_mean - 0.1).abs().total_cmp(&(b.p_h_mean - 0.1).abs()))
            .then_with(|| a.threshold_schedule.cmp(&b.threshold_schedule))
    });
    candidates.into_iter().next()
}

fn parse_schedule(schedule: &str) -> Result<Vec<usize>> {
    schedule
        .split(';')
        .filter(|part| !part.is_empty())
        .map(|part| part.parse::<usize>().with_context(|| format!("invalid threshold '{part}'")))
        .collect()
}

fn compute_exact_ph_for_keys(
    case: &DiagnosticsCase,
    keys: &[Key],
    decoder: &DecoderSection,
    t: usize,
    errors: &[Mask256],
) -> Result<Vec<KeyExactStats>> {
    let progress = ProgressReporter::new(
        format!("unified p_h {} t={t}", case.name),
        keys.len(),
    );
    let mut rows = Vec::with_capacity(keys.len());
    for key in keys {
        let runtime = build_key_runtime(key.clone())?;
        let mut failures = 0usize;
        for error in errors {
            let result = decode_with_logs(&runtime, *error, decoder)?;
            if result.outcome != Outcome::Success {
                failures += 1;
            }
        }
        let error_count = errors.len().max(1);
        let p_h = failures as f64 / error_count as f64;
        rows.push(KeyExactStats {
            key_id: key.key_id,
            p_h,
            failure_count: failures,
            error_count,
        });
        progress.tick();
    }
    progress.finish();
    Ok(rows)
}

fn diagnostics_profile_config(
    case: &DiagnosticsCase,
    t: usize,
    decoder: &DecoderSection,
) -> crate::io::ProfileSection {
    let max_u = t.max(1).min(4);
    let mut fixed_thresholds = decoder.thresholds.clone();
    fixed_thresholds.sort_unstable();
    fixed_thresholds.dedup();
    crate::io::ProfileSection {
        u_values: (1..=max_u).collect(),
        gathering_m_max: case.d.min(3),
        gathering_l_max: case.d.min(3),
        fixedpoint_m_max: case.d.min(3),
        fixedpoint_thresholds: fixed_thresholds,
    }
}

fn build_score_map(model: &str, profiles: &[ProfileRecord]) -> BTreeMap<u64, f64> {
    profiles
        .iter()
        .map(|profile| (profile.key_id, model_score(model, profile)))
        .collect()
}

fn build_model_artifacts(
    case: &DiagnosticsCase,
    model: &str,
    bin_budget: usize,
    profiles: &[ProfileRecord],
) -> Result<ModelArtifacts> {
    let score_map = build_score_map(model, profiles);
    let bins = build_bins_for_model(
        &format!("diagnostics_unified_{}_{}", case.r, case.d),
        profiles,
        bin_budget,
        model,
    )?;
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

fn build_hierarchical_model_artifacts(
    case: &DiagnosticsCase,
    profiles: &[ProfileRecord],
    m2_bin_map: &BTreeMap<u64, usize>,
    coord_budget: usize,
) -> Result<(ModelArtifacts, BTreeMap<u64, HierKeyAssignment>)> {
    let c4_bins = quantize_profile_coordinate(
        profiles,
        coord_budget,
        |profile| profile.c4_loc as f64,
    );
    let dmax_bins = quantize_profile_coordinate(
        profiles,
        coord_budget,
        |profile| profile.d_max as f64,
    );
    let gather_bins = quantize_profile_coordinate(
        profiles,
        coord_budget,
        |profile| max_json_numeric_value_u64(&parse_json_u64_map(&profile.gather_json)) as f64,
    );
    let absorb_bins = quantize_profile_coordinate(
        profiles,
        coord_budget,
        |profile| profile.b_fixed as f64,
    );
    let omega_bins = quantize_profile_coordinate(
        profiles,
        coord_budget,
        |profile| profile.omega_pair_max as f64,
    );

    let mut tuples = profiles
        .iter()
        .map(|profile| {
            let tuple = (
                *m2_bin_map.get(&profile.key_id).unwrap_or(&0),
                *c4_bins.get(&profile.key_id).unwrap_or(&0),
                *dmax_bins.get(&profile.key_id).unwrap_or(&0),
                *gather_bins.get(&profile.key_id).unwrap_or(&0),
                *absorb_bins.get(&profile.key_id).unwrap_or(&0),
                *omega_bins.get(&profile.key_id).unwrap_or(&0),
            );
            (profile.key_id, tuple)
        })
        .collect::<Vec<_>>();
    tuples.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

    let mut global_order = BTreeSet::new();
    let mut refinement_order_by_primary: BTreeMap<usize, BTreeSet<(usize, usize, usize, usize, usize)>> =
        BTreeMap::new();
    for (_, tuple) in &tuples {
        global_order.insert(*tuple);
        refinement_order_by_primary
            .entry(tuple.0)
            .or_default()
            .insert((tuple.1, tuple.2, tuple.3, tuple.4, tuple.5));
    }
    let global_rank = global_order
        .iter()
        .enumerate()
        .map(|(index, tuple)| (*tuple, index))
        .collect::<BTreeMap<_, _>>();
    let refinement_rank = refinement_order_by_primary
        .iter()
        .map(|(primary, tuples)| {
            let map = tuples
                .iter()
                .enumerate()
                .map(|(index, tuple)| (*tuple, index))
                .collect::<BTreeMap<_, _>>();
            (*primary, map)
        })
        .collect::<BTreeMap<_, _>>();

    let mut score_map = BTreeMap::new();
    let mut bin_map = BTreeMap::new();
    let mut bin_label_map = BTreeMap::new();
    let mut assignments = BTreeMap::new();
    for (key_id, tuple) in tuples {
        let refinement_tuple = (tuple.1, tuple.2, tuple.3, tuple.4, tuple.5);
        let global_bin_id = global_rank[&tuple];
        let refinement_bin = refinement_rank[&tuple.0][&refinement_tuple];
        let tuple_text = format!(
            "M2={}|C4={}|Dmax={}|G={}|B={}|Omega={}",
            tuple.0, tuple.1, tuple.2, tuple.3, tuple.4, tuple.5
        );
        score_map.insert(key_id, global_bin_id as f64);
        bin_map.insert(key_id, global_bin_id);
        bin_label_map.insert(global_bin_id, tuple_text.clone());
        assignments.insert(
            key_id,
            HierKeyAssignment {
                tuple_text,
                global_bin_id,
                primary_bin: tuple.0,
                refinement_bin,
                rank_score: global_bin_id as f64,
            },
        );
    }

    let _ = case;
    Ok((
        ModelArtifacts {
            score_map,
            bin_map,
            bin_label_map,
        },
        assignments,
    ))
}

fn quantize_profile_coordinate(
    profiles: &[ProfileRecord],
    bin_budget: usize,
    value_fn: impl Fn(&ProfileRecord) -> f64,
) -> BTreeMap<u64, usize> {
    let mut pairs = profiles
        .iter()
        .map(|profile| (profile.key_id, value_fn(profile)))
        .collect::<Vec<_>>();
    pairs.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    let n = pairs.len().max(1);
    let bins = bin_budget.max(1).min(n);
    pairs
        .into_iter()
        .enumerate()
        .map(|(index, (key_id, _))| (key_id, index * bins / n))
        .collect()
}

fn build_model_ranking_summary_row(
    run_id: &str,
    case: &DiagnosticsCase,
    t: usize,
    model: &str,
    score_map: &BTreeMap<u64, f64>,
    bin_map: &BTreeMap<u64, usize>,
    bin_rows: &[BinSummaryRow],
    exact_map: &BTreeMap<u64, f64>,
    threshold_schedule: &str,
    error_count: usize,
) -> ModelRankingSummaryRow {
    let mut pairs = exact_map
        .iter()
        .map(|(key_id, p_h)| (*key_id, *p_h, *score_map.get(key_id).unwrap_or(&0.0)))
        .collect::<Vec<_>>();
    pairs.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
    let p_h_values = exact_map.values().copied().collect::<Vec<_>>();
    let mut aligned = exact_map
        .iter()
        .map(|(key_id, p_h)| (*key_id, *p_h, *score_map.get(key_id).unwrap_or(&0.0)))
        .collect::<Vec<_>>();
    aligned.sort_by(|a, b| a.0.cmp(&b.0));
    let score_values = aligned.iter().map(|(_, _, score)| *score).collect::<Vec<_>>();
    let exact_values = aligned.iter().map(|(_, p_h, _)| *p_h).collect::<Vec<_>>();
    let direct_average_bound = direct_average_bound(bin_rows);
    let empirical_average = mean_f64(&p_h_values);

    ModelRankingSummaryRow {
        run_id: run_id.to_string(),
        r: case.r,
        d: case.d,
        t,
        model: model.to_string(),
        key_count: pairs.len(),
        p_h_mean: mean_f64(&p_h_values),
        p_h_min: p_h_values.iter().copied().min_by(|a, b| a.total_cmp(b)).unwrap_or(0.0),
        p_h_median: {
            let mut sorted = p_h_values.clone();
            sorted.sort_by(|a, b| a.total_cmp(b));
            median_sorted(&sorted)
        },
        p_h_max: p_h_values.iter().copied().max_by(|a, b| a.total_cmp(b)).unwrap_or(0.0),
        spearman_with_p_h: spearman_rank_corr(&score_values, &exact_values),
        pearson_with_p_h: pearson(&score_values, &exact_values),
        kendall_with_p_h: kendall_tau_b(&score_values, &exact_values),
        auc_bad_any: auc_binary(&score_values, &exact_values.iter().map(|value| *value > 0.0).collect::<Vec<_>>()),
        tail_l1_error: ranking_tail_l1_error(&pairs, exact_map),
        top_10_percent_recall: top_fraction_recall(&pairs, exact_map, 0.10),
        direct_average_bound,
        empirical_average,
        direct_bound_looseness: safe_ratio_f64(direct_average_bound, empirical_average),
        comments: format!(
            "same-key comparison; threshold_schedule={threshold_schedule}; error_count_per_key={error_count}; highest_bin={}",
            bin_map.values().copied().max().unwrap_or(0)
        ),
    }
}

fn build_bin_summary_rows(
    run_id: &str,
    case: &DiagnosticsCase,
    t: usize,
    model: &str,
    bin_map: &BTreeMap<u64, usize>,
    bin_label_map: &BTreeMap<usize, String>,
    exact_map: &BTreeMap<u64, f64>,
) -> Vec<BinSummaryRow> {
    let total = exact_map.len().max(1) as f64;
    let mut grouped = BTreeMap::<usize, Vec<f64>>::new();
    for (key_id, p_h) in exact_map {
        grouped.entry(*bin_map.get(key_id).unwrap_or(&0)).or_default().push(*p_h);
    }
    grouped
        .into_iter()
        .map(|(bin_id, values)| BinSummaryRow {
            run_id: run_id.to_string(),
            r: case.r,
            d: case.d,
            t,
            model: model.to_string(),
            bin_id,
            bin_label: bin_label_map.get(&bin_id).cloned().unwrap_or_default(),
            key_count: values.len(),
            bin_mass: values.len() as f64 / total,
            p_bin_mean: mean_f64(&values),
            p_bin_max: values
                .iter()
                .copied()
                .max_by(|a, b| a.total_cmp(b))
                .unwrap_or(0.0),
        })
        .collect()
}

fn build_tail_l1_summary_row(
    run_id: &str,
    case: &DiagnosticsCase,
    t: usize,
    model: &str,
    score_map: &BTreeMap<u64, f64>,
    _bin_map: &BTreeMap<u64, usize>,
    bin_rows: &[BinSummaryRow],
    exact_map: &BTreeMap<u64, f64>,
) -> TailL1SummaryRow {
    let mut aligned = exact_map
        .iter()
        .map(|(key_id, p_h)| (*key_id, *p_h, *score_map.get(key_id).unwrap_or(&0.0)))
        .collect::<Vec<_>>();
    aligned.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
    let scores_by_key = exact_map
        .iter()
        .map(|(key_id, p_h)| (*score_map.get(key_id).unwrap_or(&0.0), *p_h))
        .collect::<Vec<_>>();
    let score_values = scores_by_key.iter().map(|(score, _)| *score).collect::<Vec<_>>();
    let exact_values = scores_by_key.iter().map(|(_, p_h)| *p_h).collect::<Vec<_>>();
    let empirical_average = mean_f64(&exact_values);
    let direct_average_bound = direct_average_bound(bin_rows);
    let empirical_tail = |tau| exact_values.iter().filter(|value| **value > tau).count() as f64 / exact_values.len().max(1) as f64;
    let model_tail = |tau| bin_rows.iter().filter(|row| row.p_bin_max > tau).map(|row| row.bin_mass).sum::<f64>();

    TailL1SummaryRow {
        run_id: run_id.to_string(),
        r: case.r,
        d: case.d,
        t,
        model: model.to_string(),
        key_count: exact_values.len(),
        empirical_average,
        direct_average_bound,
        direct_bound_looseness: format_optional_f64(safe_ratio_f64(direct_average_bound, empirical_average)),
        tail_l1_error: ranking_tail_l1_error(&aligned, exact_map),
        spearman_with_p_h: spearman_rank_corr(&score_values, &exact_values),
        pearson_with_p_h: pearson(&score_values, &exact_values),
        kendall_with_p_h: kendall_tau_b(&score_values, &exact_values),
        auc_bad_any: auc_binary(&score_values, &exact_values.iter().map(|value| *value > 0.0).collect::<Vec<_>>()),
        empirical_tail_tau_0: empirical_tail(0.0),
        empirical_tail_tau_0_01: empirical_tail(0.01),
        empirical_tail_tau_0_05: empirical_tail(0.05),
        empirical_tail_tau_0_1: empirical_tail(0.1),
        empirical_tail_tau_0_25: empirical_tail(0.25),
        empirical_tail_tau_0_5: empirical_tail(0.5),
        empirical_tail_tau_0_75: empirical_tail(0.75),
        empirical_tail_tau_0_99: empirical_tail(0.99),
        model_tail_tau_0: model_tail(0.0),
        model_tail_tau_0_01: model_tail(0.01),
        model_tail_tau_0_05: model_tail(0.05),
        model_tail_tau_0_1: model_tail(0.1),
        model_tail_tau_0_25: model_tail(0.25),
        model_tail_tau_0_5: model_tail(0.5),
        model_tail_tau_0_75: model_tail(0.75),
        model_tail_tau_0_99: model_tail(0.99),
    }
}

fn build_bad_key_recall_summary_row(
    run_id: &str,
    case: &DiagnosticsCase,
    t: usize,
    model: &str,
    score_map: &BTreeMap<u64, f64>,
    bin_map: &BTreeMap<u64, usize>,
    exact_map: &BTreeMap<u64, f64>,
) -> BadKeyRecallSummaryRow {
    let mut ranked = exact_map
        .iter()
        .map(|(key_id, p_h)| (*key_id, *p_h, *score_map.get(key_id).unwrap_or(&0.0)))
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
    let top_k = ((ranked.len() as f64 * 0.10).ceil() as usize).max(1).min(ranked.len());
    let highest_bin = bin_map.values().copied().max().unwrap_or(0);
    let highest_bin_keys = bin_map
        .iter()
        .filter_map(|(key_id, bin_id)| (*bin_id == highest_bin).then_some(*key_id))
        .collect::<HashSet<_>>();
    let top_k_keys = ranked.iter().take(top_k).map(|(key_id, _, _)| *key_id).collect::<HashSet<_>>();
    let bad_any = exact_map.iter().filter_map(|(key_id, p_h)| (*p_h > 0.0).then_some(*key_id)).collect::<HashSet<_>>();
    let bad_high = exact_map.iter().filter_map(|(key_id, p_h)| (*p_h >= 0.5).then_some(*key_id)).collect::<HashSet<_>>();
    let bad_cat = exact_map.iter().filter_map(|(key_id, p_h)| ((*p_h - 1.0).abs() <= 1e-15).then_some(*key_id)).collect::<HashSet<_>>();
    BadKeyRecallSummaryRow {
        run_id: run_id.to_string(),
        r: case.r,
        d: case.d,
        t,
        model: model.to_string(),
        top_k,
        highest_bin,
        bad_any_recall_at_highest_bin: set_recall(&highest_bin_keys, &bad_any),
        bad_high_recall_at_highest_bin: set_recall(&highest_bin_keys, &bad_high),
        bad_catastrophic_recall_at_highest_bin: set_recall(&highest_bin_keys, &bad_cat),
        bad_any_precision_at_highest_bin: set_precision(&highest_bin_keys, &bad_any),
        bad_high_precision_at_highest_bin: set_precision(&highest_bin_keys, &bad_high),
        bad_catastrophic_precision_at_highest_bin: set_precision(&highest_bin_keys, &bad_cat),
        bad_any_recall_at_top_k: set_recall(&top_k_keys, &bad_any),
        bad_high_recall_at_top_k: set_recall(&top_k_keys, &bad_high),
        bad_catastrophic_recall_at_top_k: set_recall(&top_k_keys, &bad_cat),
    }
}

fn build_static_key_bounds_rows(
    run_id: &str,
    model: &str,
    bin_rows: &[BinSummaryRow],
    exact_map: &BTreeMap<u64, f64>,
) -> Vec<StaticKeyBoundsRow> {
    let empirical_average = mean_f64(&exact_map.values().copied().collect::<Vec<_>>());
    [1usize, 2, 5, 10, 25, 50, 100]
        .into_iter()
        .map(|q| {
            let empirical_fail_q = exact_map
                .values()
                .map(|p_h| 1.0 - (1.0 - p_h).powf(q as f64))
                .sum::<f64>()
                / exact_map.len().max(1) as f64;
            let profile_bound_fail_q = bin_rows
                .iter()
                .map(|row| row.bin_mass * (1.0 - (1.0 - row.p_bin_max).powf(q as f64)))
                .sum::<f64>();
            let average_union_bound = (q as f64 * empirical_average).min(1.0);
            StaticKeyBoundsRow {
                run_id: run_id.to_string(),
                model: model.to_string(),
                q,
                empirical_fail_q,
                profile_bound_fail_q,
                average_union_bound,
                profile_bound_looseness: format_optional_f64(safe_ratio_f64(
                    profile_bound_fail_q,
                    empirical_fail_q,
                )),
            }
        })
        .collect()
}

fn build_worst_key_mechanism_rows(
    run_id: &str,
    t: usize,
    per_key_hier_rows: &[PerKeyProfileWithHierRow],
    failure_rows: &[FailureTaxonomyRow],
    threshold_schedule: &str,
    exact_stats_map: &BTreeMap<u64, KeyExactStats>,
) -> Vec<WorstKeyMechanismRow> {
    let mut selected = per_key_hier_rows
        .iter()
        .filter(|row| {
            if t == 1 {
                row.base.p_h > 0.0
            } else {
                true
            }
        })
        .cloned()
        .collect::<Vec<_>>();
    selected.sort_by(|a, b| {
        b.base
            .p_h
            .total_cmp(&a.base.p_h)
            .then_with(|| a.base.key_id.cmp(&b.base.key_id))
    });
    if t != 1 && selected.len() > 10 {
        selected.truncate(10);
    }
    selected
        .into_iter()
        .map(|row| {
            let rows = failure_rows
                .iter()
                .filter(|failure| {
                    failure.key_id == row.base.key_id
                        && failure.t == t
                        && failure.threshold_schedule == threshold_schedule
                })
                .collect::<Vec<_>>();
            let dominant_failure_type = dominant_failure_type(&rows);
            let timeout_count = rows.iter().filter(|failure| failure.outcome == "timeout").count();
            let wrong_zero_count = rows.iter().filter(|failure| failure.outcome == "wrong_zero").count();
            let divergence_like_count = rows
                .iter()
                .filter(|failure| failure.outcome == "divergence_like")
                .count();
            WorstKeyMechanismRow {
                run_id: run_id.to_string(),
                key_id: row.base.key_id,
                p_h: row.base.p_h,
                failure_count: exact_stats_map
                    .get(&row.base.key_id)
                    .map(|stats| stats.failure_count)
                    .unwrap_or(row.base.failure_count),
                error_count: exact_stats_map
                    .get(&row.base.key_id)
                    .map(|stats| stats.error_count)
                    .unwrap_or(row.base.error_count),
                M2_bin: row.base.M2_bin,
                M3_hier_tuple: row.M3_hier_tuple,
                Dmax: row.base.Dmax,
                C4_loc: row.base.C4_loc,
                near_tail_u1: row.base.near_tail_u1,
                near_tail_u2: row.base.near_tail_u2,
                near_tail_u3: row.base.near_tail_u3,
                near_tail_u4: row.base.near_tail_u4,
                B_fixed_count: row.base.B_fixed_count,
                Omega_max: row.base.Omega_max,
                dominant_failure_type,
                timeout_count,
                wrong_zero_count,
                divergence_like_count,
                max_clean_counter: row.base.max_clean_counter_t1,
                explanation_hint: explanation_hint(row.base.M2_bin, row.base.Dmax, row.base.B_fixed_count, row.base.Omega_max),
            }
        })
        .collect()
}

fn ranking_tail_l1_error(
    score_ranked: &[(u64, f64, f64)],
    exact_map: &BTreeMap<u64, f64>,
) -> f64 {
    let mut exact_ranked = exact_map.iter().map(|(key_id, p_h)| (*key_id, *p_h)).collect::<Vec<_>>();
    exact_ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let total_exact = exact_ranked.iter().map(|(_, p_h)| *p_h).sum::<f64>().max(1e-15);

    let mut exact_prefix = Vec::with_capacity(exact_ranked.len());
    let mut acc = 0.0;
    for (_, p_h) in &exact_ranked {
        acc += *p_h;
        exact_prefix.push(acc / total_exact);
    }

    let mut score_prefix = Vec::with_capacity(score_ranked.len());
    let mut score_acc = 0.0;
    for (key_id, _, _) in score_ranked {
        score_acc += exact_map[key_id];
        score_prefix.push(score_acc / total_exact);
    }

    exact_prefix
        .iter()
        .zip(score_prefix.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .sum::<f64>()
        / exact_prefix.len().max(1) as f64
}

fn top_fraction_recall(
    score_ranked: &[(u64, f64, f64)],
    exact_map: &BTreeMap<u64, f64>,
    fraction: f64,
) -> f64 {
    let k = ((score_ranked.len() as f64 * fraction).ceil() as usize)
        .max(1)
        .min(score_ranked.len());
    let score_top = score_ranked.iter().take(k).map(|(key_id, _, _)| *key_id).collect::<HashSet<_>>();
    let mut exact_ranked = exact_map.iter().map(|(key_id, p_h)| (*key_id, *p_h)).collect::<Vec<_>>();
    exact_ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let exact_top = exact_ranked.iter().take(k).map(|(key_id, _)| *key_id).collect::<HashSet<_>>();
    score_top.intersection(&exact_top).count() as f64 / exact_top.len().max(1) as f64
}

fn compute_max_clean_counter_map(
    case: &DiagnosticsCase,
    keys: &[Key],
    errors: &[Mask256],
) -> Result<BTreeMap<u64, usize>> {
    let mut out = BTreeMap::new();
    for key in keys {
        let runtime = build_key_runtime(key.clone())?;
        let mut max_clean = 0usize;
        for error in errors {
            let syndrome = syndrome_for_error(&runtime, *error)?;
            for bit_index in 0..(2 * case.r) {
                if error.contains(bit_index) {
                    continue;
                }
                let counter = runtime.column_masks[bit_index].and(syndrome).popcount() as usize;
                max_clean = max_clean.max(counter);
            }
        }
        out.insert(key.key_id, max_clean);
    }
    Ok(out)
}

fn parse_json_u64_map(payload: &str) -> BTreeMap<String, u64> {
    serde_json::from_str(payload).unwrap_or_default()
}

fn parse_json_f64_map(payload: &str) -> BTreeMap<String, f64> {
    serde_json::from_str(payload).unwrap_or_default()
}

fn max_json_numeric_value_u64(values: &BTreeMap<String, u64>) -> u64 {
    values.values().copied().max().unwrap_or(0)
}

fn join_support(values: &[u16]) -> String {
    values.iter().map(u16::to_string).collect::<Vec<_>>().join(";")
}

fn sum_lambda_ge(lambda_map: &BTreeMap<String, u64>, min_k: u32) -> u64 {
    lambda_map
        .iter()
        .filter_map(|(key, value)| key.parse::<u32>().ok().map(|parsed| (parsed, *value)))
        .filter(|(parsed, _)| *parsed >= min_k)
        .map(|(_, value)| value)
        .sum()
}

fn gather_min_ell_or_count(gather_map: &BTreeMap<String, u64>, m: usize) -> String {
    let mut ells = gather_map
        .iter()
        .filter_map(|(key, _)| {
            let mut parts = key.split(':');
            let parsed_m = parts.next()?.parse::<usize>().ok()?;
            let parsed_ell = parts.next()?.parse::<usize>().ok()?;
            (parsed_m == m).then_some(parsed_ell)
        })
        .collect::<Vec<_>>();
    ells.sort_unstable();
    ells.first().map(|value| value.to_string()).unwrap_or_default()
}

fn direct_average_bound(bin_rows: &[BinSummaryRow]) -> f64 {
    bin_rows.iter().map(|row| row.bin_mass * row.p_bin_max).sum()
}

fn safe_ratio_f64(numer: f64, denom: f64) -> Option<f64> {
    (denom.abs() > 1e-15).then_some(numer / denom)
}

fn format_optional_f64(value: Option<f64>) -> String {
    value.map(|inner| format!("{inner:.17}")).unwrap_or_else(|| "NA".to_string())
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

fn auc_binary(scores: &[f64], labels: &[bool]) -> f64 {
    if scores.len() != labels.len() || scores.is_empty() {
        return 0.0;
    }
    let positives = scores
        .iter()
        .zip(labels.iter())
        .filter_map(|(score, label)| (*label).then_some(*score))
        .collect::<Vec<_>>();
    let negatives = scores
        .iter()
        .zip(labels.iter())
        .filter_map(|(score, label)| (!*label).then_some(*score))
        .collect::<Vec<_>>();
    if positives.is_empty() || negatives.is_empty() {
        return 0.5;
    }
    let mut wins = 0.0;
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

fn kendall_tau_b(xs: &[f64], ys: &[f64]) -> f64 {
    if xs.len() != ys.len() || xs.len() < 2 {
        return 0.0;
    }
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
    if denom <= 1e-15 {
        0.0
    } else {
        (concordant - discordant) / denom
    }
}

fn dominant_failure_type(rows: &[&FailureTaxonomyRow]) -> String {
    let mut counts = BTreeMap::new();
    for row in rows.iter().filter(|row| row.outcome != "success") {
        *counts.entry(row.outcome.clone()).or_insert(0usize) += 1;
    }
    counts
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)))
        .map(|(label, _)| label)
        .unwrap_or_else(|| "success".to_string())
}

fn explanation_hint(m2_bin: usize, dmax: u32, b_fixed: u64, omega_max: u32) -> String {
    let mut signals = 0usize;
    if m2_bin > 0 {
        signals += 1;
    }
    if dmax >= 3 || omega_max >= 3 {
        signals += 1;
    }
    if b_fixed > 0 {
        signals += 1;
    }
    if signals >= 2 {
        "mixed".to_string()
    } else if m2_bin > 0 {
        "high M2 overlap".to_string()
    } else if dmax >= 3 || omega_max >= 3 {
        "high Dmax".to_string()
    } else if b_fixed > 0 {
        "fixed-point indicator".to_string()
    } else {
        "unknown".to_string()
    }
}

fn run_single_threshold_setting(
    case: &DiagnosticsCase,
    keys: &[Key],
    decoder: &DecoderSection,
    t: usize,
    schedule_label: &str,
    errors: &[Mask256],
) -> Result<(ThresholdSweepRow, Vec<FailureTaxonomyRow>)> {
    let mut key_failure_rates = Vec::with_capacity(keys.len());
    let mut total_success = 0u128;
    let mut total_timeout = 0u128;
    let mut total_wrong_zero = 0u128;
    let mut total_nonzero_halt = 0u128;
    let mut total_errors = 0u128;
    let mut final_residual_sum = 0u128;
    let mut final_syndrome_sum = 0u128;
    let mut taxonomy_rows = Vec::with_capacity(keys.len() * errors.len());

    for key in keys {
        let runtime = build_key_runtime(key.clone())?;
        let mut key_failures = 0u128;
        for (error_id, error) in errors.iter().copied().enumerate() {
            let initial_syndrome = syndrome_for_error(&runtime, error)?;
            let result = decode_with_logs(&runtime, error, decoder)?;
            total_errors += 1;
            final_residual_sum += u128::from(result.final_residual_weight);
            final_syndrome_sum += u128::from(result.final_syndrome_weight);
            match result.outcome {
                Outcome::Success => total_success += 1,
                Outcome::Timeout => {
                    total_timeout += 1;
                    key_failures += 1;
                }
                Outcome::WrongZero => {
                    total_wrong_zero += 1;
                    key_failures += 1;
                }
                Outcome::NonzeroHalt => {
                    total_nonzero_halt += 1;
                    key_failures += 1;
                }
            }
            taxonomy_rows.push(FailureTaxonomyRow {
                r: case.r,
                d: case.d,
                t,
                key_id: key.key_id,
                error_id,
                threshold_schedule: schedule_label.to_string(),
                outcome: classify_outcome(case, t, &result),
                iterations_used: result.iterations,
                initial_syndrome_weight: initial_syndrome.popcount(),
                final_syndrome_weight: result.final_syndrome_weight,
                final_residual_weight: result.final_residual_weight,
                flipped_total: result.logs.iter().map(|log| u32::from(log.flip_count)).sum(),
                wrong_zero: result.outcome == Outcome::WrongZero,
            });
        }
        let key_total = errors.len() as u128;
        key_failure_rates.push(if key_total == 0 {
            0.0
        } else {
            key_failures as f64 / key_total as f64
        });
    }

    key_failure_rates.sort_by(|a, b| a.total_cmp(b));
    let p_h_mean = mean_f64(&key_failure_rates);
    let p_h_min = key_failure_rates.first().copied().unwrap_or(0.0);
    let p_h_max = key_failure_rates.last().copied().unwrap_or(0.0);
    let p_h_median = median_sorted(&key_failure_rates);

    let total_errors_f = total_errors.max(1) as f64;
    let row = ThresholdSweepRow {
        r: case.r,
        d: case.d,
        t,
        key_count: keys.len(),
        error_count_per_key: errors.len(),
        threshold_schedule: schedule_label.to_string(),
        max_iterations: decoder.iterations,
        p_h_mean,
        p_h_min,
        p_h_median,
        p_h_max,
        success_rate: total_success as f64 / total_errors_f,
        timeout_rate: total_timeout as f64 / total_errors_f,
        wrong_zero_rate: total_wrong_zero as f64 / total_errors_f,
        nonzero_halt_rate: total_nonzero_halt as f64 / total_errors_f,
        mean_final_residual_weight: final_residual_sum as f64 / total_errors_f,
        mean_final_syndrome_weight: final_syndrome_sum as f64 / total_errors_f,
    };

    Ok((row, taxonomy_rows))
}

fn build_sweep_settings(case: &DiagnosticsCase) -> Vec<Vec<usize>> {
    let mut settings = Vec::new();
    for &iterations in &case.sweep_max_iterations {
        for &threshold in &case.sweep_constant_thresholds {
            settings.push(vec![threshold; iterations]);
        }
    }
    for schedule in &case.sweep_schedules {
        if !schedule.is_empty() {
            settings.push(schedule.clone());
        }
    }
    let mut dedup = HashSet::new();
    settings
        .into_iter()
        .filter(|schedule| dedup.insert(schedule_to_string(schedule)))
        .collect()
}

fn generate_error_masks(case: &DiagnosticsCase, t: usize, seed: u64) -> Result<Vec<Mask256>> {
    let bit_count = 2 * case.r;
    if t > bit_count {
        bail!("diagnostics t={t} exceeds bit count {bit_count}");
    }
    if t == 0 {
        return Ok(vec![Mask256::empty()]);
    }

    if t <= case.exact_error_weight_max {
        let combos = iter_combinations(bit_count, t)?
            .map(mask_from_combo)
            .collect::<Result<Vec<_>>>()?;
        return Ok(combos);
    }

    let target = case.error_sample_count.max(1);
    sample_error_masks(bit_count, t, target, seed)
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
        for idx in 0..t {
            let swap_idx = idx + rng.gen_range(bit_count - idx);
            values.swap(idx, swap_idx);
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
        let mut mask = Mask256::empty();
        for &bit in &values[..t] {
            mask.set(bit)?;
        }
        out.push(mask);
    }
    if out.is_empty() {
        bail!("failed to sample any error masks for bit_count={bit_count}, t={t}");
    }
    Ok(out)
}

fn classify_outcome(case: &DiagnosticsCase, t: usize, result: &DecodeResult) -> String {
    let divergence_like = result
        .logs
        .iter()
        .any(|log| {
            usize::from(log.flip_count) >= case.divergence_flip_threshold
                || usize::from(log.residual_weight) >= case.divergence_residual_threshold
        })
        || usize::from(result.final_residual_weight) >= case.divergence_residual_threshold.max(t);
    if divergence_like && result.outcome != Outcome::Success {
        "divergence_like".to_string()
    } else {
        outcome_label(result.outcome).to_string()
    }
}

fn outcome_label(outcome: Outcome) -> &'static str {
    match outcome {
        Outcome::Success => "success",
        Outcome::Timeout => "timeout",
        Outcome::WrongZero => "wrong_zero",
        Outcome::NonzeroHalt => "nonzero_halt",
    }
}

fn schedule_to_string(thresholds: &[usize]) -> String {
    thresholds
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(";")
}

fn mean_f64(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn median_sorted(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    }
}

fn spearman_rank_corr(xs: &[f64], ys: &[f64]) -> f64 {
    if xs.len() != ys.len() || xs.is_empty() {
        return 0.0;
    }
    if all_equal(xs) || all_equal(ys) {
        return 0.0;
    }
    let rx = rank_values(xs);
    let ry = rank_values(ys);
    pearson(&rx, &ry)
}

fn rank_values(values: &[f64]) -> Vec<f64> {
    let mut indexed = values.iter().copied().enumerate().collect::<Vec<_>>();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    let mut ranks = vec![0.0; values.len()];
    for (rank, (idx, _)) in indexed.into_iter().enumerate() {
        ranks[idx] = rank as f64 + 1.0;
    }
    ranks
}

fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    if xs.len() != ys.len() || xs.is_empty() {
        return 0.0;
    }
    let n = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den_x = 0.0;
    let mut den_y = 0.0;
    for (x, y) in xs.iter().zip(ys) {
        let dx = *x - mean_x;
        let dy = *y - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }
    if den_x <= 1e-15 || den_y <= 1e-15 {
        0.0
    } else {
        num / (den_x.sqrt() * den_y.sqrt())
    }
}

fn all_equal(values: &[f64]) -> bool {
    values
        .first()
        .map(|first| values.iter().all(|value| (*value - *first).abs() <= 1e-15))
        .unwrap_or(true)
}

fn build_summary(
    zero_error_rows: &[ZeroErrorSanityRow],
    geometry_rows: &[SingleBitGeometryRow],
    update_rows: &[SyndromeUpdateRow],
    sweep_rows: &[ThresholdSweepRow],
    failure_rows: &[FailureTaxonomyRow],
    ranking_rows: &[ModelRankingSummaryRow],
) -> DiagnosticsSummaryJson {
    let zero_error_pass_rate = ratio_count(zero_error_rows.iter().filter(|row| row.success).count(), zero_error_rows.len());
    let geometry_pass_rate = ratio_count(
        geometry_rows.iter().filter(|row| row.geometry_ok).count(),
        geometry_rows.len(),
    );
    let update_pass_rate = ratio_count(
        update_rows.iter().filter(|row| row.syndromes_equal).count(),
        update_rows.len(),
    );
    let non_saturated_settings = sweep_rows
        .iter()
        .filter(|row| (0.001..=0.5).contains(&row.p_h_mean))
        .count();

    let mut failure_counts = BTreeMap::new();
    for row in failure_rows {
        *failure_counts.entry(row.outcome.clone()).or_insert(0usize) += 1;
    }

    let verdict = if zero_error_pass_rate < 1.0 || geometry_pass_rate < 1.0 || update_pass_rate < 1.0 {
        "implementation/convention bug found"
    } else if !ranking_rows.is_empty() {
        "non-saturated regime found and ready for model comparison"
    } else if non_saturated_settings > 0 {
        "non-saturated regime found and ready for model comparison"
    } else if sweep_rows.iter().any(|row| row.t == 1 && row.p_h_mean > 0.95) {
        "threshold schedule problem found"
    } else {
        "toy parameter regime too hard/degenerate"
    }
    .to_string();

    let recommendation = if verdict == "implementation/convention bug found" {
        "Current results are only pipeline sanity checks until the decoder invariants are fixed."
    } else if verdict == "non-saturated regime found and ready for model comparison" {
        "Current results are usable as a small diagnostic experiment and can support same-key model comparison next."
    } else if verdict == "threshold schedule problem found" {
        "Current results are not yet usable for model claims; retune thresholds before paper-facing interpretation."
    } else {
        "Current results are not yet usable for model claims; the toy regime is too saturated and should be softened."
    }
    .to_string();

    DiagnosticsSummaryJson {
        verdict,
        zero_error_pass_rate,
        single_bit_geometry_pass_rate: geometry_pass_rate,
        syndrome_update_pass_rate: update_pass_rate,
        non_saturated_setting_count: non_saturated_settings,
        failure_counts,
        recommendation,
    }
}

fn ratio_count(numer: usize, denom: usize) -> f64 {
    if denom == 0 {
        0.0
    } else {
        numer as f64 / denom as f64
    }
}

fn write_zero_error_csv(rows: &[ZeroErrorSanityRow], out_dir: &Path) -> Result<()> {
    let mut writer = csv_writer(out_dir.join("zero_error_sanity.csv"))?;
    writeln!(writer, "r,d,key_id,success,final_syndrome_weight,failure_type")?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{},{},{}",
            row.r, row.d, row.key_id, row.success, row.final_syndrome_weight, row.failure_type
        )?;
    }
    Ok(())
}

fn write_single_bit_geometry_csv(rows: &[SingleBitGeometryRow], out_dir: &Path) -> Result<()> {
    let mut writer = csv_writer(out_dir.join("single_bit_geometry.csv"))?;
    writeln!(
        writer,
        "r,d,key_id,bit_index,expected_counter,observed_true_counter,syndrome_weight,geometry_ok,block,position"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{},{}",
            row.r,
            row.d,
            row.key_id,
            row.bit_index,
            row.expected_counter,
            row.observed_true_counter,
            row.syndrome_weight,
            row.geometry_ok,
            row.block,
            row.position
        )?;
    }
    Ok(())
}

fn write_syndrome_update_csv(rows: &[SyndromeUpdateRow], out_dir: &Path) -> Result<()> {
    let mut writer = csv_writer(out_dir.join("syndrome_update_consistency.csv"))?;
    writeln!(
        writer,
        "r,d,t,key_id,error_id,iteration,incremental_syndrome_weight,recomputed_syndrome_weight,syndromes_equal,flipped_count"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{},{}",
            row.r,
            row.d,
            row.t,
            row.key_id,
            row.error_id,
            row.iteration,
            row.incremental_syndrome_weight,
            row.recomputed_syndrome_weight,
            row.syndromes_equal,
            row.flipped_count
        )?;
    }
    Ok(())
}

fn write_counter_distributions_csv(rows: &[CounterDistributionRow], out_dir: &Path) -> Result<()> {
    let mut writer = csv_writer(out_dir.join("counter_distributions.csv"))?;
    writeln!(writer, "r,d,t,key_id,error_id,iteration,bit_type,counter")?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{}",
            row.r, row.d, row.t, row.key_id, row.error_id, row.iteration, row.bit_type, row.counter
        )?;
    }
    Ok(())
}

fn write_threshold_sweep_csv(rows: &[ThresholdSweepRow], out_dir: &Path) -> Result<()> {
    let mut writer = csv_writer(out_dir.join("threshold_sweep.csv"))?;
    writeln!(
        writer,
        "r,d,t,key_count,error_count_per_key,threshold_schedule,max_iterations,p_h_mean,p_h_min,p_h_median,p_h_max,success_rate,timeout_rate,wrong_zero_rate,nonzero_halt_rate,mean_final_residual_weight,mean_final_syndrome_weight"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17}",
            row.r,
            row.d,
            row.t,
            row.key_count,
            row.error_count_per_key,
            row.threshold_schedule,
            row.max_iterations,
            row.p_h_mean,
            row.p_h_min,
            row.p_h_median,
            row.p_h_max,
            row.success_rate,
            row.timeout_rate,
            row.wrong_zero_rate,
            row.nonzero_halt_rate,
            row.mean_final_residual_weight,
            row.mean_final_syndrome_weight
        )?;
    }
    Ok(())
}

fn write_failure_taxonomy_csv(rows: &[FailureTaxonomyRow], out_dir: &Path) -> Result<()> {
    let mut writer = csv_writer(out_dir.join("failure_taxonomy.csv"))?;
    writeln!(
        writer,
        "r,d,t,key_id,error_id,threshold_schedule,outcome,iterations_used,initial_syndrome_weight,final_syndrome_weight,final_residual_weight,flipped_total,wrong_zero"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{},{},{},{},{}",
            row.r,
            row.d,
            row.t,
            row.key_id,
            row.error_id,
            row.threshold_schedule,
            row.outcome,
            row.iterations_used,
            row.initial_syndrome_weight,
            row.final_syndrome_weight,
            row.final_residual_weight,
            row.flipped_total,
            row.wrong_zero
        )?;
    }
    Ok(())
}

fn write_unified_model_comparison_csv(rows: &[UnifiedModelComparisonRow], out_dir: &Path) -> Result<()> {
    let mut writer = csv_writer(out_dir.join("unified_model_comparison.csv"))?;
    writeln!(
        writer,
        "run_id,r,d,t,key_id,p_h,M0_score,M1_score,M2_score,M3_old_score,M3_hier_rank_score,profile_bin_M0,profile_bin_M1,profile_bin_M2,profile_bin_M3_old,profile_bin_M3_hier,M3_hier_tuple"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{},{},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{},{},{},{},{},{}",
            row.run_id,
            row.r,
            row.d,
            row.t,
            row.key_id,
            row.p_h,
            row.M0_score,
            row.M1_score,
            row.M2_score,
            row.M3_old_score,
            row.M3_hier_rank_score,
            row.profile_bin_M0,
            row.profile_bin_M1,
            row.profile_bin_M2,
            row.profile_bin_M3_old,
            row.profile_bin_M3_hier,
            csv_escape(&row.M3_hier_tuple)
        )?;
    }
    Ok(())
}

fn write_model_ranking_summary_csv(rows: &[ModelRankingSummaryRow], out_dir: &Path) -> Result<()> {
    let mut writer = csv_writer(out_dir.join("model_ranking_summary.csv"))?;
    writeln!(
        writer,
        "run_id,r,d,t,model,key_count,p_h_mean,p_h_min,p_h_median,p_h_max,spearman_with_p_h,pearson_with_p_h,kendall_with_p_h,auc_bad_any,tail_l1_error,top_10_percent_recall,direct_average_bound,empirical_average,direct_bound_looseness,comments"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{},{},{},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{},{}",
            row.run_id,
            row.r,
            row.d,
            row.t,
            row.model,
            row.key_count,
            row.p_h_mean,
            row.p_h_min,
            row.p_h_median,
            row.p_h_max,
            row.spearman_with_p_h,
            row.pearson_with_p_h,
            row.kendall_with_p_h,
            row.auc_bad_any,
            row.tail_l1_error,
            row.top_10_percent_recall,
            row.direct_average_bound,
            row.empirical_average,
            format_optional_f64(row.direct_bound_looseness),
            csv_escape(&row.comments)
        )?;
    }
    Ok(())
}

fn write_per_key_profiles_csv(rows: &[PerKeyProfileRow], out_dir: &Path) -> Result<()> {
    let mut writer = csv_writer(out_dir.join("per_key_profiles.csv"))?;
    writeln!(
        writer,
        "run_id,key_id,h0_support,h1_support,p_h,failure_count,error_count,M0_score,M0_bin,M1_score,M1_bin,M2_score,M2_bin,M3_old_score,M3_old_bin,Dmax,C4_loc,Lambda_ge_2_count,Lambda_ge_3_count,Lambda_max_count,near_tail_u1,near_tail_u2,near_tail_u3,near_tail_u4,G_2_ell_min_or_count,G_3_ell_min_or_count,B_fixed_count,Omega_max,max_clean_counter_t1,notes"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{},{:.17},{},{},{:.17},{},{:.17},{},{:.17},{},{:.17},{},{},{},{},{},{},{:.17},{:.17},{:.17},{:.17},{},{},{},{},{},{}",
            row.run_id,
            row.key_id,
            csv_escape(&row.h0_support),
            csv_escape(&row.h1_support),
            row.p_h,
            row.failure_count,
            row.error_count,
            row.M0_score,
            row.M0_bin,
            row.M1_score,
            row.M1_bin,
            row.M2_score,
            row.M2_bin,
            row.M3_old_score,
            row.M3_old_bin,
            row.Dmax,
            row.C4_loc,
            row.Lambda_ge_2_count,
            row.Lambda_ge_3_count,
            row.Lambda_max_count,
            row.near_tail_u1,
            row.near_tail_u2,
            row.near_tail_u3,
            row.near_tail_u4,
            csv_escape(&row.G_2_ell_min_or_count),
            csv_escape(&row.G_3_ell_min_or_count),
            row.B_fixed_count,
            row.Omega_max,
            csv_escape(&row.max_clean_counter_t1),
            csv_escape(&row.notes)
        )?;
    }
    Ok(())
}

fn write_per_key_profiles_with_hier_csv(rows: &[PerKeyProfileWithHierRow], out_dir: &Path) -> Result<()> {
    let mut writer = csv_writer(out_dir.join("per_key_profiles_with_m3_hier.csv"))?;
    writeln!(
        writer,
        "run_id,key_id,h0_support,h1_support,p_h,failure_count,error_count,M0_score,M0_bin,M1_score,M1_bin,M2_score,M2_bin,M3_old_score,M3_old_bin,Dmax,C4_loc,Lambda_ge_2_count,Lambda_ge_3_count,Lambda_max_count,near_tail_u1,near_tail_u2,near_tail_u3,near_tail_u4,G_2_ell_min_or_count,G_3_ell_min_or_count,B_fixed_count,Omega_max,max_clean_counter_t1,notes,M3_hier_tuple,M3_hier_bin,M3_hier_rank_score,M3_hier_primary_bin,M3_hier_refinement_bin"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{},{:.17},{},{},{:.17},{},{:.17},{},{:.17},{},{:.17},{},{},{},{},{},{},{:.17},{:.17},{:.17},{:.17},{},{},{},{},{},{},{},{},{:.17},{},{}",
            row.base.run_id,
            row.base.key_id,
            csv_escape(&row.base.h0_support),
            csv_escape(&row.base.h1_support),
            row.base.p_h,
            row.base.failure_count,
            row.base.error_count,
            row.base.M0_score,
            row.base.M0_bin,
            row.base.M1_score,
            row.base.M1_bin,
            row.base.M2_score,
            row.base.M2_bin,
            row.base.M3_old_score,
            row.base.M3_old_bin,
            row.base.Dmax,
            row.base.C4_loc,
            row.base.Lambda_ge_2_count,
            row.base.Lambda_ge_3_count,
            row.base.Lambda_max_count,
            row.base.near_tail_u1,
            row.base.near_tail_u2,
            row.base.near_tail_u3,
            row.base.near_tail_u4,
            csv_escape(&row.base.G_2_ell_min_or_count),
            csv_escape(&row.base.G_3_ell_min_or_count),
            row.base.B_fixed_count,
            row.base.Omega_max,
            csv_escape(&row.base.max_clean_counter_t1),
            csv_escape(&row.base.notes),
            csv_escape(&row.M3_hier_tuple),
            row.M3_hier_bin,
            row.M3_hier_rank_score,
            row.M3_hier_primary_bin,
            row.M3_hier_refinement_bin
        )?;
    }
    Ok(())
}

fn write_bin_summary_csv(rows: &[BinSummaryRow], out_dir: &Path) -> Result<()> {
    let mut writer = csv_writer(out_dir.join("bin_summary.csv"))?;
    writeln!(writer, "run_id,r,d,t,model,bin_id,bin_label,key_count,bin_mass,p_bin_mean,p_bin_max")?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{:.17},{:.17},{:.17}",
            row.run_id,
            row.r,
            row.d,
            row.t,
            row.model,
            row.bin_id,
            csv_escape(&row.bin_label),
            row.key_count,
            row.bin_mass,
            row.p_bin_mean,
            row.p_bin_max
        )?;
    }
    Ok(())
}

fn write_tail_l1_summary_csv(rows: &[TailL1SummaryRow], out_dir: &Path) -> Result<()> {
    let mut writer = csv_writer(out_dir.join("tail_l1_summary.csv"))?;
    writeln!(
        writer,
        "run_id,r,d,t,model,key_count,empirical_average,direct_average_bound,direct_bound_looseness,tail_l1_error,spearman_with_p_h,pearson_with_p_h,kendall_with_p_h,auc_bad_any,empirical_tail_tau_0,empirical_tail_tau_0_01,empirical_tail_tau_0_05,empirical_tail_tau_0_1,empirical_tail_tau_0_25,empirical_tail_tau_0_5,empirical_tail_tau_0_75,empirical_tail_tau_0_99,model_tail_tau_0,model_tail_tau_0_01,model_tail_tau_0_05,model_tail_tau_0_1,model_tail_tau_0_25,model_tail_tau_0_5,model_tail_tau_0_75,model_tail_tau_0_99"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{},{},{},{:.17},{:.17},{},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17}",
            row.run_id,
            row.r,
            row.d,
            row.t,
            row.model,
            row.key_count,
            row.empirical_average,
            row.direct_average_bound,
            row.direct_bound_looseness,
            row.tail_l1_error,
            row.spearman_with_p_h,
            row.pearson_with_p_h,
            row.kendall_with_p_h,
            row.auc_bad_any,
            row.empirical_tail_tau_0,
            row.empirical_tail_tau_0_01,
            row.empirical_tail_tau_0_05,
            row.empirical_tail_tau_0_1,
            row.empirical_tail_tau_0_25,
            row.empirical_tail_tau_0_5,
            row.empirical_tail_tau_0_75,
            row.empirical_tail_tau_0_99,
            row.model_tail_tau_0,
            row.model_tail_tau_0_01,
            row.model_tail_tau_0_05,
            row.model_tail_tau_0_1,
            row.model_tail_tau_0_25,
            row.model_tail_tau_0_5,
            row.model_tail_tau_0_75,
            row.model_tail_tau_0_99
        )?;
    }
    Ok(())
}

fn write_bad_key_recall_summary_csv(rows: &[BadKeyRecallSummaryRow], out_dir: &Path) -> Result<()> {
    let mut writer = csv_writer(out_dir.join("bad_key_recall_summary.csv"))?;
    writeln!(
        writer,
        "run_id,r,d,t,model,top_k,highest_bin,bad_any_recall_at_highest_bin,bad_high_recall_at_highest_bin,bad_catastrophic_recall_at_highest_bin,bad_any_precision_at_highest_bin,bad_high_precision_at_highest_bin,bad_catastrophic_precision_at_highest_bin,bad_any_recall_at_top_k,bad_high_recall_at_top_k,bad_catastrophic_recall_at_top_k"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17}",
            row.run_id,
            row.r,
            row.d,
            row.t,
            row.model,
            row.top_k,
            row.highest_bin,
            row.bad_any_recall_at_highest_bin,
            row.bad_high_recall_at_highest_bin,
            row.bad_catastrophic_recall_at_highest_bin,
            row.bad_any_precision_at_highest_bin,
            row.bad_high_precision_at_highest_bin,
            row.bad_catastrophic_precision_at_highest_bin,
            row.bad_any_recall_at_top_k,
            row.bad_high_recall_at_top_k,
            row.bad_catastrophic_recall_at_top_k
        )?;
    }
    Ok(())
}

fn write_static_key_bounds_csv(rows: &[StaticKeyBoundsRow], out_dir: &Path) -> Result<()> {
    let mut writer = csv_writer(out_dir.join("static_key_bounds.csv"))?;
    writeln!(
        writer,
        "run_id,model,q,empirical_fail_q,profile_bound_fail_q,average_union_bound,profile_bound_looseness"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{},{:.17},{:.17},{:.17},{}",
            row.run_id,
            row.model,
            row.q,
            row.empirical_fail_q,
            row.profile_bound_fail_q,
            row.average_union_bound,
            row.profile_bound_looseness
        )?;
    }
    Ok(())
}

fn write_worst_key_mechanisms_csv(rows: &[WorstKeyMechanismRow], out_dir: &Path) -> Result<()> {
    let mut writer = csv_writer(out_dir.join("worst_key_mechanisms.csv"))?;
    writeln!(
        writer,
        "run_id,key_id,p_h,failure_count,error_count,M2_bin,M3_hier_tuple,Dmax,C4_loc,near_tail_u1,near_tail_u2,near_tail_u3,near_tail_u4,B_fixed_count,Omega_max,dominant_failure_type,timeout_count,wrong_zero_count,divergence_like_count,max_clean_counter,explanation_hint"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{},{},{:.17},{},{},{},{},{},{},{:.17},{:.17},{:.17},{:.17},{},{},{},{},{},{},{},{}",
            row.run_id,
            row.key_id,
            row.p_h,
            row.failure_count,
            row.error_count,
            row.M2_bin,
            csv_escape(&row.M3_hier_tuple),
            row.Dmax,
            row.C4_loc,
            row.near_tail_u1,
            row.near_tail_u2,
            row.near_tail_u3,
            row.near_tail_u4,
            row.B_fixed_count,
            row.Omega_max,
            row.dominant_failure_type,
            row.timeout_count,
            row.wrong_zero_count,
            row.divergence_like_count,
            csv_escape(&row.max_clean_counter),
            csv_escape(&row.explanation_hint)
        )?;
    }
    Ok(())
}

fn write_run_metadata_json(metadata: &RunMetadata, out_dir: &Path) -> Result<()> {
    let payload = serde_json::to_vec_pretty(metadata)?;
    fs::write(out_dir.join("run_metadata.json"), payload)
        .with_context(|| format!("failed to write {}", out_dir.join("run_metadata.json").display()))
}

fn write_shared_keys_jsonl(rows: &[String], out_dir: &Path) -> Result<()> {
    let path = out_dir.join("shared_keys.jsonl");
    let file = File::create(&path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for row in rows {
        writer.write_all(row.as_bytes())?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

fn write_summary_markdown(
    out_dir: &Path,
    summary: &DiagnosticsSummaryJson,
    sweep_rows: &[ThresholdSweepRow],
    failure_rows: &[FailureTaxonomyRow],
    ranking_rows: &[ModelRankingSummaryRow],
) -> Result<()> {
    let mut lines = Vec::new();
    lines.push("# Decoder Diagnostics Summary".to_string());
    lines.push(String::new());
    lines.push("## Executive verdict".to_string());
    lines.push(summary.verdict.clone());
    lines.push(String::new());
    lines.push("## Sanity test results".to_string());
    lines.push(format!(
        "- Zero-error pass rate: {:.3}",
        summary.zero_error_pass_rate
    ));
    lines.push(format!(
        "- Single-bit geometry pass rate: {:.3}",
        summary.single_bit_geometry_pass_rate
    ));
    lines.push(format!(
        "- Syndrome update consistency pass rate: {:.3}",
        summary.syndrome_update_pass_rate
    ));
    lines.push(String::new());
    lines.push("## Threshold sweep conclusion".to_string());
    let mut non_saturated = sweep_rows
        .iter()
        .filter(|row| (0.001..=0.5).contains(&row.p_h_mean))
        .collect::<Vec<_>>();
    non_saturated.sort_by(|a, b| a.p_h_mean.total_cmp(&b.p_h_mean));
    if non_saturated.is_empty() {
        lines.push("- No non-saturated setting found in this diagnostic run.".to_string());
    } else {
        for row in non_saturated.iter().take(5) {
            lines.push(format!(
                "- r={}, d={}, t={}, schedule={}, p_h_mean={:.6}, success_rate={:.6}",
                row.r, row.d, row.t, row.threshold_schedule, row.p_h_mean, row.success_rate
            ));
        }
    }
    lines.push(String::new());
    lines.push("## Failure taxonomy".to_string());
    if failure_rows.is_empty() {
        lines.push("- No failure rows recorded.".to_string());
    } else {
        let mut counts = BTreeMap::new();
        for row in failure_rows {
            *counts.entry(row.outcome.clone()).or_insert(0usize) += 1;
        }
        for (outcome, count) in counts {
            lines.push(format!("- {}: {}", outcome, count));
        }
    }
    lines.push(String::new());
    lines.push("## M0/M1/M2/M3 comparison".to_string());
    if ranking_rows.is_empty() {
        lines.push("- Unified same-key model comparison was not generated because no non-saturated regime was selected.".to_string());
    } else {
        for row in ranking_rows {
            lines.push(format!(
                "- {} at (r={}, d={}, t={}): spearman={:.6}, pearson={:.6}, tail_l1_error={:.6}, top_10_recall={:.6}",
                row.model,
                row.r,
                row.d,
                row.t,
                row.spearman_with_p_h,
                row.pearson_with_p_h,
                row.tail_l1_error,
                row.top_10_percent_recall
            ));
        }
    }
    lines.push(String::new());
    lines.push("## Recommendation for the paper".to_string());
    lines.push(summary.recommendation.clone());
    fs::write(out_dir.join("diagnostics_summary.md"), lines.join("\n") + "\n")
        .with_context(|| format!("failed to write {}", out_dir.join("diagnostics_summary.md").display()))
}

fn write_summary_json(out_dir: &Path, summary: &DiagnosticsSummaryJson) -> Result<()> {
    let payload = serde_json::to_vec_pretty(summary)?;
    fs::write(out_dir.join("diagnostics_summary.json"), payload)
        .with_context(|| format!("failed to write {}", out_dir.join("diagnostics_summary.json").display()))
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
        if upper <= 1 {
            0
        } else {
            (self.next_u64() % upper as u64) as usize
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{build_sweep_settings, schedule_to_string, DiagnosticsCase};

    fn sample_case() -> DiagnosticsCase {
        DiagnosticsCase {
            name: "sample".to_string(),
            r: 13,
            d: 3,
            key_mode: "sampled_keys".to_string(),
            key_count: 4,
            geometry_key_count: 2,
            update_key_count: 2,
            distribution_key_count: 2,
            baseline_thresholds: vec![2, 2, 2],
            wrong_zero_is_failure: true,
            t_values: vec![0, 1, 2],
            distribution_t_values: vec![1, 2],
            sweep_max_iterations: vec![1, 3],
            sweep_constant_thresholds: vec![1, 2],
            sweep_schedules: vec![vec![2, 1, 1]],
            error_sample_count: 8,
            exact_error_weight_max: 1,
            divergence_flip_threshold: 8,
            divergence_residual_threshold: 8,
        }
    }

    #[test]
    fn schedule_string_is_stable() {
        assert_eq!(schedule_to_string(&[3, 2, 1]), "3;2;1");
    }

    #[test]
    fn sweep_settings_deduplicate() {
        let case = sample_case();
        let settings = build_sweep_settings(&case);
        let labels = settings.iter().map(|entry| schedule_to_string(entry)).collect::<Vec<_>>();
        assert!(labels.contains(&"1".to_string()));
        assert!(labels.contains(&"1;1;1".to_string()));
        assert!(labels.contains(&"2;1;1".to_string()));
        assert_eq!(labels.len(), labels.iter().collect::<std::collections::HashSet<_>>().len());
    }
}
