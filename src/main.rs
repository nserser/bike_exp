mod binning;
mod bitset;
mod certify;
mod combinatorics;
mod decoder;
mod diagnostics;
mod enumerate;
mod io;
mod key;
mod metrics;
mod m3;
mod profile;
mod progress;
mod rare_event;
mod state;

use std::path::PathBuf;
use std::fs::File;
use std::io::{BufWriter, Write};

use anyhow::{bail, Result};
use binning::build_all_bins;
use certify::certify_direct_bins;
use clap::{Parser, Subcommand};
use diagnostics::run_diagnostics_command;
use m3::{run_m3_experiment_command, run_m3_threshold_sweep_command};
use enumerate::enumerate_exact_keys_with_states;
use io::{
    copy_resolved_config, default_image_tag, detect_git_commit, ensure_artifact_layout,
    file_sha256, load_config, manifest_path, now_utc, write_manifest, write_string, Manifest,
};
use key::{enumerate_all_keys, sample_keys_uniform, Key};
use metrics::compute_direct_metrics;
use profile::compute_profiles;
use progress::ProgressReporter;
use rare_event::run_rare_event_diagnostics;

#[derive(Debug, Parser)]
#[command(name = "dfrcert")]
#[command(about = "Dependency-aware DFR certification scaffold", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Run {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    Profiles {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    Decode {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        profiles: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    Certify {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        inputs: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    RareEvent {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        inputs: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    Calibrate {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    Diagnostics {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    M3Experiment {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    M3ThresholdSweep {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run { config, out } => run_command("run", &config, &out),
        Commands::Profiles { config, out } => profiles_command("profiles", &config, &out),
        Commands::Decode {
            config,
            profiles: _profiles,
            out,
        } => run_command("decode", &config, &out),
        Commands::Certify {
            config,
            inputs: _inputs,
            out,
        } => prepare_run("certify", &config, &out),
        Commands::RareEvent {
            config,
            inputs,
            out,
        } => rare_event_command("rare-event", &config, &inputs, &out),
        Commands::Calibrate { config, out } => calibrate_command("calibrate", &config, &out),
        Commands::Diagnostics { config, out } => run_diagnostics_command(&config, &out),
        Commands::M3Experiment { config, out } => run_m3_experiment_command(&config, &out),
        Commands::M3ThresholdSweep { config, out } => run_m3_threshold_sweep_command(&config, &out),
    }
}

fn prepare_run(command_name: &str, config_path: &PathBuf, out_dir: &PathBuf) -> Result<()> {
    let config = load_config(config_path)?;
    ensure_artifact_layout(out_dir)?;
    copy_resolved_config(config_path, out_dir)?;

    let manifest = Manifest {
        run_id: config.run.run_id.clone(),
        binary_version: env!("CARGO_PKG_VERSION").to_string(),
        docker_image_tag: default_image_tag(),
        command: format!("dfrcert {command_name} --config {}", config_path.display()),
        config_hash: file_sha256(config_path)?,
        git_commit: detect_git_commit(),
        start_timestamp_utc: now_utc(),
        end_timestamp_utc: None,
        seed: config.run.seed,
        keys_processed: 0,
        errors_processed: 0,
    };

    write_manifest(&manifest_path(out_dir), &manifest)?;

    println!(
        "prepared scaffold run '{}' in {}",
        config.run.run_id,
        out_dir.display()
    );
    Ok(())
}

fn run_command(command_name: &str, config_path: &PathBuf, out_dir: &PathBuf) -> Result<()> {
    let config = load_config(config_path)?;
    ensure_artifact_layout(out_dir)?;
    copy_resolved_config(config_path, out_dir)?;

    eprintln!("[progress] stage 1/6 selecting keys");
    let keys = select_keys(&config)?;
    if keys.is_empty() {
        bail!("no keys enumerated");
    }
    eprintln!("[progress] stage 1/6 complete: {} keys selected", keys.len());
    eprintln!("[progress] stage 2/6 computing profiles");
    let profiles = compute_profiles(&keys, config.parameters.t, &config.profile)?;
    eprintln!("[progress] stage 2/6 complete");
    eprintln!("[progress] stage 3/6 exact decoding and state aggregation");
    let run = enumerate_exact_keys_with_states(
        &config.run.run_id,
        &keys,
        config.parameters.t,
        &config.decoder,
        &config.state,
    )?;
    eprintln!("[progress] stage 3/6 complete");
    eprintln!("[progress] stage 4/6 building bins");
    let bins = build_all_bins(&config.run.run_id, &profiles, &config.binning, &config.state.models)?;
    eprintln!("[progress] stage 4/6 complete");
    eprintln!("[progress] stage 5/6 certifying bounds");
    let direct = certify_direct_bins(
        &config.run.run_id,
        &bins,
        &run.summary.rows,
        &run.state_records,
    )?;
    eprintln!("[progress] stage 5/6 complete");
    eprintln!("[progress] stage 6/6 computing metrics and writing artifacts");
    let (metrics, top_tail) = compute_direct_metrics(
        &config.run.run_id,
        &bins,
        &direct.bin_bounds,
        &direct.fs_bounds,
        &direct.tail_curves,
        &run.summary.rows,
        &config.metrics.top_tail_fractions,
    );

    write_keys_csv(&config.run.run_id, &config.run.mode, config.run.seed, &keys, out_dir)?;
    write_profiles_csv(&config.run.run_id, &profiles, out_dir)?;
    write_bins_csv(&bins, out_dir)?;

    let mut ph_exact_csv = String::from(
        "run_id,key_id,total_errors,failures,success,timeout,wrongzero,nonzerohalt,p_h,exact_errors,decoder_iterations,thresholds\n",
    );
    for summary in &run.summary.rows {
        ph_exact_csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{:.17},{},{},\"{}\"\n",
            config.run.run_id,
            summary.key_id,
            summary.total_errors,
            summary.failures,
            summary.success,
            summary.timeout,
            summary.wrongzero,
            summary.nonzerohalt,
            summary.p_h,
            summary.exact_errors,
            summary.decoder_iterations,
            summary.thresholds.iter().map(usize::to_string).collect::<Vec<_>>().join(";"),
        ));
    }
    write_string(&out_dir.join("csv").join("ph_exact.csv"), &ph_exact_csv)?;
    write_state_counts_jsonl(&run.state_records, out_dir)?;
    write_bin_bounds_csv(&direct.bin_bounds, out_dir)?;
    write_fs_bounds_csv(&direct.fs_bounds, out_dir)?;
    write_tail_curves_csv(&direct.tail_curves, out_dir)?;
    write_metrics_csv(&metrics, out_dir)?;
    write_top_tail_csv(&top_tail, out_dir)?;
    write_key_scores_csv(
        &config.run.run_id,
        &bins,
        &direct.bin_bounds,
        &direct.fs_bounds,
        &run.summary.rows,
        out_dir,
    )?;
    write_ablation_csv(
        &config.run.run_id,
        &profiles,
        &config.binning,
        &run.summary.rows,
        out_dir,
    )?;
    write_transition_envelopes_jsonl(&direct.transition_envelopes, out_dir)?;
    eprintln!("[progress] stage 6/6 complete");

    let manifest = Manifest {
        run_id: config.run.run_id.clone(),
        binary_version: env!("CARGO_PKG_VERSION").to_string(),
        docker_image_tag: default_image_tag(),
        command: format!("dfrcert {command_name} --config {}", config_path.display()),
        config_hash: file_sha256(config_path)?,
        git_commit: detect_git_commit(),
        start_timestamp_utc: now_utc(),
        end_timestamp_utc: None,
        seed: config.run.seed,
        keys_processed: run.summary.total_keys,
        errors_processed: run.summary.total_errors as u64,
    };
    write_manifest(&manifest_path(out_dir), &manifest)?;

    println!(
        "decoded {} keys for run '{}' into {}",
        run.summary.total_keys,
        config.run.run_id,
        out_dir.display()
    );
    Ok(())
}

fn profiles_command(command_name: &str, config_path: &PathBuf, out_dir: &PathBuf) -> Result<()> {
    let config = load_config(config_path)?;
    ensure_artifact_layout(out_dir)?;
    copy_resolved_config(config_path, out_dir)?;

    eprintln!("[progress] stage 1/3 selecting keys");
    let keys = select_keys(&config)?;
    eprintln!("[progress] stage 1/3 complete: {} keys selected", keys.len());
    eprintln!("[progress] stage 2/3 computing profiles");
    let profiles = compute_profiles(&keys, config.parameters.t, &config.profile)?;
    eprintln!("[progress] stage 2/3 complete");
    write_keys_csv(&config.run.run_id, &config.run.mode, config.run.seed, &keys, out_dir)?;
    write_profiles_csv(&config.run.run_id, &profiles, out_dir)?;
    eprintln!("[progress] stage 3/3 building bins");
    let bins = build_all_bins(&config.run.run_id, &profiles, &config.binning, &config.state.models)?;
    write_bins_csv(&bins, out_dir)?;
    eprintln!("[progress] stage 3/3 complete");

    let manifest = Manifest {
        run_id: config.run.run_id.clone(),
        binary_version: env!("CARGO_PKG_VERSION").to_string(),
        docker_image_tag: default_image_tag(),
        command: format!("dfrcert {command_name} --config {}", config_path.display()),
        config_hash: file_sha256(config_path)?,
        git_commit: detect_git_commit(),
        start_timestamp_utc: now_utc(),
        end_timestamp_utc: None,
        seed: config.run.seed,
        keys_processed: keys.len() as u64,
        errors_processed: 0,
    };
    write_manifest(&manifest_path(out_dir), &manifest)?;
    println!(
        "computed {} profiles for run '{}' into {}",
        keys.len(),
        config.run.run_id,
        out_dir.display()
    );
    Ok(())
}

fn select_keys(config: &io::RunConfig) -> Result<Vec<Key>> {
    match config.run.mode.as_str() {
        "exact_full_keyspace" => enumerate_all_keys(config.parameters.r, config.parameters.d),
        "sampled_keys" => Ok(sample_keys_uniform(
            config.parameters.r,
            config.parameters.d,
            config.sampling.key_count.max(1),
            config.run.seed,
        )),
        "stratified_keys" => {
            let candidate_count = (config.sampling.key_count.max(1) * 8).max(64);
            let candidates = sample_keys_uniform(
                config.parameters.r,
                config.parameters.d,
                candidate_count,
                config.run.seed,
            );
            select_stratified_keys(config, &candidates)
        }
        other => bail!("unsupported mode '{other}'"),
    }
}

fn select_stratified_keys(config: &io::RunConfig, candidates: &[Key]) -> Result<Vec<Key>> {
    let profiles = compute_profiles(candidates, config.parameters.t, &config.profile)?;
    let target = config.sampling.key_count.max(1).min(candidates.len());
    let mut ranked_c4 = profiles.clone();
    ranked_c4.sort_by(|a, b| b.c4_loc.cmp(&a.c4_loc).then_with(|| a.key_id.cmp(&b.key_id)));
    let mut ranked_dmax = profiles.clone();
    ranked_dmax.sort_by(|a, b| b.d_max.cmp(&a.d_max).then_with(|| a.key_id.cmp(&b.key_id)));
    let mut ranked_fixed = profiles.clone();
    ranked_fixed.sort_by(|a, b| b.b_fixed.cmp(&a.b_fixed).then_with(|| a.key_id.cmp(&b.key_id)));
    let mut ranked_near = profiles.clone();
    ranked_near.sort_by(|a, b| {
        max_profile_json(&b.an_u_json)
            .total_cmp(&max_profile_json(&a.an_u_json))
            .then_with(|| a.key_id.cmp(&b.key_id))
    });

    let key_map = candidates
        .iter()
        .cloned()
        .map(|key| (key.key_id, key))
        .collect::<std::collections::BTreeMap<_, _>>();

    let mut selected = std::collections::BTreeSet::new();
    let sources = [&ranked_c4, &ranked_dmax, &ranked_fixed, &ranked_near];
    let mut cursor = 0usize;
    while selected.len() < target {
        let mut progressed = false;
        for source in &sources {
            if let Some(profile) = source.get(cursor) {
                selected.insert(profile.key_id);
                progressed = true;
                if selected.len() >= target {
                    break;
                }
            }
        }
        if !progressed {
            break;
        }
        cursor += 1;
    }

    if selected.len() < target {
        for key in candidates {
            selected.insert(key.key_id);
            if selected.len() >= target {
                break;
            }
        }
    }

    Ok(selected
        .into_iter()
        .filter_map(|key_id| key_map.get(&key_id).cloned())
        .collect())
}

fn max_profile_json(payload: &str) -> f64 {
    serde_json::from_str::<serde_json::Value>(payload)
        .ok()
        .and_then(|value| value.as_object().cloned())
        .map(|obj| {
            obj.values()
                .filter_map(|v| v.as_f64())
                .fold(0.0_f64, f64::max)
        })
        .unwrap_or(0.0)
}

fn rare_event_command(
    command_name: &str,
    config_path: &PathBuf,
    inputs_dir: &PathBuf,
    out_dir: &PathBuf,
) -> Result<()> {
    let config = load_config(config_path)?;
    ensure_artifact_layout(out_dir)?;
    copy_resolved_config(config_path, out_dir)?;
    eprintln!("[progress] stage 1/2 loading keys");
    let keys_path = inputs_dir.join("csv").join("keys.csv");
    let keys = if keys_path.exists() {
        load_keys_from_csv(&keys_path)?
    } else {
        select_keys(&config)?
    };
    eprintln!("[progress] stage 1/2 complete: {} keys loaded", keys.len());
    eprintln!("[progress] stage 2/2 running rare-event diagnostics");
    let records = run_rare_event_diagnostics(&config.run.run_id, &keys, &config)?;
    write_rare_event_jsonl(&records, out_dir)?;
    eprintln!("[progress] stage 2/2 complete");

    let manifest = Manifest {
        run_id: config.run.run_id.clone(),
        binary_version: env!("CARGO_PKG_VERSION").to_string(),
        docker_image_tag: default_image_tag(),
        command: format!(
            "dfrcert {command_name} --config {} --inputs {}",
            config_path.display(),
            inputs_dir.display()
        ),
        config_hash: file_sha256(config_path)?,
        git_commit: detect_git_commit(),
        start_timestamp_utc: now_utc(),
        end_timestamp_utc: None,
        seed: config.run.seed,
        keys_processed: keys.len() as u64,
        errors_processed: 0,
    };
    write_manifest(&manifest_path(out_dir), &manifest)?;
    println!(
        "computed {} rare-event records for run '{}' into {}",
        records.len(),
        config.run.run_id,
        out_dir.display()
    );
    Ok(())
}

fn calibrate_command(command_name: &str, config_path: &PathBuf, out_dir: &PathBuf) -> Result<()> {
    let config = load_config(config_path)?;
    ensure_artifact_layout(out_dir)?;
    copy_resolved_config(config_path, out_dir)?;

    eprintln!("[progress] stage 1/3 selecting keys");
    let keys = select_keys(&config)?;
    if keys.is_empty() {
        bail!("no keys available for calibration");
    }
    eprintln!("[progress] stage 1/3 complete: {} keys selected", keys.len());

    let mut candidates = std::collections::BTreeSet::new();
    for &threshold in &config.decoder.thresholds {
        candidates.insert(threshold);
    }
    for &threshold in &config.profile.fixedpoint_thresholds {
        candidates.insert(threshold);
    }
    if candidates.is_empty() {
        bail!("no threshold candidates available for calibration");
    }
    eprintln!("[progress] stage 2/3 evaluating {} threshold candidates", candidates.len());
    let threshold_progress = ProgressReporter::new("calibrate thresholds", candidates.len());

    let mut csv = String::from(
        "run_id,threshold,iterations,key_count,total_errors,total_failures,mean_p_h,max_p_h,success_rate\n",
    );
    let mut best: Option<(usize, f64, f64)> = None;
    for threshold in candidates {
        let mut trial = config.clone();
        trial.decoder.thresholds = vec![threshold; config.decoder.iterations];
        let run = enumerate_exact_keys_with_states(
            &config.run.run_id,
            &keys,
            config.parameters.t,
            &trial.decoder,
            &trial.state,
        )?;
        let total_errors = run.summary.total_errors;
        let total_failures = run.summary.rows.iter().map(|row| row.failures).sum::<u128>();
        let mean_p_h = if run.summary.rows.is_empty() {
            0.0
        } else {
            run.summary.rows.iter().map(|row| row.p_h).sum::<f64>() / run.summary.rows.len() as f64
        };
        let max_p_h = run.summary.rows.iter().map(|row| row.p_h).fold(0.0_f64, f64::max);
        let success_rate = if total_errors == 0 {
            1.0
        } else {
            1.0 - (total_failures as f64 / total_errors as f64)
        };
        csv.push_str(&format!(
            "{},{},{},{},{},{},{:.17},{:.17},{:.17}\n",
            config.run.run_id,
            threshold,
            config.decoder.iterations,
            keys.len(),
            total_errors,
            total_failures,
            mean_p_h,
            max_p_h,
            success_rate,
        ));

        match best {
            None => best = Some((threshold, mean_p_h, max_p_h)),
            Some((_, best_mean, best_max)) => {
                if mean_p_h < best_mean || (mean_p_h == best_mean && max_p_h < best_max) {
                    best = Some((threshold, mean_p_h, max_p_h));
                }
            }
        }
        threshold_progress.tick();
    }
    threshold_progress.finish();
    eprintln!("[progress] stage 2/3 complete");
    write_string(&out_dir.join("csv").join("threshold_scan.csv"), &csv)?;

    if let Some((best_threshold, _, _)) = best {
        let payload = serde_json::json!({
            "run_id": config.run.run_id,
            "best_threshold": best_threshold,
            "recommended_schedule": vec![best_threshold; config.decoder.iterations],
        });
        write_string(
            &out_dir.join("json").join("threshold_recommendation.json"),
            &serde_json::to_string_pretty(&payload)?,
        )?;
    }
    eprintln!("[progress] stage 3/3 writing artifacts");

    let manifest = Manifest {
        run_id: config.run.run_id.clone(),
        binary_version: env!("CARGO_PKG_VERSION").to_string(),
        docker_image_tag: default_image_tag(),
        command: format!("dfrcert {command_name} --config {}", config_path.display()),
        config_hash: file_sha256(config_path)?,
        git_commit: detect_git_commit(),
        start_timestamp_utc: now_utc(),
        end_timestamp_utc: None,
        seed: config.run.seed,
        keys_processed: keys.len() as u64,
        errors_processed: 0,
    };
    write_manifest(&manifest_path(out_dir), &manifest)?;
    eprintln!("[progress] stage 3/3 complete");
    println!(
        "calibrated {} thresholds for run '{}' into {}",
        csv.lines().count().saturating_sub(1),
        config.run.run_id,
        out_dir.display()
    );
    Ok(())
}

fn write_profiles_csv(
    run_id: &str,
    profiles: &[profile::ProfileRecord],
    out_dir: &PathBuf,
) -> Result<()> {
    let mut csv = String::from(
        "run_id,key_id,r,d,t,c4_loc,d_max,lambda_json,an_u_json,gather_json,b_fixed,omega_pair_max,profile_hash\n",
    );
    for profile in profiles {
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},\"{}\",\"{}\",\"{}\",{},{},{}\n",
            run_id,
            profile.key_id,
            profile.r,
            profile.d,
            profile.t,
            profile.c4_loc,
            profile.d_max,
            profile.lambda_json.replace('"', "\"\""),
            profile.an_u_json.replace('"', "\"\""),
            profile.gather_json.replace('"', "\"\""),
            profile.b_fixed,
            profile.omega_pair_max,
            profile.profile_hash,
        ));
    }
    write_string(&out_dir.join("csv").join("profiles.csv"), &csv)
}

fn write_keys_csv(
    run_id: &str,
    key_source: &str,
    seed: u64,
    keys: &[Key],
    out_dir: &PathBuf,
) -> Result<()> {
    let mut csv = String::from("run_id,key_id,r,d,h0,h1,key_source,seed,index\n");
    for (index, key) in keys.iter().enumerate() {
        csv.push_str(&format!(
            "{},{},{},{},\"{}\",\"{}\",{},{},{}\n",
            run_id,
            key.key_id,
            key.r,
            key.d,
            key.h0.iter().map(u16::to_string).collect::<Vec<_>>().join(";"),
            key.h1.iter().map(u16::to_string).collect::<Vec<_>>().join(";"),
            key_source,
            seed,
            index
        ));
    }
    write_string(&out_dir.join("csv").join("keys.csv"), &csv)
}

fn write_bins_csv(bins: &[binning::BinAssignment], out_dir: &PathBuf) -> Result<()> {
    let mut csv = String::from("run_id,model,bin_budget,bin_id,key_count,bin_descriptor_json\n");
    for bin in bins {
        csv.push_str(&format!(
            "{},{},{},{},{},\"{}\"\n",
            bin.run_id,
            bin.model,
            bin.bin_budget,
            bin.bin_id,
            bin.key_ids.len(),
            bin.descriptor_json.replace('"', "\"\""),
        ));
    }
    write_string(&out_dir.join("csv").join("bins.csv"), &csv)
}

fn write_bin_bounds_csv(bounds: &[certify::BinBoundRecord], out_dir: &PathBuf) -> Result<()> {
    let mut csv = String::from(
        "run_id,model,bin_budget,bin_id,key_count,nu_phi,p_phi_max,p_phi_mean,ph_min,ph_median,ph_p95,ph_max\n",
    );
    for bound in bounds {
        csv.push_str(&format!(
            "{},{},{},{},{},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17}\n",
            bound.run_id,
            bound.model,
            bound.bin_budget,
            bound.bin_id,
            bound.key_count,
            bound.nu_phi,
            bound.p_phi_max,
            bound.p_phi_mean,
            bound.ph_min,
            bound.ph_median,
            bound.ph_p95,
            bound.ph_max,
        ));
    }
    write_string(&out_dir.join("csv").join("bin_bounds.csv"), &csv)
}

fn write_fs_bounds_csv(bounds: &[certify::FsBoundRecord], out_dir: &PathBuf) -> Result<()> {
    let mut csv = String::from(
        "run_id,model,bin_budget,bin_id,key_count,nu_phi,p_phi_fs_plus,alpha_total_final,bad_mass_final,nonterminal_mass_final,row_mass_max,vacuous_flag\n",
    );
    for bound in bounds {
        csv.push_str(&format!(
            "{},{},{},{},{},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{}\n",
            bound.run_id,
            bound.model,
            bound.bin_budget,
            bound.bin_id,
            bound.key_count,
            bound.nu_phi,
            bound.p_phi_fs_plus,
            bound.alpha_total_final,
            bound.bad_mass_final,
            bound.nonterminal_mass_final,
            bound.row_mass_max,
            bound.vacuous_flag,
        ));
    }
    write_string(&out_dir.join("csv").join("fs_bounds.csv"), &csv)
}

fn write_tail_curves_csv(points: &[certify::TailCurvePoint], out_dir: &PathBuf) -> Result<()> {
    let mut csv = String::from("run_id,model,bin_budget,bound_type,tau,tail_value\n");
    for point in points {
        csv.push_str(&format!(
            "{},{},{},{},{:.17},{:.17}\n",
            point.run_id,
            point.model,
            point.bin_budget,
            point.bound_type,
            point.tau,
            point.tail_value,
        ));
    }
    write_string(&out_dir.join("csv").join("tail_curves.csv"), &csv)
}

fn write_metrics_csv(records: &[metrics::MetricRecord], out_dir: &PathBuf) -> Result<()> {
    let mut csv = String::from(
        "run_id,model,bin_budget,bound_type,coverage,median_looseness,p95_looseness,max_looseness,vacuous_bin_fraction,spearman,tail_l1_error,delta_bound,delta_exact\n",
    );
    for record in records {
        csv.push_str(&format!(
            "{},{},{},{},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17}\n",
            record.run_id,
            record.model,
            record.bin_budget,
            record.bound_type,
            record.coverage,
            record.median_looseness,
            record.p95_looseness,
            record.max_looseness,
            record.vacuous_bin_fraction,
            record.spearman,
            record.tail_l1_error,
            record.delta_bound,
            record.delta_exact,
        ));
    }
    write_string(&out_dir.join("csv").join("metrics.csv"), &csv)
}

fn write_top_tail_csv(records: &[metrics::TopTailRecord], out_dir: &PathBuf) -> Result<()> {
    let mut csv =
        String::from("run_id,model,bin_budget,bound_type,tail_fraction,precision,recall\n");
    for record in records {
        csv.push_str(&format!(
            "{},{},{},{},{:.17},{:.17},{:.17}\n",
            record.run_id,
            record.model,
            record.bin_budget,
            record.bound_type,
            record.tail_fraction,
            record.precision,
            record.recall,
        ));
    }
    write_string(&out_dir.join("csv").join("top_tail_recall.csv"), &csv)
}

fn write_key_scores_csv(
    run_id: &str,
    bins: &[binning::BinAssignment],
    bin_bounds: &[certify::BinBoundRecord],
    fs_bounds: &[certify::FsBoundRecord],
    exact_rows: &[enumerate::EnumerationSummary],
    out_dir: &PathBuf,
) -> Result<()> {
    let exact_map = exact_rows
        .iter()
        .map(|row| (row.key_id, row.p_h))
        .collect::<std::collections::BTreeMap<_, _>>();
    let direct_lookup = bin_bounds
        .iter()
        .map(|bound| ((bound.model.clone(), bound.bin_budget, bound.bin_id), bound.p_phi_max))
        .collect::<std::collections::BTreeMap<_, _>>();
    let fs_lookup = fs_bounds
        .iter()
        .map(|bound| ((bound.model.clone(), bound.bin_budget, bound.bin_id), bound.p_phi_fs_plus))
        .collect::<std::collections::BTreeMap<_, _>>();

    let mut csv = String::from("run_id,model,bin_budget,bound_type,key_id,score,exact_p_h\n");
    for bin in bins {
        let direct_score = direct_lookup
            .get(&(bin.model.clone(), bin.bin_budget, bin.bin_id))
            .copied()
            .unwrap_or(0.0);
        let fs_score = fs_lookup
            .get(&(bin.model.clone(), bin.bin_budget, bin.bin_id))
            .copied()
            .unwrap_or(direct_score);
        for key_id in &bin.key_ids {
            let exact_p_h = exact_map.get(key_id).copied().unwrap_or(0.0);
            csv.push_str(&format!(
                "{},{},{},direct_bin,{},{:.17},{:.17}\n",
                run_id, bin.model, bin.bin_budget, key_id, direct_score, exact_p_h
            ));
            csv.push_str(&format!(
                "{},{},{},finite_state,{},{:.17},{:.17}\n",
                run_id, bin.model, bin.bin_budget, key_id, fs_score, exact_p_h
            ));
        }
    }
    write_string(&out_dir.join("csv").join("key_scores.csv"), &csv)
}

fn write_state_counts_jsonl(records: &[state::StateCountRecord], out_dir: &PathBuf) -> Result<()> {
    let path = out_dir.join("json").join("state_counts.jsonl");
    let file = File::create(&path)?;
    let mut writer = BufWriter::new(file);
    for record in records {
        serde_json::to_writer(&mut writer, record)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

fn write_transition_envelopes_jsonl(
    records: &[certify::TransitionEnvelopeRecord],
    out_dir: &PathBuf,
) -> Result<()> {
    let path = out_dir.join("json").join("transition_envelopes.jsonl");
    let file = File::create(&path)?;
    let mut writer = BufWriter::new(file);
    for record in records {
        let value = serde_json::json!({
            "run_id": record.run_id,
            "model": record.model,
            "bin_budget": record.bin_budget,
            "bin_id": record.bin_id,
            "iteration": record.iteration,
            "state_id": record.state_id,
            "state_tuple": record.state_tuple,
            "denom_key_count": record.denom_key_count,
            "pairs": record.pairs.iter().map(|pair| serde_json::json!({
                "m": pair.m,
                "l": pair.l,
                "q_plus": pair.q_plus,
                "observed_successor_state_ids": pair.observed_successor_state_ids,
            })).collect::<Vec<_>>(),
            "row_sum_q_plus": record.row_sum_q_plus,
        });
        serde_json::to_writer(&mut writer, &value)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

fn write_rare_event_jsonl(records: &[rare_event::RareEventRecord], out_dir: &PathBuf) -> Result<()> {
    let path = out_dir.join("json").join("rare_event.jsonl");
    let file = File::create(&path)?;
    let mut writer = BufWriter::new(file);
    for record in records {
        let value = serde_json::json!({
            "run_id": record.run_id,
            "key_id": record.key_id,
            "u": record.u,
            "tail_mass_A_union": record.tail_mass_a_union,
            "conditional_total": record.conditional_total,
            "conditional_failures": record.conditional_failures,
            "conditional_failure_rate": record.conditional_failure_rate,
            "contribution_proxy": record.contribution_proxy,
            "exact_conditional": record.exact_conditional,
        });
        serde_json::to_writer(&mut writer, &value)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

fn load_keys_from_csv(path: &PathBuf) -> Result<Vec<Key>> {
    let text = std::fs::read_to_string(path)?;
    let mut keys = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        if idx == 0 || line.trim().is_empty() {
            continue;
        }
        let parts = parse_csv_line(line);
        if parts.len() < 6 {
            continue;
        }
        keys.push(Key {
            key_id: parts[1].parse()?,
            r: parts[2].parse()?,
            d: parts[3].parse()?,
            h0: parse_support_field(&parts[4]),
            h1: parse_support_field(&parts[5]),
        });
    }
    Ok(keys)
}

fn parse_csv_line(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    for ch in line.chars() {
        match ch {
            '"' => in_quotes = !in_quotes,
            ',' if !in_quotes => {
                out.push(current.clone());
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    out.push(current);
    out
}

fn parse_support_field(field: &str) -> Vec<u16> {
    field
        .split(';')
        .filter(|part| !part.is_empty())
        .filter_map(|part| part.parse::<u16>().ok())
        .collect()
}

fn write_ablation_csv(
    run_id: &str,
    profiles: &[profile::ProfileRecord],
    binning: &io::BinningSection,
    exact_rows: &[enumerate::EnumerationSummary],
    out_dir: &PathBuf,
) -> Result<()> {
    let variants = vec![
        ("M3_base".to_string(), "c4_loc+d_max".to_string()),
        ("M3_near".to_string(), "c4_loc+d_max+A_N".to_string()),
        ("M3_gather".to_string(), "c4_loc+d_max+A_N+G_H".to_string()),
        ("M3_fixed".to_string(), "c4_loc+d_max+A_N+B_H".to_string()),
        ("M3_full".to_string(), "c4_loc+d_max+A_N+G_H+B_H+omega".to_string()),
    ];
    let model_names = variants.iter().map(|(name, _)| name.clone()).collect::<Vec<_>>();
    let bins = build_all_bins(run_id, profiles, binning, &model_names)?;
    let direct = certify_direct_bins(run_id, &bins, exact_rows, &[])?;
    let (metrics, top_tail) = compute_direct_metrics(
        run_id,
        &bins,
        &direct.bin_bounds,
        &direct.fs_bounds,
        &direct.tail_curves,
        exact_rows,
        &[0.05],
    );
    let recall_lookup = top_tail
        .iter()
        .filter(|row| row.bound_type == "direct_bin")
        .map(|row| ((row.model.clone(), row.bin_budget), row.recall))
        .collect::<std::collections::BTreeMap<_, _>>();

    let mut csv = String::from(
        "run_id,model_variant,bin_budget,coordinates,coverage,median_looseness,p95_looseness,spearman,tail_l1_error,recall_top_5pct\n",
    );
    for record in metrics.iter().filter(|row| row.bound_type == "direct_bin") {
        let coords = variants
            .iter()
            .find(|(name, _)| *name == record.model)
            .map(|(_, coords)| coords.as_str())
            .unwrap_or("");
        let recall = recall_lookup
            .get(&(record.model.clone(), record.bin_budget))
            .copied()
            .unwrap_or(0.0);
        csv.push_str(&format!(
            "{},{},{},{},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17}\n",
            run_id,
            record.model,
            record.bin_budget,
            coords,
            record.coverage,
            record.median_looseness,
            record.p95_looseness,
            record.spearman,
            record.tail_l1_error,
            recall,
        ));
    }
    write_string(&out_dir.join("csv").join("ablation.csv"), &csv)
}
