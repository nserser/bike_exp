use std::collections::BTreeMap;
use std::env;

use anyhow::{bail, Result};
use rayon::prelude::*;

use crate::bitset::Mask256;
use crate::combinatorics::{binom_u128, iter_combinations};
use crate::decoder::{decode_fast, decode_with_logs, Outcome};
use crate::io::{DecoderSection, StateSection};
use crate::key::{build_key_runtime, key_to_support_string, Key, KeyRuntime};
use crate::progress::ProgressReporter;
use crate::state::{
    aggregate_state_events, compute_state_descriptor, terminal_state, AggregationEvent, StateCountRecord,
};

#[derive(Clone, Debug, Default)]
pub struct EnumerationSummary {
    pub key_id: u64,
    pub total_errors: u128,
    pub failures: u128,
    pub success: u128,
    pub timeout: u128,
    pub wrongzero: u128,
    pub nonzerohalt: u128,
    pub p_h: f64,
    pub exact_errors: bool,
    pub decoder_iterations: usize,
    pub thresholds: Vec<usize>,
    pub key_support: String,
}

#[derive(Clone, Debug, Default)]
pub struct ExactRunSummary {
    pub rows: Vec<EnumerationSummary>,
    pub total_keys: u64,
    pub total_errors: u128,
}

#[derive(Clone, Debug, Default)]
pub struct ExactRunWithStates {
    pub summary: ExactRunSummary,
    pub state_records: Vec<StateCountRecord>,
}

fn intra_key_progress_enabled() -> bool {
    env::var("DFRCERT_INTRA_KEY_PROGRESS")
        .ok()
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            !normalized.is_empty() && normalized != "0" && normalized != "false" && normalized != "no"
        })
        .unwrap_or(false)
}

pub fn enumerate_key_exact(
    key_rt: &KeyRuntime,
    t: usize,
    decoder_config: &DecoderSection,
) -> Result<EnumerationSummary> {
    let bit_count = 2 * key_rt.key.r;
    if t > bit_count {
        bail!("error weight t={t} exceeds available bit positions {bit_count}");
    }

    let mut summary = EnumerationSummary {
        key_id: key_rt.key.key_id,
        total_errors: 0,
        failures: 0,
        success: 0,
        timeout: 0,
        wrongzero: 0,
        nonzerohalt: 0,
        p_h: 0.0,
        exact_errors: true,
        decoder_iterations: decoder_config.iterations,
        thresholds: decoder_config.thresholds.clone(),
        key_support: key_to_support_string(&key_rt.key),
    };

    for combo in iter_combinations(bit_count, t)? {
        let mut error = Mask256::empty();
        for bit in combo {
            error.set(bit)?;
        }

        summary.total_errors += 1;
        match decode_fast(key_rt, error, decoder_config)? {
            Outcome::Success => summary.success += 1,
            Outcome::Timeout => {
                summary.timeout += 1;
                summary.failures += 1;
            }
            Outcome::WrongZero => {
                summary.wrongzero += 1;
                summary.failures += 1;
            }
            Outcome::NonzeroHalt => {
                summary.nonzerohalt += 1;
                summary.failures += 1;
            }
        }
    }

    let expected_total = binom_u128(bit_count as u32, t as u32);
    if summary.total_errors != expected_total {
        bail!(
            "enumerated {} errors, expected {}",
            summary.total_errors,
            expected_total
        );
    }
    if summary.success + summary.timeout + summary.wrongzero + summary.nonzerohalt != summary.total_errors {
        bail!("outcome counts do not sum to total errors");
    }

    summary.p_h = if summary.total_errors == 0 {
        0.0
    } else {
        summary.failures as f64 / summary.total_errors as f64
    };
    Ok(summary)
}

pub fn enumerate_exact_keys(
    keys: &[Key],
    t: usize,
    decoder_config: &DecoderSection,
) -> Result<ExactRunSummary> {
    let progress = ProgressReporter::new("exact decode", keys.len());
    let mut rows = keys
        .par_iter()
        .enumerate()
        .map(|(index, key)| {
            let runtime = build_key_runtime(key.clone())?;
            let summary = enumerate_key_exact(&runtime, t, decoder_config)?;
            progress.tick();
            Ok((index, summary))
        })
        .collect::<Vec<Result<(usize, EnumerationSummary)>>>();

    rows.sort_by_key(|entry| match entry {
        Ok((index, _)) => *index,
        Err(_) => usize::MAX,
    });

    let mut run = ExactRunSummary {
        rows: Vec::with_capacity(keys.len()),
        total_keys: keys.len() as u64,
        total_errors: 0,
    };

    for entry in rows {
        let (_, summary) = entry?;
        run.total_errors += summary.total_errors;
        run.rows.push(summary);
    }
    progress.finish();

    Ok(run)
}

pub fn enumerate_exact_keys_with_states(
    run_id: &str,
    keys: &[Key],
    t: usize,
    decoder_config: &DecoderSection,
    state_config: &StateSection,
) -> Result<ExactRunWithStates> {
    let progress = ProgressReporter::new("exact decode + states", keys.len());
    let mut rows = keys
        .par_iter()
        .enumerate()
        .map(|(index, key)| {
            let runtime = build_key_runtime(key.clone())?;
            let (summary, state_records) =
                enumerate_key_exact_with_states(run_id, &runtime, t, decoder_config, state_config)?;
            progress.tick();
            Ok((index, summary, state_records))
        })
        .collect::<Vec<Result<(usize, EnumerationSummary, Vec<StateCountRecord>)>>>();

    rows.sort_by_key(|entry| match entry {
        Ok((index, _, _)) => *index,
        Err(_) => usize::MAX,
    });

    let mut run = ExactRunWithStates {
        summary: ExactRunSummary {
            rows: Vec::with_capacity(keys.len()),
            total_keys: keys.len() as u64,
            total_errors: 0,
        },
        state_records: Vec::new(),
    };

    for entry in rows {
        let (_, summary, state_records) = entry?;
        run.summary.total_errors += summary.total_errors;
        run.summary.rows.push(summary);
        run.state_records.extend(state_records);
    }
    progress.finish();

    Ok(run)
}

fn enumerate_key_exact_with_states(
    run_id: &str,
    key_rt: &KeyRuntime,
    t: usize,
    decoder_config: &DecoderSection,
    state_config: &StateSection,
) -> Result<(EnumerationSummary, Vec<StateCountRecord>)> {
    let bit_count = 2 * key_rt.key.r;
    if t > bit_count {
        bail!("error weight t={t} exceeds available bit positions {bit_count}");
    }
    let expected_total = binom_u128(bit_count as u32, t as u32);
    let progress = if intra_key_progress_enabled() {
        Some(ProgressReporter::new(
            format!("stage 3 key {} exact errors", key_rt.key.key_id),
            expected_total.try_into().unwrap_or(usize::MAX),
        ))
    } else {
        None
    };

    let mut summary = EnumerationSummary {
        key_id: key_rt.key.key_id,
        total_errors: 0,
        failures: 0,
        success: 0,
        timeout: 0,
        wrongzero: 0,
        nonzerohalt: 0,
        p_h: 0.0,
        exact_errors: true,
        decoder_iterations: decoder_config.iterations,
        thresholds: decoder_config.thresholds.clone(),
        key_support: key_to_support_string(&key_rt.key),
    };
    let model_names = state_config
        .models
        .iter()
        .filter(|model| *model != "M0")
        .cloned()
        .collect::<Vec<_>>();
    let mut events_by_model: BTreeMap<String, Vec<AggregationEvent>> = BTreeMap::new();

    for combo in iter_combinations(bit_count, t)? {
        let mut error = Mask256::empty();
        for bit in combo {
            error.set(bit)?;
        }

        summary.total_errors += 1;
        if let Some(progress) = &progress {
            progress.tick();
        }
        let result = decode_with_logs(key_rt, error, decoder_config)?;
        match result.outcome {
            Outcome::Success => summary.success += 1,
            Outcome::Timeout => {
                summary.timeout += 1;
                summary.failures += 1;
            }
            Outcome::WrongZero => {
                summary.wrongzero += 1;
                summary.failures += 1;
            }
            Outcome::NonzeroHalt => {
                summary.nonzerohalt += 1;
                summary.failures += 1;
            }
        }

        for model in &model_names {
            for (idx, log) in result.logs.iter().enumerate() {
                let threshold = decoder_config.thresholds[log.i];
                let descriptor = compute_state_descriptor(
                    model,
                    key_rt,
                    log.e_mask,
                    log.syndrome,
                    state_config,
                    threshold,
                );
                let successor = if let Some(next_log) = result.logs.get(idx + 1) {
                    compute_state_descriptor(
                        model,
                        key_rt,
                        next_log.e_mask,
                        next_log.syndrome,
                        state_config,
                        decoder_config.thresholds[next_log.i],
                    )
                } else {
                    match result.outcome {
                        Outcome::Success => terminal_state(model, "succ"),
                        Outcome::Timeout | Outcome::WrongZero | Outcome::NonzeroHalt => {
                            terminal_state(model, "bad")
                        }
                    }
                };
                events_by_model
                    .entry(model.clone())
                    .or_default()
                    .push(AggregationEvent {
                        iteration: log.i,
                        descriptor,
                        missed_count: log.missed_count,
                        false_count: log.false_count,
                        successor_state_id: successor.state_id,
                    });
            }
        }
    }

    if summary.total_errors != expected_total {
        bail!(
            "enumerated {} errors, expected {}",
            summary.total_errors,
            expected_total
        );
    }
    summary.p_h = if summary.total_errors == 0 {
        0.0
    } else {
        summary.failures as f64 / summary.total_errors as f64
    };
    if let Some(progress) = &progress {
        progress.finish();
    }

    let aggregation = aggregate_state_events(run_id, key_rt.key.key_id, &events_by_model);
    Ok((summary, aggregation.records))
}

#[cfg(test)]
mod tests {
    use crate::io::DecoderSection;
    use crate::key::{build_key_runtime, enumerate_all_keys, Key};

    use super::{enumerate_exact_keys, enumerate_exact_keys_with_states, enumerate_key_exact};

    #[test]
    fn exact_enumeration_counts_match_binomial_total() {
        let key = Key {
            key_id: 0,
            r: 5,
            d: 2,
            h0: vec![0, 1],
            h1: vec![0, 1],
        };
        let runtime = build_key_runtime(key).unwrap();
        let decoder = DecoderSection {
            iterations: 2,
            thresholds: vec![2, 2],
            wrong_zero_is_failure: true,
            flip_policy: "all_ge_threshold".to_string(),
        };
        let summary = enumerate_key_exact(&runtime, 2, &decoder).unwrap();
        assert_eq!(summary.total_errors, 45);
        assert_eq!(
            summary.success + summary.timeout + summary.wrongzero + summary.nonzerohalt,
            summary.total_errors
        );
    }

    #[test]
    fn exact_run_summary_accumulates_all_keys() {
        let keys = enumerate_all_keys(3, 1).unwrap();
        let decoder = DecoderSection {
            iterations: 2,
            thresholds: vec![1, 1],
            wrong_zero_is_failure: true,
            flip_policy: "all_ge_threshold".to_string(),
        };
        let run = enumerate_exact_keys(&keys, 1, &decoder).unwrap();
        assert_eq!(run.total_keys, 9);
        assert_eq!(run.rows.len(), 9);
        assert_eq!(run.total_errors, 54);
    }

    #[test]
    fn exact_run_with_states_collects_records() {
        let keys = enumerate_all_keys(3, 1).unwrap();
        let decoder = DecoderSection {
            iterations: 2,
            thresholds: vec![1, 1],
            wrong_zero_is_failure: true,
            flip_policy: "all_ge_threshold".to_string(),
        };
        let state = crate::io::StateSection {
            models: vec!["M1".to_string(), "M2".to_string(), "M3".to_string()],
            u_bins: vec![0, 1, 99],
            omega_bins: vec![0, 1, 2, 99],
            strict_absorbing: false,
        };
        let run = enumerate_exact_keys_with_states("run", &keys, 1, &decoder, &state).unwrap();
        assert_eq!(run.summary.total_keys, 9);
        assert!(!run.state_records.is_empty());
    }
}
