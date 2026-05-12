use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::thread;

use anyhow::{bail, Result};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use crate::binning::BinAssignment;
use crate::enumerate::EnumerationSummary;
use crate::progress::ProgressReporter;
use crate::state::{terminal_state, StateCountRecord};

#[derive(Clone, Debug)]
pub struct BinBoundRecord {
    pub run_id: String,
    pub model: String,
    pub bin_budget: usize,
    pub bin_id: usize,
    pub key_count: usize,
    pub nu_phi: f64,
    pub p_phi_max: f64,
    pub p_phi_mean: f64,
    pub ph_min: f64,
    pub ph_median: f64,
    pub ph_p95: f64,
    pub ph_max: f64,
}

#[derive(Clone, Debug)]
pub struct TailCurvePoint {
    pub run_id: String,
    pub model: String,
    pub bin_budget: usize,
    pub bound_type: String,
    pub tau: f64,
    pub tail_value: f64,
}

#[derive(Clone, Debug, Default)]
pub struct CertificationSummary {
    pub bin_bounds: Vec<BinBoundRecord>,
    pub tail_curves: Vec<TailCurvePoint>,
    pub fs_bounds: Vec<FsBoundRecord>,
    pub transition_envelopes: Vec<TransitionEnvelopeRecord>,
}

#[derive(Clone, Debug)]
pub struct FsBoundRecord {
    pub run_id: String,
    pub model: String,
    pub bin_budget: usize,
    pub bin_id: usize,
    pub key_count: usize,
    pub nu_phi: f64,
    pub p_phi_fs_plus: f64,
    pub alpha_total_final: f64,
    pub bad_mass_final: f64,
    pub nonterminal_mass_final: f64,
    pub row_mass_max: f64,
    pub vacuous_flag: bool,
}

#[derive(Clone, Debug)]
pub struct TransitionPairRecord {
    pub m: u16,
    pub l: u16,
    pub q_plus: f64,
    pub observed_successor_state_ids: Vec<u64>,
}

#[derive(Clone, Debug)]
pub struct TransitionEnvelopeRecord {
    pub run_id: String,
    pub model: String,
    pub bin_budget: usize,
    pub bin_id: usize,
    pub iteration: usize,
    pub state_id: u64,
    pub state_tuple: serde_json::Value,
    pub denom_key_count: usize,
    pub pairs: Vec<TransitionPairRecord>,
    pub row_sum_q_plus: f64,
}

pub fn certify_direct_bins(
    run_id: &str,
    bins: &[BinAssignment],
    exact_rows: &[EnumerationSummary],
    state_records: &[StateCountRecord],
) -> Result<CertificationSummary> {
    if exact_rows.is_empty() {
        bail!("exact rows are required for direct certification");
    }

    let exact_by_key: BTreeMap<u64, &EnumerationSummary> =
        exact_rows.iter().map(|row| (row.key_id, row)).collect();
    let total_keys = exact_rows.len() as f64;
    let direct_progress = ProgressReporter::new("certify direct bins", bins.len());
    let mut bin_bounds = install_cert_pool(|| {
        bins.par_iter()
            .enumerate()
            .map(|(index, bin)| {
                let result = compute_bin_bound(index, run_id, bin, &exact_by_key, total_keys);
                direct_progress.tick();
                result
            })
            .collect::<Vec<_>>()
    })?
    .into_iter()
    .collect::<Result<Vec<_>>>()?;
    direct_progress.finish();
    bin_bounds.sort_by_key(|(index, _)| *index);
    let bin_bounds = bin_bounds.into_iter().map(|(_, record)| record).collect::<Vec<_>>();

    let (fs_bounds, transition_envelopes) = certify_finite_state_bins(run_id, bins, exact_rows, state_records)?;
    let tail_curves = build_tail_curves(run_id, &bin_bounds, &fs_bounds, exact_rows);
    Ok(CertificationSummary {
        bin_bounds,
        tail_curves,
        fs_bounds,
        transition_envelopes,
    })
}

fn build_tail_curves(
    run_id: &str,
    bin_bounds: &[BinBoundRecord],
    fs_bounds: &[FsBoundRecord],
    exact_rows: &[EnumerationSummary],
) -> Vec<TailCurvePoint> {
    let tau_grid = build_tau_grid(exact_rows);
    let exact_map = exact_tail_by_tau(exact_rows, &tau_grid);

    let mut direct_groups: BTreeMap<(String, usize), Vec<&BinBoundRecord>> = BTreeMap::new();
    for bin in bin_bounds {
        direct_groups
            .entry((bin.model.clone(), bin.bin_budget))
            .or_default()
            .push(bin);
    }
    let mut fs_groups: BTreeMap<(String, usize), Vec<&FsBoundRecord>> = BTreeMap::new();
    for bin in fs_bounds {
        fs_groups
            .entry((bin.model.clone(), bin.bin_budget))
            .or_default()
            .push(bin);
    }

    let mut out = Vec::new();
    for ((model, budget), bins) in direct_groups {
        for &tau in &tau_grid {
            out.push(TailCurvePoint {
                run_id: run_id.to_string(),
                model: model.clone(),
                bin_budget: budget,
                bound_type: "exact".to_string(),
                tau,
                tail_value: exact_map[&ordered_f64(tau)],
            });

            let tail_direct = bins
                .iter()
                .filter(|bin| bin.p_phi_max > tau)
                .map(|bin| bin.nu_phi)
                .sum::<f64>();
            out.push(TailCurvePoint {
                run_id: run_id.to_string(),
                model: model.clone(),
                bin_budget: budget,
                bound_type: "direct_bin".to_string(),
                tau,
                tail_value: tail_direct,
            });

            if let Some(fs_bins) = fs_groups.get(&(model.clone(), budget)) {
                let tail_fs = fs_bins
                    .iter()
                    .filter(|bin| bin.p_phi_fs_plus > tau)
                    .map(|bin| bin.nu_phi)
                    .sum::<f64>();
                out.push(TailCurvePoint {
                    run_id: run_id.to_string(),
                    model: model.clone(),
                    bin_budget: budget,
                    bound_type: "finite_state".to_string(),
                    tau,
                    tail_value: tail_fs,
                });
            }
        }
    }

    out
}

fn certify_finite_state_bins(
    run_id: &str,
    bins: &[BinAssignment],
    exact_rows: &[EnumerationSummary],
    state_records: &[StateCountRecord],
) -> Result<(Vec<FsBoundRecord>, Vec<TransitionEnvelopeRecord>)> {
    let exact_by_key: BTreeMap<u64, &EnumerationSummary> =
        exact_rows.iter().map(|row| (row.key_id, row)).collect();
    let total_keys = exact_rows.len() as f64;
    let fs_progress = ProgressReporter::new("certify finite-state bins", bins.len());
    let mut outputs = install_cert_pool(|| {
        bins.par_iter()
            .enumerate()
            .map(|(index, bin)| {
                let result =
                    certify_single_fs_bin(index, run_id, bin, total_keys, &exact_by_key, state_records);
                fs_progress.tick();
                result
            })
            .collect::<Vec<_>>()
    })?
    .into_iter()
    .collect::<Result<Vec<_>>>()?;
    fs_progress.finish();

    outputs.sort_by_key(|output| output.index);
    let mut fs_bounds = Vec::new();
    let mut transition_envelopes = Vec::new();
    for output in outputs {
        if let Some(bound) = output.fs_bound {
            fs_bounds.push(bound);
        }
        transition_envelopes.extend(output.transition_envelopes);
    }

    Ok((fs_bounds, transition_envelopes))
}

fn compute_bin_bound(
    index: usize,
    run_id: &str,
    bin: &BinAssignment,
    exact_by_key: &BTreeMap<u64, &EnumerationSummary>,
    total_keys: f64,
) -> Result<(usize, BinBoundRecord)> {
    let mut ph_values = Vec::with_capacity(bin.key_ids.len());
    for key_id in &bin.key_ids {
        let Some(row) = exact_by_key.get(key_id) else {
            bail!("missing exact row for key_id={key_id}");
        };
        ph_values.push(row.p_h);
    }
    ph_values.sort_by(|a, b| a.total_cmp(b));
    let key_count = ph_values.len();
    let p_phi_max = *ph_values.last().unwrap_or(&0.0);
    let p_phi_mean = ph_values.iter().sum::<f64>() / key_count as f64;
    let ph_min = ph_values[0];
    let ph_median = percentile_sorted(&ph_values, 0.5);
    let ph_p95 = percentile_sorted(&ph_values, 0.95);
    let ph_max = p_phi_max;
    Ok((
        index,
        BinBoundRecord {
            run_id: run_id.to_string(),
            model: bin.model.clone(),
            bin_budget: bin.bin_budget,
            bin_id: bin.bin_id,
            key_count,
            nu_phi: key_count as f64 / total_keys,
            p_phi_max,
            p_phi_mean,
            ph_min,
            ph_median,
            ph_p95,
            ph_max,
        },
    ))
}

struct FsBinOutput {
    index: usize,
    fs_bound: Option<FsBoundRecord>,
    transition_envelopes: Vec<TransitionEnvelopeRecord>,
}

fn certify_single_fs_bin(
    index: usize,
    run_id: &str,
    bin: &BinAssignment,
    total_keys: f64,
    exact_by_key: &BTreeMap<u64, &EnumerationSummary>,
    state_records: &[StateCountRecord],
) -> Result<FsBinOutput> {
    if bin.model == "M0" {
        let key_count = bin.key_ids.len();
        let p_phi_fs_plus = bin
            .key_ids
            .iter()
            .filter_map(|key_id| exact_by_key.get(key_id).map(|row| row.p_h))
            .fold(0.0_f64, f64::max);
        return Ok(FsBinOutput {
            index,
            fs_bound: Some(FsBoundRecord {
                run_id: run_id.to_string(),
                model: bin.model.clone(),
                bin_budget: bin.bin_budget,
                bin_id: bin.bin_id,
                key_count,
                nu_phi: key_count as f64 / total_keys,
                p_phi_fs_plus,
                alpha_total_final: p_phi_fs_plus,
                bad_mass_final: p_phi_fs_plus,
                nonterminal_mass_final: 0.0,
                row_mass_max: 1.0,
                vacuous_flag: p_phi_fs_plus >= 1.0,
            }),
            transition_envelopes: Vec::new(),
        });
    }

    let key_set = bin.key_ids.iter().copied().collect::<BTreeSet<_>>();
    let relevant = state_records
        .iter()
        .filter(|record| record.model == bin.model && key_set.contains(&record.key_id))
        .collect::<Vec<_>>();
    if relevant.is_empty() {
        return Ok(FsBinOutput {
            index,
            fs_bound: None,
            transition_envelopes: Vec::new(),
        });
    }

    let mut state_rows: BTreeMap<(usize, u64), Vec<&StateCountRecord>> = BTreeMap::new();
    let mut successor_by_pair: BTreeMap<(usize, u64, u16, u16), BTreeSet<u64>> = BTreeMap::new();
    let mut state_tuple_by_key: BTreeMap<(usize, u64), serde_json::Value> = BTreeMap::new();
    for record in &relevant {
        state_rows
            .entry((record.iteration, record.state_id))
            .or_default()
            .push(*record);
        for ml_successor in &record.ml_successor_counts {
            successor_by_pair
                .entry((record.iteration, record.state_id, ml_successor.m, ml_successor.l))
                .or_default()
                .extend(ml_successor.successor_counts.iter().map(|s| s.state_id));
        }
        state_tuple_by_key
            .entry((record.iteration, record.state_id))
            .or_insert_with(|| record.state_tuple.clone());
    }

    let succ_id = terminal_state(&bin.model, "succ").state_id;
    let bad_id = terminal_state(&bin.model, "bad").state_id;

    let mut pair_qplus: BTreeMap<(usize, u64, u16, u16), f64> = BTreeMap::new();
    let mut transition_envelopes = Vec::new();
    for ((iteration, state_id), records) in &state_rows {
        let mut q_by_pair: BTreeMap<(u16, u16), f64> = BTreeMap::new();
        for record in records {
            let denom = record.count as f64;
            for pair in &record.ml_counts {
                let q = pair.count as f64 / denom;
                let entry = q_by_pair.entry((pair.m, pair.l)).or_insert(0.0);
                *entry = entry.max(q);
            }
        }
        let row_sum_q_plus = q_by_pair.values().sum::<f64>();
        for (&(m, l), &q_plus) in &q_by_pair {
            pair_qplus.insert((*iteration, *state_id, m, l), q_plus);
        }
        transition_envelopes.push(TransitionEnvelopeRecord {
            run_id: run_id.to_string(),
            model: bin.model.clone(),
            bin_budget: bin.bin_budget,
            bin_id: bin.bin_id,
            iteration: *iteration,
            state_id: *state_id,
            state_tuple: state_tuple_by_key[&(*iteration, *state_id)].clone(),
            denom_key_count: records.len(),
            pairs: q_by_pair
                .into_iter()
                .map(|((m, l), q_plus)| TransitionPairRecord {
                    m,
                    l,
                    q_plus,
                    observed_successor_state_ids: successor_by_pair
                        .get(&(*iteration, *state_id, m, l))
                        .cloned()
                        .unwrap_or_default()
                        .into_iter()
                        .collect(),
                })
                .collect(),
            row_sum_q_plus,
        });
    }

    let mut alpha: BTreeMap<u64, f64> = BTreeMap::new();
    let mut iter0_state_probs: BTreeMap<u64, f64> = BTreeMap::new();
    for record in relevant.iter().filter(|record| record.iteration == 0) {
        let total_errors = exact_by_key[&record.key_id].total_errors as f64;
        let prob = record.count as f64 / total_errors;
        let entry = iter0_state_probs.entry(record.state_id).or_insert(0.0);
        *entry = entry.max(prob);
    }
    alpha.extend(iter0_state_probs);

    let max_iteration = relevant.iter().map(|record| record.iteration).max().unwrap_or(0);
    let mut row_mass_max = 0.0_f64;
    for iteration in 0..=max_iteration {
        let active_states = alpha.keys().copied().collect::<Vec<_>>();
        let mut next_alpha: BTreeMap<u64, f64> = BTreeMap::new();
        for state_id in active_states {
            let mass = alpha.get(&state_id).copied().unwrap_or(0.0);
            if state_id == succ_id {
                *next_alpha.entry(succ_id).or_insert(0.0) += mass;
                continue;
            }
            if state_id == bad_id {
                *next_alpha.entry(bad_id).or_insert(0.0) += mass;
                continue;
            }

            let mut kernel_row: BTreeMap<u64, f64> = BTreeMap::new();
            let pair_entries = pair_qplus
                .iter()
                .filter(|((it, sid, _, _), _)| *it == iteration && *sid == state_id)
                .collect::<Vec<_>>();

            if pair_entries.is_empty() {
                kernel_row.insert(bad_id, 1.0);
            } else {
                for ((_, _, m, l), q_plus) in pair_entries {
                    let successors = successor_by_pair
                        .get(&(iteration, state_id, *m, *l))
                        .cloned()
                        .unwrap_or_else(|| {
                            let mut set = BTreeSet::new();
                            set.insert(bad_id);
                            set
                        });
                    for successor in successors {
                        *kernel_row.entry(successor).or_insert(0.0) += *q_plus;
                    }
                }
            }

            let kernel_mass = kernel_row.values().sum::<f64>();
            row_mass_max = row_mass_max.max(kernel_mass);
            for (successor, k_plus) in kernel_row {
                *next_alpha.entry(successor).or_insert(0.0) += mass * k_plus;
            }
        }
        alpha = next_alpha;
    }

    let alpha_total_final = alpha.values().sum::<f64>();
    let bad_mass_final = alpha.get(&bad_id).copied().unwrap_or(0.0);
    let nonterminal_mass_final = alpha
        .iter()
        .filter(|(state_id, _)| **state_id != succ_id && **state_id != bad_id)
        .map(|(_, mass)| *mass)
        .sum::<f64>();
    let p_phi_fs_plus = (bad_mass_final + nonterminal_mass_final).min(1.0);

    Ok(FsBinOutput {
        index,
        fs_bound: Some(FsBoundRecord {
            run_id: run_id.to_string(),
            model: bin.model.clone(),
            bin_budget: bin.bin_budget,
            bin_id: bin.bin_id,
            key_count: bin.key_ids.len(),
            nu_phi: bin.key_ids.len() as f64 / total_keys,
            p_phi_fs_plus,
            alpha_total_final,
            bad_mass_final,
            nonterminal_mass_final,
            row_mass_max,
            vacuous_flag: p_phi_fs_plus >= 1.0,
        }),
        transition_envelopes,
    })
}

fn install_cert_pool<T, F>(work: F) -> Result<T>
where
    T: Send,
    F: FnOnce() -> T + Send,
{
    let threads = certify_thread_count();
    if threads <= 1 {
        return Ok(work());
    }
    let pool = ThreadPoolBuilder::new().num_threads(threads).build()?;
    Ok(pool.install(work))
}

fn certify_thread_count() -> usize {
    env::var("DFRCERT_CERTIFY_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or_else(|| {
            thread::available_parallelism()
                .map(|count| count.get().min(4))
                .unwrap_or(1)
        })
}

fn build_tau_grid(exact_rows: &[EnumerationSummary]) -> Vec<f64> {
    let mut values = BTreeSet::new();
    values.insert(ordered_f64(0.0));
    for row in exact_rows {
        values.insert(ordered_f64(row.p_h));
    }
    values
        .into_iter()
        .map(|v| f64::from_bits(v.0))
        .collect()
}

fn exact_tail_by_tau(
    exact_rows: &[EnumerationSummary],
    tau_grid: &[f64],
) -> BTreeMap<OrderedF64, f64> {
    let total = exact_rows.len() as f64;
    tau_grid
        .iter()
        .map(|&tau| {
            let count = exact_rows.iter().filter(|row| row.p_h > tau).count() as f64;
            (ordered_f64(tau), count / total)
        })
        .collect()
}

fn percentile_sorted(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let idx = ((values.len() - 1) as f64 * q).round() as usize;
    values[idx.min(values.len() - 1)]
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct OrderedF64(pub u64);

pub fn ordered_f64(value: f64) -> OrderedF64 {
    OrderedF64(value.to_bits())
}

#[cfg(test)]
mod tests {
    use crate::binning::BinAssignment;
    use crate::enumerate::EnumerationSummary;
    use crate::state::StateCountRecord;

    use super::certify_direct_bins;

    #[test]
    fn direct_certification_builds_bin_bounds() {
        let bins = vec![BinAssignment {
            run_id: "run".to_string(),
            model: "M0".to_string(),
            bin_budget: 1,
            bin_id: 0,
            key_ids: vec![0, 1],
            descriptor_json: "{}".to_string(),
        }];
        let rows = vec![
            EnumerationSummary {
                key_id: 0,
                p_h: 0.2,
                ..Default::default()
            },
            EnumerationSummary {
                key_id: 1,
                p_h: 0.4,
                ..Default::default()
            },
        ];
        let summary = certify_direct_bins("run", &bins, &rows, &Vec::<StateCountRecord>::new()).unwrap();
        assert_eq!(summary.bin_bounds.len(), 1);
        assert_eq!(summary.bin_bounds[0].p_phi_max, 0.4);
        assert!(!summary.tail_curves.is_empty());
        assert_eq!(summary.fs_bounds.len(), 1);
        assert_eq!(summary.fs_bounds[0].model, "M0");
    }
}
