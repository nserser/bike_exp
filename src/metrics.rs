use std::collections::{BTreeMap, BTreeSet};

use crate::binning::BinAssignment;
use crate::certify::{BinBoundRecord, FsBoundRecord, TailCurvePoint};
use crate::enumerate::EnumerationSummary;

#[derive(Clone, Debug)]
pub struct MetricRecord {
    pub run_id: String,
    pub model: String,
    pub bin_budget: usize,
    pub bound_type: String,
    pub coverage: f64,
    pub median_looseness: f64,
    pub p95_looseness: f64,
    pub max_looseness: f64,
    pub vacuous_bin_fraction: f64,
    pub spearman: f64,
    pub tail_l1_error: f64,
    pub delta_bound: f64,
    pub delta_exact: f64,
}

#[derive(Clone, Debug)]
pub struct TopTailRecord {
    pub run_id: String,
    pub model: String,
    pub bin_budget: usize,
    pub bound_type: String,
    pub tail_fraction: f64,
    pub precision: f64,
    pub recall: f64,
}

pub fn compute_direct_metrics(
    run_id: &str,
    bins: &[BinAssignment],
    bin_bounds: &[BinBoundRecord],
    fs_bounds: &[FsBoundRecord],
    tail_curves: &[TailCurvePoint],
    exact_rows: &[EnumerationSummary],
    top_tail_fractions: &[f64],
) -> (Vec<MetricRecord>, Vec<TopTailRecord>) {
    let exact_map: BTreeMap<u64, f64> = exact_rows.iter().map(|row| (row.key_id, row.p_h)).collect();
    let delta_exact = if exact_rows.is_empty() {
        0.0
    } else {
        exact_rows.iter().map(|row| row.p_h).sum::<f64>() / exact_rows.len() as f64
    };

    let mut assignments: BTreeMap<(String, usize), Vec<(u64, f64)>> = BTreeMap::new();
    let bound_lookup: BTreeMap<(String, usize, usize), &BinBoundRecord> = bin_bounds
        .iter()
        .map(|bin| ((bin.model.clone(), bin.bin_budget, bin.bin_id), bin))
        .collect();
    for bin in bins {
        let Some(bound) = bound_lookup.get(&(bin.model.clone(), bin.bin_budget, bin.bin_id)) else {
            continue;
        };
        for key_id in &bin.key_ids {
            assignments
                .entry((bin.model.clone(), bin.bin_budget))
                .or_default()
                .push((*key_id, bound.p_phi_max));
        }
    }

    let mut metrics = Vec::new();
    let mut top_tail = Vec::new();
    for ((model, budget), key_scores) in &assignments {
        metrics.push(metric_for_group(
            run_id,
            model,
            *budget,
            "direct_bin",
            key_scores,
            &exact_map,
            tail_curves,
            bin_bounds
                .iter()
                .filter(|bin| bin.model == *model && bin.bin_budget == *budget)
                .map(|bin| (bin.nu_phi, bin.p_phi_max, bin.p_phi_max >= 1.0))
                .collect(),
            delta_exact,
        ));

        top_tail.extend(compute_top_tail(
            run_id,
            model,
            *budget,
            "direct_bin",
            key_scores,
            &exact_map,
            top_tail_fractions,
        ));
    }

    if !fs_bounds.is_empty() {
        let fs_lookup: BTreeMap<(String, usize, usize), &FsBoundRecord> = fs_bounds
            .iter()
            .map(|bin| ((bin.model.clone(), bin.bin_budget, bin.bin_id), bin))
            .collect();
        let mut fs_assignments: BTreeMap<(String, usize), Vec<(u64, f64)>> = BTreeMap::new();
        for bin in bins {
            let Some(bound) = fs_lookup.get(&(bin.model.clone(), bin.bin_budget, bin.bin_id)) else {
                continue;
            };
            for key_id in &bin.key_ids {
                fs_assignments
                    .entry((bin.model.clone(), bin.bin_budget))
                    .or_default()
                    .push((*key_id, bound.p_phi_fs_plus));
            }
        }
        for ((model, budget), key_scores) in &fs_assignments {
            metrics.push(metric_for_group(
                run_id,
                model,
                *budget,
                "finite_state",
                key_scores,
                &exact_map,
                tail_curves,
                fs_bounds
                    .iter()
                    .filter(|bin| bin.model == *model && bin.bin_budget == *budget)
                    .map(|bin| (bin.nu_phi, bin.p_phi_fs_plus, bin.vacuous_flag))
                    .collect(),
                delta_exact,
            ));

            top_tail.extend(compute_top_tail(
                run_id,
                model,
                *budget,
                "finite_state",
                key_scores,
                &exact_map,
                top_tail_fractions,
            ));
        }
    }

    (metrics, top_tail)
}

fn metric_for_group(
    run_id: &str,
    model: &str,
    budget: usize,
    bound_type: &str,
    key_scores: &[(u64, f64)],
    exact_map: &BTreeMap<u64, f64>,
    tail_curves: &[TailCurvePoint],
    bin_summaries: Vec<(f64, f64, bool)>,
    delta_exact: f64,
) -> MetricRecord {
    let mut looseness = Vec::new();
    let mut zero_gaps = Vec::new();
    let mut coverage_hits = 0usize;
    for (key_id, bound) in key_scores {
        let exact = exact_map[key_id];
        if exact <= *bound + 1e-15 {
            coverage_hits += 1;
        }
        if exact > 0.0 {
            looseness.push(bound / exact);
        } else {
            zero_gaps.push(*bound);
        }
    }

    let vacuous_bin_fraction = if bin_summaries.is_empty() {
        0.0
    } else {
        bin_summaries.iter().filter(|(_, _, vacuous)| *vacuous).count() as f64
            / bin_summaries.len() as f64
    };
    let delta_bound = bin_summaries.iter().map(|(nu, bound, _)| nu * bound).sum::<f64>();
    let spearman = spearman_for_scores(key_scores, exact_map);
    let tail_l1_error = tail_l1_for_group(model, budget, bound_type, tail_curves);

    let mut looseness_sorted = looseness;
    looseness_sorted.sort_by(|a, b| a.total_cmp(b));
    let median_looseness = percentile_sorted_or_zero(&looseness_sorted, 0.5);
    let p95_looseness = percentile_sorted_or_zero(&looseness_sorted, 0.95);
    let max_looseness = looseness_sorted
        .last()
        .copied()
        .or_else(|| zero_gaps.iter().copied().max_by(|a, b| a.total_cmp(b)))
        .unwrap_or(0.0);

    MetricRecord {
        run_id: run_id.to_string(),
        model: model.to_string(),
        bin_budget: budget,
        bound_type: bound_type.to_string(),
        coverage: coverage_hits as f64 / key_scores.len() as f64,
        median_looseness,
        p95_looseness,
        max_looseness,
        vacuous_bin_fraction,
        spearman,
        tail_l1_error,
        delta_bound,
        delta_exact,
    }
}

fn compute_top_tail(
    run_id: &str,
    model: &str,
    budget: usize,
    bound_type: &str,
    key_scores: &[(u64, f64)],
    exact_map: &BTreeMap<u64, f64>,
    fractions: &[f64],
) -> Vec<TopTailRecord> {
    let mut exact_ranked = key_scores
        .iter()
        .map(|(key_id, score)| (*key_id, exact_map[key_id], *score))
        .collect::<Vec<_>>();
    exact_ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| b.0.cmp(&a.0)));

    let mut score_ranked = exact_ranked.clone();
    score_ranked.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| b.0.cmp(&a.0)));

    let mut out = Vec::new();
    for &fraction in fractions {
        let k = ((key_scores.len() as f64 * fraction).ceil() as usize).max(1).min(key_scores.len());
        let exact_top: BTreeSet<u64> = exact_ranked.iter().take(k).map(|(key_id, _, _)| *key_id).collect();
        let score_top: BTreeSet<u64> = score_ranked.iter().take(k).map(|(key_id, _, _)| *key_id).collect();
        let hits = score_top.intersection(&exact_top).count() as f64;
        out.push(TopTailRecord {
            run_id: run_id.to_string(),
            model: model.to_string(),
            bin_budget: budget,
            bound_type: bound_type.to_string(),
            tail_fraction: fraction,
            precision: hits / score_top.len() as f64,
            recall: hits / exact_top.len() as f64,
        });
    }
    out
}

fn spearman_for_scores(key_scores: &[(u64, f64)], exact_map: &BTreeMap<u64, f64>) -> f64 {
    let exact_values = key_scores.iter().map(|(key_id, _)| exact_map[key_id]).collect::<Vec<_>>();
    let score_values = key_scores.iter().map(|(_, score)| *score).collect::<Vec<_>>();
    spearman_rank_corr(&score_values, &exact_values)
}

fn spearman_rank_corr(xs: &[f64], ys: &[f64]) -> f64 {
    if xs.len() != ys.len() || xs.is_empty() {
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
    if den_x == 0.0 || den_y == 0.0 {
        0.0
    } else {
        num / (den_x.sqrt() * den_y.sqrt())
    }
}

fn tail_l1_for_group(model: &str, budget: usize, bound_type: &str, tail_curves: &[TailCurvePoint]) -> f64 {
    let exact = tail_curves
        .iter()
        .filter(|point| point.model == model && point.bin_budget == budget && point.bound_type == "exact")
        .map(|point| (point.tau.to_bits(), point.tail_value))
        .collect::<BTreeMap<_, _>>();
    let direct = tail_curves
        .iter()
        .filter(|point| point.model == model && point.bin_budget == budget && point.bound_type == bound_type)
        .map(|point| (point.tau.to_bits(), point.tail_value))
        .collect::<BTreeMap<_, _>>();
    if exact.is_empty() {
        return 0.0;
    }
    exact.keys()
        .map(|tau| (direct[tau] - exact[tau]).max(0.0))
        .sum::<f64>()
        / exact.len() as f64
}

fn percentile_sorted_or_zero(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        let idx = ((values.len() - 1) as f64 * q).round() as usize;
        values[idx.min(values.len() - 1)]
    }
}

#[cfg(test)]
mod tests {
    use crate::binning::BinAssignment;
    use crate::certify::{BinBoundRecord, FsBoundRecord, TailCurvePoint};
    use crate::enumerate::EnumerationSummary;

    use super::compute_direct_metrics;

    #[test]
    fn direct_metrics_compute_coverage() {
        let bins = vec![BinAssignment {
            run_id: "run".to_string(),
            model: "M0".to_string(),
            bin_budget: 1,
            bin_id: 0,
            key_ids: vec![0, 1],
            descriptor_json: "{}".to_string(),
        }];
        let bin_bounds = vec![BinBoundRecord {
            run_id: "run".to_string(),
            model: "M0".to_string(),
            bin_budget: 1,
            bin_id: 0,
            key_count: 2,
            nu_phi: 1.0,
            p_phi_max: 0.4,
            p_phi_mean: 0.3,
            ph_min: 0.2,
            ph_median: 0.3,
            ph_p95: 0.4,
            ph_max: 0.4,
        }];
        let tails = vec![
            TailCurvePoint {
                run_id: "run".to_string(),
                model: "M0".to_string(),
                bin_budget: 1,
                bound_type: "exact".to_string(),
                tau: 0.0,
                tail_value: 1.0,
            },
            TailCurvePoint {
                run_id: "run".to_string(),
                model: "M0".to_string(),
                bin_budget: 1,
                bound_type: "direct_bin".to_string(),
                tau: 0.0,
                tail_value: 1.0,
            },
        ];
        let exact = vec![
            EnumerationSummary { key_id: 0, p_h: 0.2, ..Default::default() },
            EnumerationSummary { key_id: 1, p_h: 0.4, ..Default::default() },
        ];
        let (metrics, top_tail) =
            compute_direct_metrics("run", &bins, &bin_bounds, &Vec::<FsBoundRecord>::new(), &tails, &exact, &[0.5]);
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].coverage, 1.0);
        assert_eq!(top_tail.len(), 1);
    }
}
