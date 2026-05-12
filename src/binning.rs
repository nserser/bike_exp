use std::collections::BTreeMap;

use anyhow::{bail, Result};
use serde_json::json;

use crate::io::BinningSection;
use crate::profile::ProfileRecord;

#[derive(Clone, Debug)]
pub struct BinAssignment {
    pub run_id: String,
    pub model: String,
    pub bin_budget: usize,
    pub bin_id: usize,
    pub key_ids: Vec<u64>,
    pub descriptor_json: String,
}

pub fn build_all_bins(
    run_id: &str,
    profiles: &[ProfileRecord],
    binning: &BinningSection,
    models: &[String],
) -> Result<Vec<BinAssignment>> {
    let mut out = Vec::new();
    for &budget in &binning.bin_budgets {
        for model in models {
            let bins = build_bins_for_model(run_id, profiles, budget, model)?;
            out.extend(bins);
        }
    }
    Ok(out)
}

pub fn build_bins_for_model(
    run_id: &str,
    profiles: &[ProfileRecord],
    bin_budget: usize,
    model: &str,
) -> Result<Vec<BinAssignment>> {
    if bin_budget == 0 {
        bail!("bin budget must be positive");
    }
    if profiles.is_empty() {
        return Ok(Vec::new());
    }

    let mut scored: Vec<(u64, f64)> = profiles
        .iter()
        .map(|profile| (profile.key_id, model_score(model, profile)))
        .collect();
    scored.sort_by(|(key_a, score_a), (key_b, score_b)| {
        score_a
            .total_cmp(score_b)
            .then_with(|| key_a.cmp(key_b))
    });

    let n = scored.len();
    let target_bins = if model == "M0" { 1 } else { bin_budget.min(n) };
    let mut groups: BTreeMap<usize, Vec<(u64, f64)>> = BTreeMap::new();
    for (index, item) in scored.into_iter().enumerate() {
        let bin_id = index * target_bins / n;
        groups.entry(bin_id).or_default().push(item);
    }

    let mut bins = Vec::with_capacity(groups.len());
    for (bin_id, entries) in groups {
        let score_min = entries.first().map(|(_, s)| *s).unwrap_or(0.0);
        let score_max = entries.last().map(|(_, s)| *s).unwrap_or(0.0);
        let key_ids = entries.iter().map(|(key_id, _)| *key_id).collect::<Vec<_>>();
        let descriptor = json!({
            "model": model,
            "score_min": score_min,
            "score_max": score_max,
            "key_id_min": key_ids.iter().min(),
            "key_id_max": key_ids.iter().max(),
        });
        bins.push(BinAssignment {
            run_id: run_id.to_string(),
            model: model.to_string(),
            bin_budget,
            bin_id,
            key_ids,
            descriptor_json: serde_json::to_string(&descriptor)?,
        });
    }
    Ok(bins)
}

pub fn model_score(model: &str, profile: &ProfileRecord) -> f64 {
    let an_u = max_json_numeric_value(&profile.an_u_json);
    let gather = max_json_numeric_value(&profile.gather_json);
    match model {
        "M0" => 0.0,
        "M1" => profile.c4_loc as f64,
        "M2" => an_u * 1_000_000.0 + profile.d_max as f64,
        "M3_base" => profile.c4_loc as f64 + 100.0 * profile.d_max as f64,
        "M3_near" => {
            profile.c4_loc as f64
                + 100.0 * profile.d_max as f64
                + 1_000_000.0 * an_u
        }
        "M3_gather" => {
            profile.c4_loc as f64
                + 100.0 * profile.d_max as f64
                + 1_000_000.0 * an_u
                + 10_000.0 * gather
        }
        "M3_fixed" => {
            profile.c4_loc as f64
                + 100.0 * profile.d_max as f64
                + 1_000_000.0 * an_u
                + 1_000.0 * profile.b_fixed as f64
        }
        "M3_full" => {
            profile.c4_loc as f64
                + 100.0 * profile.d_max as f64
                + 1_000_000.0 * an_u
                + 10_000.0 * gather
                + 1_000.0 * profile.b_fixed as f64
                + 100.0 * profile.omega_pair_max as f64
        }
        "M3_old" | "M3" => {
            profile.c4_loc as f64
                + 100.0 * profile.d_max as f64
                + 1_000_000.0 * an_u
                + 10_000.0 * gather
                + 1_000.0 * profile.b_fixed as f64
                + 100.0 * profile.omega_pair_max as f64
        }
        _ => 0.0,
    }
}

fn max_json_numeric_value(payload: &str) -> f64 {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(payload) else {
        return 0.0;
    };
    value
        .as_object()
        .map(|obj| {
            obj.values()
                .filter_map(|v| v.as_f64())
                .fold(0.0_f64, f64::max)
        })
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::build_bins_for_model;
    use crate::profile::ProfileRecord;

    fn sample_profiles() -> Vec<ProfileRecord> {
        vec![
            ProfileRecord {
                key_id: 0,
                r: 5,
                d: 2,
                t: 2,
                c4_loc: 0,
                d_max: 1,
                lambda_json: "{}".to_string(),
                an_u_json: "{\"1\":0.1}".to_string(),
                gather_json: "{\"1:1\":2}".to_string(),
                b_fixed: 0,
                omega_pair_max: 0,
                profile_hash: "a".to_string(),
            },
            ProfileRecord {
                key_id: 1,
                r: 5,
                d: 2,
                t: 2,
                c4_loc: 4,
                d_max: 2,
                lambda_json: "{}".to_string(),
                an_u_json: "{\"1\":0.5}".to_string(),
                gather_json: "{\"1:1\":5}".to_string(),
                b_fixed: 2,
                omega_pair_max: 2,
                profile_hash: "b".to_string(),
            },
        ]
    }

    #[test]
    fn m0_produces_single_bin() {
        let bins = build_bins_for_model("run", &sample_profiles(), 8, "M0").unwrap();
        assert_eq!(bins.len(), 1);
        assert_eq!(bins[0].key_ids.len(), 2);
    }

    #[test]
    fn quantile_bins_respect_budget() {
        let bins = build_bins_for_model("run", &sample_profiles(), 2, "M1").unwrap();
        assert!(bins.len() <= 2);
        assert_eq!(bins.iter().map(|b| b.key_ids.len()).sum::<usize>(), 2);
    }
}
