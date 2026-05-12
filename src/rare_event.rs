use anyhow::Result;
use rayon::prelude::*;

use crate::bitset::Mask256;
use crate::combinatorics::iter_combinations;
use crate::decoder::{decode_fast, Outcome};
use crate::io::RunConfig;
use crate::key::{build_key_runtime, Key};
use crate::progress::ProgressReporter;

#[derive(Clone, Debug)]
pub struct RareEventRecord {
    pub run_id: String,
    pub key_id: u64,
    pub u: usize,
    pub tail_mass_a_union: f64,
    pub conditional_total: u64,
    pub conditional_failures: u64,
    pub conditional_failure_rate: f64,
    pub contribution_proxy: f64,
    pub exact_conditional: bool,
}

pub fn run_rare_event_diagnostics(
    run_id: &str,
    keys: &[Key],
    config: &RunConfig,
) -> Result<Vec<RareEventRecord>> {
    let selected_keys = keys.iter().take(32).cloned().collect::<Vec<_>>();
    let progress = ProgressReporter::new("rare-event keys", selected_keys.len());
    let mut grouped = selected_keys
        .into_par_iter()
        .enumerate()
        .map(|(index, key)| {
            let runtime = build_key_runtime(key)?;
            let mut records = Vec::with_capacity(config.profile.u_values.len());
            for &u in &config.profile.u_values {
                let mut conditional_total = 0u64;
                let mut conditional_failures = 0u64;
                let bit_count = 2 * runtime.key.r;

                if config.sampling.error_sample_count == 0 && bit_count <= 64 {
                    for combo in iter_combinations(bit_count, config.parameters.t)? {
                        let mut error = Mask256::empty();
                        for bit in combo {
                            error.set(bit)?;
                        }
                        if near_overlap(&runtime, error) < u {
                            continue;
                        }
                        conditional_total += 1;
                        if decode_fast(&runtime, error, &config.decoder)? != Outcome::Success {
                            conditional_failures += 1;
                        }
                    }
                } else {
                    let draws = config.sampling.error_sample_count.max(5000).min(20000);
                    let mut rng = XorShift64::new(config.run.seed ^ runtime.key.key_id);
                    for _ in 0..draws {
                        let error = sample_error_mask(bit_count, config.parameters.t, &mut rng)?;
                        if near_overlap(&runtime, error) < u {
                            continue;
                        }
                        conditional_total += 1;
                        if decode_fast(&runtime, error, &config.decoder)? != Outcome::Success {
                            conditional_failures += 1;
                        }
                    }
                }

                let conditional_failure_rate = if conditional_total == 0 {
                    0.0
                } else {
                    conditional_failures as f64 / conditional_total as f64
                };
                let tail_mass =
                    near_tail_mass_union(runtime.key.r, runtime.key.d, config.parameters.t, u);
                records.push(RareEventRecord {
                    run_id: run_id.to_string(),
                    key_id: runtime.key.key_id,
                    u,
                    tail_mass_a_union: tail_mass,
                    conditional_total,
                    conditional_failures,
                    conditional_failure_rate,
                    contribution_proxy: tail_mass * conditional_failure_rate,
                    exact_conditional: config.sampling.error_sample_count == 0 && bit_count <= 64,
                });
            }
            progress.tick();
            Ok((index, records))
        })
        .collect::<Vec<Result<(usize, Vec<RareEventRecord>)>>>();

    grouped.sort_by_key(|entry| match entry {
        Ok((index, _)) => *index,
        Err(_) => usize::MAX,
    });

    let mut out = Vec::new();
    for entry in grouped {
        let (_, mut records) = entry?;
        out.append(&mut records);
    }
    progress.finish();
    Ok(out)
}

fn near_tail_mass_union(r: usize, d: usize, t: usize, u: usize) -> f64 {
    let family_size = 2 * r;
    let tail = crate::combinatorics::hypergeom_tail(2 * r, d, t, u);
    (family_size as f64 * tail).min(1.0)
}

fn near_overlap(key_rt: &crate::key::KeyRuntime, residual: Mask256) -> usize {
    let r = key_rt.key.r;
    let mut block0 = Vec::new();
    let mut block1 = Vec::new();
    for pos in residual.iter_ones(2 * r) {
        if pos < r {
            block0.push(pos);
        } else {
            block1.push(pos - r);
        }
    }

    let mut best = 0usize;
    for shift in 0..r {
        let overlap0 = block0
            .iter()
            .filter(|&&pos| key_rt.key.h0.contains(&(((pos + r - shift) % r) as u16)))
            .count();
        let overlap1 = block1
            .iter()
            .filter(|&&pos| key_rt.key.h1.contains(&(((pos + r - shift) % r) as u16)))
            .count();
        best = best.max(overlap0).max(overlap1);
    }
    best
}

fn sample_error_mask(bit_count: usize, weight: usize, rng: &mut XorShift64) -> Result<Mask256> {
    let mut values = (0..bit_count).collect::<Vec<_>>();
    for idx in 0..weight {
        let swap_idx = idx + rng.gen_range(bit_count - idx);
        values.swap(idx, swap_idx);
    }
    let mut mask = Mask256::empty();
    for &bit in &values[..weight] {
        mask.set(bit)?;
    }
    Ok(mask)
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
    use crate::io::{
        BinningSection, DecoderSection, MetricsSection, ParametersSection, ProfileSection, RunConfig,
        RunSection, SamplingSection, StateSection,
    };
    use crate::key::Key;

    use super::run_rare_event_diagnostics;

    #[test]
    fn rare_event_generates_records() {
        let config = RunConfig {
            run: RunSection {
                run_id: "rare".to_string(),
                seed: 1,
                mode: "sampled_keys".to_string(),
                output_dir: "artifacts/rare".to_string(),
            },
            parameters: ParametersSection { r: 5, d: 2, t: 2 },
            decoder: DecoderSection {
                iterations: 2,
                thresholds: vec![2, 2],
                wrong_zero_is_failure: true,
                flip_policy: "all_ge_threshold".to_string(),
            },
            profile: ProfileSection {
                u_values: vec![1, 2],
                gathering_m_max: 2,
                gathering_l_max: 2,
                fixedpoint_m_max: 2,
                fixedpoint_thresholds: vec![2],
            },
            state: StateSection {
                models: vec!["M2".to_string(), "M3".to_string()],
                u_bins: vec![0, 1, 2, 99],
                omega_bins: vec![0, 1, 2, 99],
                strict_absorbing: false,
            },
            binning: BinningSection {
                bin_budgets: vec![2],
                quantile_ties: "stable".to_string(),
            },
            sampling: SamplingSection {
                key_count: 2,
                error_sample_count: 0,
                train_fraction: 1.0,
            },
            metrics: MetricsSection {
                q_values: vec![1],
                top_tail_fractions: vec![0.1],
            },
        };
        let keys = vec![Key {
            key_id: 0,
            r: 5,
            d: 2,
            h0: vec![0, 1],
            h1: vec![0, 2],
        }];
        let records = run_rare_event_diagnostics("rare", &keys, &config).unwrap();
        assert_eq!(records.len(), 2);
    }
}
