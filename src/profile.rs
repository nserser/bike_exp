use std::collections::BTreeMap;

use anyhow::Result;
use rayon::prelude::*;
use serde_json::json;

use crate::bitset::Mask256;
use crate::combinatorics::{hypergeom_tail, iter_combinations};
use crate::io::ProfileSection;
use crate::key::{build_key_runtime, Key, KeyRuntime};
use crate::progress::ProgressReporter;

#[derive(Clone, Debug)]
pub struct ProfileRecord {
    pub key_id: u64,
    pub r: usize,
    pub d: usize,
    pub t: usize,
    pub c4_loc: u64,
    pub d_max: u32,
    pub lambda_json: String,
    pub an_u_json: String,
    pub gather_json: String,
    pub b_fixed: u64,
    pub omega_pair_max: u32,
    pub profile_hash: String,
}

pub fn compute_profiles(
    keys: &[Key],
    t: usize,
    profile_config: &ProfileSection,
) -> Result<Vec<ProfileRecord>> {
    let progress = ProgressReporter::new("profiles", keys.len());
    let mut out = keys
        .par_iter()
        .enumerate()
        .map(|(index, key)| {
            let runtime = build_key_runtime(key.clone())?;
            let profile = compute_profile(&runtime, t, profile_config)?;
            progress.tick();
            Ok((index, profile))
        })
        .collect::<Vec<Result<(usize, ProfileRecord)>>>();

    let mut ordered = Vec::with_capacity(out.len());
    out.sort_by_key(|entry| match entry {
        Ok((index, _)) => *index,
        Err(_) => usize::MAX,
    });
    for entry in out {
        ordered.push(entry?.1);
    }
    progress.finish();
    Ok(ordered)
}

pub fn compute_profile(
    key_rt: &KeyRuntime,
    t: usize,
    profile_config: &ProfileSection,
) -> Result<ProfileRecord> {
    let overlaps = compute_overlap_summary(&key_rt.key);
    let an_u = compute_near_codeword_union_tails(key_rt, t, &profile_config.u_values);
    let gather = compute_gathering_counts(
        key_rt,
        profile_config.gathering_m_max,
        profile_config.gathering_l_max,
    )?;
    let b_fixed = compute_fixed_point_count(
        key_rt,
        profile_config.fixedpoint_m_max,
        &profile_config.fixedpoint_thresholds,
    )?;
    let omega_pair_max = compute_omega_pair_max(key_rt);

    let lambda_json = serde_json::to_string(&overlaps.lambda)?;
    let an_u_json = serde_json::to_string(&an_u)?;
    let gather_json = serde_json::to_string(&gather)?;

    let profile_hash = profile_hash(&json!({
        "c4_loc": overlaps.c4_loc,
        "d_max": overlaps.d_max,
        "lambda": overlaps.lambda,
        "an_u": an_u,
        "gather": gather,
        "b_fixed": b_fixed,
        "omega_pair_max": omega_pair_max,
    }))?;

    Ok(ProfileRecord {
        key_id: key_rt.key.key_id,
        r: key_rt.key.r,
        d: key_rt.key.d,
        t,
        c4_loc: overlaps.c4_loc,
        d_max: overlaps.d_max,
        lambda_json,
        an_u_json,
        gather_json,
        b_fixed,
        omega_pair_max,
        profile_hash,
    })
}

fn compute_overlap_summary(key: &Key) -> OverlapSummary {
    let mut c4_loc = 0u64;
    let mut d_max = 0u32;
    let mut lambda: BTreeMap<u32, u32> = BTreeMap::new();

    for a in 0..2 {
        for b in 0..2 {
            for delta in 0..key.r {
                let overlap = overlap_at_delta(key, a, b, delta) as u32;
                if a == b && delta == 0 {
                    continue;
                }
                c4_loc += binom2(overlap) as u64;
                d_max = d_max.max(overlap);
                *lambda.entry(overlap).or_insert(0) += 1;
            }
        }
    }

    OverlapSummary { c4_loc, d_max, lambda }
}

fn compute_near_codeword_union_tails(
    key_rt: &KeyRuntime,
    t: usize,
    u_values: &[usize],
) -> BTreeMap<String, f64> {
    let family_size = 2 * key_rt.key.r;
    let universe = 2 * key_rt.key.r;
    let marked = key_rt.key.d;

    let mut out = BTreeMap::new();
    for &u in u_values {
        let tail = hypergeom_tail(universe, marked, t, u);
        out.insert(u.to_string(), (family_size as f64 * tail).min(1.0));
    }
    out
}

fn compute_gathering_counts(
    key_rt: &KeyRuntime,
    m_max: usize,
    l_max: usize,
) -> Result<BTreeMap<String, u64>> {
    let mut out = BTreeMap::new();
    let bit_count = 2 * key_rt.key.r;
    for m in 1..=m_max.min(bit_count) {
        for combo in iter_combinations(bit_count, m)? {
            let syndrome_weight = syndrome_weight_for_positions(key_rt, &combo)?;
            if syndrome_weight <= l_max {
                *out.entry(format!("{m}:{syndrome_weight}")).or_insert(0) += 1;
            }
        }
    }
    Ok(out)
}

fn compute_fixed_point_count(
    key_rt: &KeyRuntime,
    m_max: usize,
    thresholds: &[usize],
) -> Result<u64> {
    let bit_count = 2 * key_rt.key.r;
    let mut count = 0u64;

    for m in 1..=m_max.min(bit_count) {
        for combo in iter_combinations(bit_count, m)? {
            let mut residual = Mask256::empty();
            for bit in combo {
                residual.set(bit)?;
            }

            let syndrome = syndrome_for_mask(key_rt, residual);
            if syndrome.is_zero() {
                continue;
            }

            if thresholds.iter().copied().any(|threshold| is_fixed_point_candidate(key_rt, syndrome, threshold)) {
                count += 1;
            }
        }
    }

    Ok(count)
}

fn compute_omega_pair_max(key_rt: &KeyRuntime) -> u32 {
    let bit_count = 2 * key_rt.key.r;
    let mut max_val = 0u32;
    for j in 0..bit_count {
        for k in j + 1..bit_count {
            let overlap = key_rt.column_masks[j].and(key_rt.column_masks[k]).popcount();
            if overlap >= 2 {
                max_val = max_val.max(overlap);
            }
        }
    }
    max_val
}

fn syndrome_weight_for_positions(key_rt: &KeyRuntime, positions: &[usize]) -> Result<usize> {
    let mut mask = Mask256::empty();
    for &pos in positions {
        mask.set(pos)?;
    }
    Ok(syndrome_for_mask(key_rt, mask).popcount() as usize)
}

fn syndrome_for_mask(key_rt: &KeyRuntime, mask: Mask256) -> crate::bitset::SyndromeMask {
    let mut syndrome = crate::bitset::SyndromeMask::empty();
    for pos in mask.iter_ones(2 * key_rt.key.r) {
        syndrome.xor_assign(key_rt.column_masks[pos]);
    }
    syndrome
}

fn is_fixed_point_candidate(
    key_rt: &KeyRuntime,
    syndrome: crate::bitset::SyndromeMask,
    threshold: usize,
) -> bool {
    (0..2 * key_rt.key.r)
        .all(|pos| (key_rt.column_masks[pos].and(syndrome).popcount() as usize) < threshold)
}

fn overlap_at_delta(key: &Key, a: usize, b: usize, delta: usize) -> usize {
    let left = if a == 0 { &key.h0 } else { &key.h1 };
    let right = if b == 0 { &key.h0 } else { &key.h1 };
    let mut count = 0usize;
    for &x in left {
        let x = usize::from(x);
        for &y in right {
            let shifted = (usize::from(y) + delta) % key.r;
            if x == shifted {
                count += 1;
            }
        }
    }
    count
}

fn binom2(x: u32) -> u32 {
    x.saturating_mul(x.saturating_sub(1)) / 2
}

fn profile_hash(value: &serde_json::Value) -> Result<String> {
    let bytes = serde_json::to_vec(value)?;
    Ok(crate::io::hex_sha256(&bytes))
}

struct OverlapSummary {
    c4_loc: u64,
    d_max: u32,
    lambda: BTreeMap<u32, u32>,
}

#[cfg(test)]
mod tests {
    use crate::io::ProfileSection;
    use crate::key::{build_key_runtime, Key};

    use super::{compute_profile, compute_profiles};

    fn test_profile_config() -> ProfileSection {
        ProfileSection {
            u_values: vec![1, 2],
            gathering_m_max: 2,
            gathering_l_max: 2,
            fixedpoint_m_max: 2,
            fixedpoint_thresholds: vec![2],
        }
    }

    #[test]
    fn profile_extracts_expected_overlap_statistics() {
        let key = Key {
            key_id: 0,
            r: 5,
            d: 2,
            h0: vec![0, 1],
            h1: vec![0, 2],
        };
        let runtime = build_key_runtime(key).unwrap();
        let profile = compute_profile(&runtime, 2, &test_profile_config()).unwrap();
        assert!(profile.c4_loc <= 12);
        assert!(profile.d_max <= 2);
        assert!(profile.lambda_json.contains("\"1\""));
        assert!(profile.an_u_json.contains("\"1\""));
        assert!(!profile.profile_hash.is_empty());
        assert_eq!(profile.r, 5);
        assert_eq!(profile.d, 2);
        assert_eq!(profile.t, 2);
    }

    #[test]
    fn profile_hash_is_stable_for_same_key() {
        let key = Key {
            key_id: 1,
            r: 5,
            d: 2,
            h0: vec![0, 2],
            h1: vec![1, 4],
        };
        let runtime = build_key_runtime(key.clone()).unwrap();
        let p1 = compute_profile(&runtime, 2, &test_profile_config()).unwrap();
        let runtime2 = build_key_runtime(key).unwrap();
        let p2 = compute_profile(&runtime2, 2, &test_profile_config()).unwrap();
        assert_eq!(p1.profile_hash, p2.profile_hash);
    }

    #[test]
    fn compute_profiles_preserves_key_count() {
        let keys = vec![
            Key {
                key_id: 0,
                r: 5,
                d: 2,
                h0: vec![0, 1],
                h1: vec![0, 2],
            },
            Key {
                key_id: 1,
                r: 5,
                d: 2,
                h0: vec![0, 3],
                h1: vec![1, 2],
            },
        ];
        let profiles = compute_profiles(&keys, 2, &test_profile_config()).unwrap();
        assert_eq!(profiles.len(), 2);
        assert_eq!(profiles[0].key_id, 0);
        assert_eq!(profiles[1].key_id, 1);
        assert!(!profiles[0].lambda_json.is_empty());
        assert!(!profiles[1].lambda_json.is_empty());
    }
}
