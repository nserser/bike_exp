use std::collections::HashSet;

use anyhow::{bail, Result};

use crate::bitset::SyndromeMask;
use crate::combinatorics::iter_combinations;

#[derive(Clone, Debug)]
pub struct Key {
    pub key_id: u64,
    pub r: usize,
    pub d: usize,
    pub h0: Vec<u16>,
    pub h1: Vec<u16>,
}

#[derive(Clone, Debug)]
pub struct KeyRuntime {
    pub key: Key,
    pub column_masks: Vec<SyndromeMask>,
}

impl Key {
    pub fn validate(&self) -> Result<()> {
        validate_support(&self.h0, self.r, self.d)?;
        validate_support(&self.h1, self.r, self.d)?;
        Ok(())
    }
}

pub fn enumerate_all_keys(r: usize, d: usize) -> Result<Vec<Key>> {
    let mut out = Vec::new();
    let mut key_id = 0u64;
    let h1_combos: Vec<Vec<usize>> = iter_combinations(r, d)?.collect();
    for h0 in iter_combinations(r, d)? {
        for h1 in &h1_combos {
            let key = Key {
                key_id,
                r,
                d,
                h0: h0.iter().map(|v| *v as u16).collect(),
                h1: h1.iter().map(|v| *v as u16).collect(),
            };
            key.validate()?;
            out.push(key);
            key_id += 1;
        }
    }
    Ok(out)
}

pub fn sample_keys_uniform(r: usize, d: usize, n: usize, seed: u64) -> Vec<Key> {
    let mut rng = XorShift64::new(seed);
    let mut seen = HashSet::new();
    let mut out = Vec::with_capacity(n);
    let max_attempts = n.saturating_mul(100).max(1000);

    for _ in 0..max_attempts {
        if out.len() >= n {
            break;
        }
        let h0 = random_support(r, d, &mut rng);
        let h1 = random_support(r, d, &mut rng);
        let key = Key {
            key_id: out.len() as u64,
            r,
            d,
            h0,
            h1,
        };
        let support = key_to_support_string(&key);
        if seen.insert(support) && key.validate().is_ok() {
            out.push(key);
        }
    }

    out
}

pub fn build_column_masks(key: &Key) -> Result<Vec<SyndromeMask>> {
    key.validate()?;
    if key.r > 128 {
        bail!("r={} exceeds SyndromeMask capacity", key.r);
    }

    let mut masks = Vec::with_capacity(2 * key.r);
    for z in 0..key.r {
        masks.push(rotated_support_mask(&key.h0, z, key.r)?);
    }
    for z in 0..key.r {
        masks.push(rotated_support_mask(&key.h1, z, key.r)?);
    }
    Ok(masks)
}

pub fn build_key_runtime(key: Key) -> Result<KeyRuntime> {
    let column_masks = build_column_masks(&key)?;
    Ok(KeyRuntime { key, column_masks })
}

pub fn key_to_support_string(key: &Key) -> String {
    let left = key.h0.iter().map(u16::to_string).collect::<Vec<_>>().join(";");
    let right = key.h1.iter().map(u16::to_string).collect::<Vec<_>>().join(";");
    format!("{left}|{right}")
}

fn validate_support(support: &[u16], r: usize, d: usize) -> Result<()> {
    if support.len() != d {
        bail!("support length {} does not match d={d}", support.len());
    }

    let mut prev = None;
    for &value in support {
        let idx = usize::from(value);
        if idx >= r {
            bail!("support entry {idx} is out of range for r={r}");
        }
        if let Some(prev_value) = prev {
            if idx <= prev_value {
                bail!("support entries must be strictly increasing");
            }
        }
        prev = Some(idx);
    }
    Ok(())
}

fn rotated_support_mask(support: &[u16], shift: usize, r: usize) -> Result<SyndromeMask> {
    let mut mask = SyndromeMask::empty();
    for &value in support {
        let rotated = (usize::from(value) + shift) % r;
        mask.set(rotated)?;
    }
    Ok(mask)
}

fn random_support(r: usize, d: usize, rng: &mut XorShift64) -> Vec<u16> {
    let mut values = (0..r as u16).collect::<Vec<_>>();
    for idx in 0..d {
        let swap_idx = idx + rng.gen_range(r - idx);
        values.swap(idx, swap_idx);
    }
    let mut support = values[..d].to_vec();
    support.sort_unstable();
    support
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
    use crate::combinatorics::binom_u128;

    use super::{build_column_masks, build_key_runtime, enumerate_all_keys, key_to_support_string, sample_keys_uniform, Key};

    #[test]
    fn enumerate_all_keys_matches_expected_count_and_order() {
        let keys = enumerate_all_keys(5, 2).unwrap();
        assert_eq!(keys.len() as u128, binom_u128(5, 2).pow(2));
        assert_eq!(keys[0].key_id, 0);
        assert_eq!(keys[0].h0, vec![0, 1]);
        assert_eq!(keys[0].h1, vec![0, 1]);
        assert_eq!(keys[1].h0, vec![0, 1]);
        assert_eq!(keys[1].h1, vec![0, 2]);
        assert_eq!(keys[10].h0, vec![0, 2]);
        assert_eq!(keys[10].h1, vec![0, 1]);
    }

    #[test]
    fn build_column_masks_rotates_supports_per_block() {
        let key = Key {
            key_id: 7,
            r: 5,
            d: 2,
            h0: vec![0, 2],
            h1: vec![1, 4],
        };
        let masks = build_column_masks(&key).unwrap();
        assert_eq!(masks.len(), 10);
        assert_eq!(masks[0].popcount(), 2);
        assert!(masks[0].contains(0));
        assert!(masks[0].contains(2));
        assert!(masks[1].contains(1));
        assert!(masks[1].contains(3));
        assert!(masks[5].contains(1));
        assert!(masks[5].contains(4));
        assert!(masks[6].contains(0));
        assert!(masks[6].contains(2));
    }

    #[test]
    fn key_runtime_builds_column_masks() {
        let key = Key {
            key_id: 1,
            r: 5,
            d: 2,
            h0: vec![0, 3],
            h1: vec![1, 2],
        };
        let runtime = build_key_runtime(key.clone()).unwrap();
        assert_eq!(runtime.key.key_id, key.key_id);
        assert_eq!(runtime.column_masks.len(), 2 * key.r);
    }

    #[test]
    fn key_support_string_is_stable() {
        let key = Key {
            key_id: 99,
            r: 7,
            d: 3,
            h0: vec![0, 2, 6],
            h1: vec![1, 3, 5],
        };
        assert_eq!(key_to_support_string(&key), "0;2;6|1;3;5");
    }

    #[test]
    fn invalid_support_is_rejected() {
        let key = Key {
            key_id: 0,
            r: 5,
            d: 2,
            h0: vec![0, 0],
            h1: vec![1, 4],
        };
        assert!(key.validate().is_err());
    }

    #[test]
    fn sampled_keys_are_deterministic_and_unique() {
        let a = sample_keys_uniform(11, 3, 10, 7);
        let b = sample_keys_uniform(11, 3, 10, 7);
        assert_eq!(a.len(), 10);
        assert_eq!(a.len(), b.len());
        let supports_a = a.iter().map(key_to_support_string).collect::<Vec<_>>();
        let supports_b = b.iter().map(key_to_support_string).collect::<Vec<_>>();
        assert_eq!(supports_a, supports_b);
        let uniq = supports_a.iter().collect::<std::collections::HashSet<_>>();
        assert_eq!(uniq.len(), 10);
    }
}
