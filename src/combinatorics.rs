use anyhow::{bail, Result};

pub fn binom_u128(n: u32, k: u32) -> u128 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    if k == 0 {
        return 1;
    }

    let mut numerators: Vec<u128> = ((n - k + 1)..=n).map(u128::from).collect();
    for divisor in 2..=u128::from(k) {
        let mut carry = divisor;
        for numerator in &mut numerators {
            let g = gcd_u128(*numerator, carry);
            if g > 1 {
                *numerator /= g;
                carry /= g;
            }
            if carry == 1 {
                break;
            }
        }
        assert_eq!(carry, 1, "failed to cancel denominator in binomial coefficient");
    }

    numerators.into_iter().fold(1u128, |acc, value| {
        acc.checked_mul(value)
            .expect("binomial coefficient exceeds u128 capacity")
    })
}

#[derive(Clone, Debug)]
pub struct CombinationIterator {
    n: usize,
    state: Option<Vec<usize>>,
}

impl CombinationIterator {
    pub fn new(n: usize, k: usize) -> Result<Self> {
        if k > n {
            bail!("k must not exceed n");
        }
        Ok(Self {
            n,
            state: Some((0..k).collect()),
        })
    }
}

impl Iterator for CombinationIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.state.clone()?;
        self.state = lexicographic_successor(&current, self.n);
        Some(current)
    }
}

pub fn iter_combinations(n: usize, k: usize) -> Result<CombinationIterator> {
    CombinationIterator::new(n, k)
}

pub fn lexicographic_successor(current: &[usize], n: usize) -> Option<Vec<usize>> {
    let k = current.len();
    if k == 0 {
        return None;
    }

    let mut next = current.to_vec();
    for idx in (0..k).rev() {
        let max_value = idx + n - k;
        if next[idx] < max_value {
            next[idx] += 1;
            for j in idx + 1..k {
                next[j] = next[j - 1] + 1;
            }
            return Some(next);
        }
    }
    None
}

pub fn hypergeom_tail(n: usize, marked: usize, draws: usize, at_least: usize) -> f64 {
    if marked > n || draws > n {
        return 0.0;
    }
    let min_successes = at_least.max(draws.saturating_sub(n - marked));
    let max_successes = marked.min(draws);
    if min_successes > max_successes {
        return 0.0;
    }

    let total_log = ln_binom(n, draws);
    let mut tail = 0.0;
    for x in min_successes..=max_successes {
        let log_p = ln_binom(marked, x) + ln_binom(n - marked, draws - x) - total_log;
        tail += log_p.exp();
    }
    tail.clamp(0.0, 1.0)
}

fn gcd_u128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn ln_binom(n: usize, k: usize) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    let k = k.min(n - k);
    if k == 0 {
        return 0.0;
    }
    let start = n - k + 1;
    let log_num: f64 = (start..=n).map(|v| (v as f64).ln()).sum();
    let log_den: f64 = (1..=k).map(|v| (v as f64).ln()).sum();
    log_num - log_den
}

#[cfg(test)]
mod tests {
    use super::{binom_u128, hypergeom_tail, iter_combinations, lexicographic_successor};

    #[test]
    fn binom_matches_known_values() {
        assert_eq!(binom_u128(5, 2), 10);
        assert_eq!(binom_u128(22, 3), 1540);
        assert_eq!(binom_u128(26, 2), 325);
        assert_eq!(binom_u128(0, 0), 1);
        assert_eq!(binom_u128(3, 4), 0);
    }

    #[test]
    fn lexicographic_successor_advances_correctly() {
        assert_eq!(lexicographic_successor(&[0, 1, 2], 5), Some(vec![0, 1, 3]));
        assert_eq!(lexicographic_successor(&[0, 3, 4], 5), Some(vec![1, 2, 3]));
        assert_eq!(lexicographic_successor(&[2, 3, 4], 5), None);
    }

    #[test]
    fn iter_combinations_returns_lexicographic_order() {
        let combos = iter_combinations(5, 3).unwrap().collect::<Vec<_>>();
        assert_eq!(
            combos,
            vec![
                vec![0, 1, 2],
                vec![0, 1, 3],
                vec![0, 1, 4],
                vec![0, 2, 3],
                vec![0, 2, 4],
                vec![0, 3, 4],
                vec![1, 2, 3],
                vec![1, 2, 4],
                vec![1, 3, 4],
                vec![2, 3, 4],
            ]
        );
    }

    #[test]
    fn iter_combinations_handles_zero_subset() {
        let combos = iter_combinations(4, 0).unwrap().collect::<Vec<_>>();
        assert_eq!(combos, vec![Vec::<usize>::new()]);
    }

    #[test]
    fn hypergeom_tail_matches_small_exact_value() {
        let tail = hypergeom_tail(10, 4, 3, 2);
        let expected = 1.0 / 3.0;
        assert!((tail - expected).abs() < 1e-12);
    }

    #[test]
    fn hypergeom_tail_rejects_impossible_region() {
        assert_eq!(hypergeom_tail(10, 3, 2, 4), 0.0);
        assert_eq!(hypergeom_tail(5, 6, 2, 1), 0.0);
    }
}
