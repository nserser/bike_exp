use anyhow::{bail, Result};

use crate::bitset::{Mask256, SyndromeMask};
use crate::io::DecoderSection;
use crate::key::KeyRuntime;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Success,
    Timeout,
    WrongZero,
    NonzeroHalt,
}

#[derive(Clone, Debug)]
pub struct IterationLog {
    pub i: usize,
    pub e_mask: Mask256,
    pub syndrome: SyndromeMask,
    pub residual_weight: u16,
    pub syndrome_weight: u16,
    pub flip_mask: Mask256,
    pub flip_count: u16,
    pub missed_mask: Mask256,
    pub false_mask: Mask256,
    pub missed_count: u16,
    pub false_count: u16,
    pub state_id: u64,
}

#[derive(Clone, Debug)]
pub struct DecodeResult {
    pub outcome: Outcome,
    pub iterations: u16,
    pub final_residual_weight: u16,
    pub final_syndrome_weight: u16,
    pub logs: Vec<IterationLog>,
}

pub fn decode_with_logs(
    key_rt: &KeyRuntime,
    error_mask: Mask256,
    decoder_config: &DecoderSection,
) -> Result<DecodeResult> {
    validate_decoder_config(decoder_config)?;

    let mut estimate = Mask256::empty();
    let mut residual = error_mask;
    let mut logs = Vec::with_capacity(decoder_config.iterations);

    if residual.is_zero() {
        return Ok(DecodeResult {
            outcome: Outcome::Success,
            iterations: 0,
            final_residual_weight: 0,
            final_syndrome_weight: 0,
            logs,
        });
    }

    for (i, threshold) in decoder_config.thresholds.iter().copied().enumerate() {
        let current_residual = residual;
        let syndrome = syndrome_for_error(key_rt, current_residual)?;
        let syndrome_weight = syndrome.popcount() as u16;
        let residual_weight = current_residual.popcount() as u16;

        let mut flip_mask = Mask256::empty();
        for pos in 0..key_rt.column_masks.len() {
            let count = key_rt.column_masks[pos].and(syndrome).popcount() as usize;
            if count >= threshold {
                flip_mask.set(pos)?;
            }
        }

        estimate.xor_assign(flip_mask);
        residual.xor_assign(flip_mask);

        let missed_mask = residual.intersection(error_mask);
        let false_mask = residual.intersection(estimate);
        let flip_count = flip_mask.popcount() as u16;
        let missed_count = missed_mask.popcount() as u16;
        let false_count = false_mask.popcount() as u16;

        logs.push(IterationLog {
            i,
            e_mask: current_residual,
            syndrome,
            residual_weight,
            syndrome_weight,
            flip_mask,
            flip_count,
            missed_mask,
            false_mask,
            missed_count,
            false_count,
            state_id: 0,
        });

        if residual.is_zero() {
            return Ok(DecodeResult {
                outcome: Outcome::Success,
                iterations: (i + 1) as u16,
                final_residual_weight: 0,
                final_syndrome_weight: 0,
                logs,
            });
        }

        let next_syndrome = syndrome_for_error(key_rt, residual)?;
        if next_syndrome.is_zero() {
            let outcome = if decoder_config.wrong_zero_is_failure {
                Outcome::WrongZero
            } else {
                Outcome::NonzeroHalt
            };
            return Ok(DecodeResult {
                outcome,
                iterations: (i + 1) as u16,
                final_residual_weight: residual.popcount() as u16,
                final_syndrome_weight: 0,
                logs,
            });
        }
    }

    let final_syndrome = syndrome_for_error(key_rt, residual)?;
    Ok(DecodeResult {
        outcome: Outcome::Timeout,
        iterations: decoder_config.iterations as u16,
        final_residual_weight: residual.popcount() as u16,
        final_syndrome_weight: final_syndrome.popcount() as u16,
        logs,
    })
}

pub fn decode_fast(
    key_rt: &KeyRuntime,
    error_mask: Mask256,
    decoder_config: &DecoderSection,
) -> Result<Outcome> {
    Ok(decode_with_logs(key_rt, error_mask, decoder_config)?.outcome)
}

pub fn syndrome_for_error(key_rt: &KeyRuntime, error_mask: Mask256) -> Result<SyndromeMask> {
    let mut syndrome = SyndromeMask::empty();
    for pos in error_mask.iter_ones(key_rt.column_masks.len()) {
        let Some(column_mask) = key_rt.column_masks.get(pos).copied() else {
            bail!("error position {pos} exceeds column mask count");
        };
        syndrome.xor_assign(column_mask);
    }
    Ok(syndrome)
}

fn validate_decoder_config(decoder_config: &DecoderSection) -> Result<()> {
    if decoder_config.thresholds.len() != decoder_config.iterations {
        bail!(
            "decoder thresholds length {} does not match iterations {}",
            decoder_config.thresholds.len(),
            decoder_config.iterations
        );
    }
    if decoder_config.flip_policy != "all_ge_threshold" {
        bail!("unsupported flip policy '{}'", decoder_config.flip_policy);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::key::build_key_runtime;
    use crate::key::Key;

    use super::{decode_with_logs, syndrome_for_error, Mask256, Outcome};
    use crate::io::DecoderSection;

    fn toy_decoder_config() -> DecoderSection {
        DecoderSection {
            iterations: 3,
            thresholds: vec![2, 2, 2],
            wrong_zero_is_failure: true,
            flip_policy: "all_ge_threshold".to_string(),
        }
    }

    #[test]
    fn syndrome_matches_manual_xor() {
        let key = Key {
            key_id: 0,
            r: 5,
            d: 2,
            h0: vec![0, 2],
            h1: vec![1, 4],
        };
        let runtime = build_key_runtime(key).unwrap();
        let mut error = Mask256::empty();
        error.set(0).unwrap();
        error.set(5).unwrap();

        let syndrome = syndrome_for_error(&runtime, error).unwrap();
        assert!(syndrome.contains(0));
        assert!(syndrome.contains(1));
        assert!(syndrome.contains(2));
        assert!(syndrome.contains(4));
        assert_eq!(syndrome.popcount(), 4);
    }

    #[test]
    fn decoder_succeeds_on_zero_error() {
        let key = Key {
            key_id: 0,
            r: 5,
            d: 2,
            h0: vec![0, 2],
            h1: vec![1, 4],
        };
        let runtime = build_key_runtime(key).unwrap();
        let result = decode_with_logs(&runtime, Mask256::empty(), &toy_decoder_config()).unwrap();
        assert_eq!(result.outcome, Outcome::Success);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn decoder_logs_consistent_weights() {
        let key = Key {
            key_id: 0,
            r: 5,
            d: 2,
            h0: vec![0, 1],
            h1: vec![0, 1],
        };
        let runtime = build_key_runtime(key).unwrap();
        let mut error = Mask256::empty();
        error.set(0).unwrap();
        error.set(5).unwrap();
        let result = decode_with_logs(&runtime, error, &toy_decoder_config()).unwrap();
        assert!(!result.logs.is_empty());
        for log in &result.logs {
            assert_eq!(log.residual_weight as usize, log.e_mask.popcount() as usize);
            assert_eq!(log.syndrome_weight, log.syndrome.popcount() as u16);
        }
    }
}
