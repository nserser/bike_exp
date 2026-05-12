#[path = "../src/bitset.rs"]
mod bitset;
#[path = "../src/combinatorics.rs"]
mod combinatorics;
#[path = "../src/key.rs"]
mod key;
#[path = "../src/io.rs"]
mod io;
#[path = "../src/decoder.rs"]
mod decoder;

use bitset::Mask256;
use decoder::{decode_with_logs, syndrome_for_error, Outcome};
use io::DecoderSection;
use key::{build_key_runtime, Key};

fn toy_decoder() -> DecoderSection {
    DecoderSection {
        iterations: 5,
        thresholds: vec![2, 2, 2, 2, 2],
        wrong_zero_is_failure: true,
        flip_policy: "all_ge_threshold".to_string(),
    }
}

#[test]
fn zero_error_always_succeeds_with_zero_final_syndrome() {
    let key = Key {
        key_id: 2,
        r: 13,
        d: 3,
        h0: vec![0, 3, 8],
        h1: vec![1, 4, 11],
    };
    let runtime = build_key_runtime(key).unwrap();
    let result = decode_with_logs(&runtime, Mask256::empty(), &toy_decoder()).unwrap();
    assert_eq!(result.outcome, Outcome::Success);
    assert_eq!(result.final_syndrome_weight, 0);
    assert_eq!(result.final_residual_weight, 0);
}

#[test]
fn logged_syndromes_match_recomputed_syndromes_every_iteration() {
    let key = Key {
        key_id: 3,
        r: 13,
        d: 3,
        h0: vec![0, 5, 9],
        h1: vec![2, 6, 10],
    };
    let runtime = build_key_runtime(key).unwrap();
    let mut error = Mask256::empty();
    error.set(0).unwrap();
    error.set(13).unwrap();
    error.set(18).unwrap();
    let result = decode_with_logs(&runtime, error, &toy_decoder()).unwrap();
    for log in &result.logs {
        let recomputed = syndrome_for_error(&runtime, log.e_mask).unwrap();
        assert_eq!(recomputed, log.syndrome, "iteration={}", log.i);
        assert_eq!(recomputed.popcount(), u32::from(log.syndrome_weight));
    }
}
