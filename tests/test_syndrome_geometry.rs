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
use decoder::syndrome_for_error;
use key::{build_key_runtime, Key};

#[test]
fn single_bit_true_counter_equals_d_for_every_position() {
    let key = Key {
        key_id: 0,
        r: 13,
        d: 3,
        h0: vec![0, 4, 9],
        h1: vec![1, 6, 10],
    };
    let runtime = build_key_runtime(key).unwrap();
    for bit_index in 0..26 {
        let mut error = Mask256::empty();
        error.set(bit_index).unwrap();
        let syndrome = syndrome_for_error(&runtime, error).unwrap();
        let counter = runtime.column_masks[bit_index].and(syndrome).popcount() as usize;
        assert_eq!(counter, 3, "bit_index={bit_index}");
        assert_eq!(syndrome.popcount(), 3, "bit_index={bit_index}");
    }
}

#[test]
fn rotated_columns_match_single_bit_syndromes() {
    let key = Key {
        key_id: 1,
        r: 13,
        d: 3,
        h0: vec![0, 2, 11],
        h1: vec![1, 5, 9],
    };
    let runtime = build_key_runtime(key).unwrap();
    for bit_index in 0..26 {
        let mut error = Mask256::empty();
        error.set(bit_index).unwrap();
        let syndrome = syndrome_for_error(&runtime, error).unwrap();
        assert_eq!(runtime.column_masks[bit_index], syndrome, "bit_index={bit_index}");
    }
}
