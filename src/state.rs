use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::{json, Value};

use crate::bitset::{Mask256, SyndromeMask};
use crate::io::{hex_sha256, StateSection};
use crate::key::KeyRuntime;

#[derive(Clone, Debug)]
pub struct StateDescriptor {
    pub model: String,
    pub state_id: u64,
    pub state_tuple: Value,
}

#[derive(Clone, Debug, Serialize)]
pub struct MlCount {
    pub m: u16,
    pub l: u16,
    pub count: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct SuccessorCount {
    pub state_id: u64,
    pub count: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct MlSuccessorCount {
    pub m: u16,
    pub l: u16,
    pub successor_counts: Vec<SuccessorCount>,
}

#[derive(Clone, Debug, Serialize)]
pub struct StateCountRecord {
    pub run_id: String,
    pub model: String,
    pub key_id: u64,
    pub iteration: usize,
    pub state_id: u64,
    pub state_tuple: Value,
    pub count: u64,
    pub ml_counts: Vec<MlCount>,
    pub successor_counts: Vec<SuccessorCount>,
    pub ml_successor_counts: Vec<MlSuccessorCount>,
}

#[derive(Clone, Debug, Default)]
pub struct StateAggregation {
    pub records: Vec<StateCountRecord>,
}

#[derive(Clone, Debug)]
pub struct AggregationEvent {
    pub iteration: usize,
    pub descriptor: StateDescriptor,
    pub missed_count: u16,
    pub false_count: u16,
    pub successor_state_id: u64,
}

pub fn compute_state_descriptor(
    model: &str,
    key_rt: &KeyRuntime,
    residual: Mask256,
    syndrome: SyndromeMask,
    state_config: &StateSection,
    threshold: usize,
) -> StateDescriptor {
    if residual.is_zero() {
        return terminal_state(model, "succ");
    }
    if syndrome.is_zero() {
        return terminal_state(model, "bad");
    }

    let residual_weight = residual.popcount() as usize;
    let syndrome_weight = syndrome.popcount() as usize;
    let u_near = near_codeword_overlap(key_rt, residual);
    let fixed = usize::from(is_fixed_point_candidate(key_rt, syndrome, threshold));
    let omega = omega_value(key_rt, residual);
    let u_bin = bin_index(u_near, &state_config.u_bins);
    let omega_bin = bin_index(omega as usize, &state_config.omega_bins);

    let state_tuple = match model {
        "M1" => json!({
            "terminal": "state",
            "tw": residual_weight,
            "sw": syndrome_weight,
        }),
        "M2" => json!({
            "terminal": "state",
            "tw": residual_weight,
            "sw": syndrome_weight,
            "u_bin": u_bin,
        }),
        "M3" => {
            if state_config.strict_absorbing && fixed == 1 {
                return terminal_state(model, "bad");
            }
            json!({
                "terminal": "state",
                "tw": residual_weight,
                "sw": syndrome_weight,
                "u_bin": u_bin,
                "fixed": fixed,
                "omega_bin": omega_bin,
            })
        }
        _ => json!({
            "terminal": "state",
            "tw": residual_weight,
            "sw": syndrome_weight,
        }),
    };

    StateDescriptor {
        model: model.to_string(),
        state_id: state_id_for(model, &state_tuple),
        state_tuple,
    }
}

pub fn terminal_state(model: &str, terminal: &str) -> StateDescriptor {
    let state_tuple = json!({ "terminal": terminal });
    StateDescriptor {
        model: model.to_string(),
        state_id: state_id_for(model, &state_tuple),
        state_tuple,
    }
}

pub fn aggregate_state_events(
    run_id: &str,
    key_id: u64,
    events_by_model: &BTreeMap<String, Vec<AggregationEvent>>,
) -> StateAggregation {
    let mut records = Vec::new();

    for (model, events) in events_by_model {
        let mut grouped: BTreeMap<(usize, u64), GroupedStateRecord> = BTreeMap::new();
        for event in events {
            let entry = grouped
                .entry((event.iteration, event.descriptor.state_id))
                .or_insert_with(|| GroupedStateRecord {
                    state_tuple: event.descriptor.state_tuple.clone(),
                    count: 0,
                    ml_counts: BTreeMap::new(),
                    successor_counts: BTreeMap::new(),
                    ml_successor_counts: BTreeMap::new(),
                });
            entry.count += 1;
            *entry
                .ml_counts
                .entry((event.missed_count, event.false_count))
                .or_insert(0) += 1;
            *entry
                .successor_counts
                .entry(event.successor_state_id)
                .or_insert(0) += 1;
            *entry
                .ml_successor_counts
                .entry((event.missed_count, event.false_count, event.successor_state_id))
                .or_insert(0) += 1;
        }

        for ((iteration, state_id), grouped_record) in grouped {
            let mut ml_successor_grouped: BTreeMap<(u16, u16), Vec<SuccessorCount>> = BTreeMap::new();
            for ((m, l, successor_state_id), count) in grouped_record.ml_successor_counts {
                ml_successor_grouped
                    .entry((m, l))
                    .or_default()
                    .push(SuccessorCount {
                        state_id: successor_state_id,
                        count,
                    });
            }
            records.push(StateCountRecord {
                run_id: run_id.to_string(),
                model: model.clone(),
                key_id,
                iteration,
                state_id,
                state_tuple: grouped_record.state_tuple,
                count: grouped_record.count,
                ml_counts: grouped_record
                    .ml_counts
                    .into_iter()
                    .map(|((m, l), count)| MlCount { m, l, count })
                    .collect(),
                successor_counts: grouped_record
                    .successor_counts
                    .into_iter()
                    .map(|(state_id, count)| SuccessorCount { state_id, count })
                    .collect(),
                ml_successor_counts: ml_successor_grouped
                    .into_iter()
                    .map(|((m, l), successor_counts)| MlSuccessorCount {
                        m,
                        l,
                        successor_counts,
                    })
                    .collect(),
            });
        }
    }

    StateAggregation { records }
}

fn state_id_for(model: &str, state_tuple: &Value) -> u64 {
    let payload = serde_json::to_vec(&json!({
        "model": model,
        "state": state_tuple,
    }))
    .unwrap_or_default();
    let digest = hex_sha256(&payload);
    u64::from_str_radix(&digest[..16], 16).unwrap_or(0)
}

fn bin_index(value: usize, bins: &[usize]) -> usize {
    bins.iter().position(|&bound| value <= bound).unwrap_or(bins.len())
}

fn near_codeword_overlap(key_rt: &KeyRuntime, residual: Mask256) -> usize {
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

fn is_fixed_point_candidate(key_rt: &KeyRuntime, syndrome: SyndromeMask, threshold: usize) -> bool {
    (0..2 * key_rt.key.r)
        .all(|pos| (key_rt.column_masks[pos].and(syndrome).popcount() as usize) < threshold)
}

fn omega_value(key_rt: &KeyRuntime, residual: Mask256) -> u32 {
    let positions = residual.iter_ones(2 * key_rt.key.r).collect::<Vec<_>>();
    let mut best = 0u32;
    for &j in &positions {
        let mut score = 0u32;
        for &k in &positions {
            if j == k {
                continue;
            }
            let overlap = key_rt.column_masks[j].and(key_rt.column_masks[k]).popcount();
            if overlap >= 2 {
                score += overlap;
            }
        }
        best = best.max(score);
    }
    best
}

struct GroupedStateRecord {
    state_tuple: Value,
    count: u64,
    ml_counts: BTreeMap<(u16, u16), u64>,
    successor_counts: BTreeMap<u64, u64>,
    ml_successor_counts: BTreeMap<(u16, u16, u64), u64>,
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::bitset::Mask256;
    use crate::io::StateSection;
    use crate::key::{build_key_runtime, Key};

    use super::{aggregate_state_events, compute_state_descriptor, terminal_state, AggregationEvent};

    fn test_state_config() -> StateSection {
        StateSection {
            models: vec!["M1".to_string(), "M2".to_string(), "M3".to_string()],
            u_bins: vec![0, 1, 2, 99],
            omega_bins: vec![0, 1, 2, 4, 99],
            strict_absorbing: false,
        }
    }

    #[test]
    fn state_descriptor_builds_terminal_and_nonterminal_states() {
        let key = Key {
            key_id: 0,
            r: 5,
            d: 2,
            h0: vec![0, 1],
            h1: vec![0, 2],
        };
        let runtime = build_key_runtime(key).unwrap();
        let mut residual = Mask256::empty();
        residual.set(0).unwrap();
        let syndrome = runtime.column_masks[0];
        let state = compute_state_descriptor("M2", &runtime, residual, syndrome, &test_state_config(), 2);
        assert_eq!(state.model, "M2");
        assert!(state.state_id > 0);

        let succ = terminal_state("M1", "succ");
        assert_eq!(succ.state_tuple["terminal"], "succ");
    }

    #[test]
    fn state_aggregation_groups_counts() {
        let mut model_events = BTreeMap::new();
        model_events.insert(
            "M1".to_string(),
            vec![
                AggregationEvent {
                    iteration: 0,
                    descriptor: terminal_state("M1", "bad"),
                    missed_count: 1,
                    false_count: 0,
                    successor_state_id: 7,
                },
                AggregationEvent {
                    iteration: 0,
                    descriptor: terminal_state("M1", "bad"),
                    missed_count: 1,
                    false_count: 0,
                    successor_state_id: 7,
                },
            ],
        );
        let aggregated = aggregate_state_events("run", 3, &model_events);
        assert_eq!(aggregated.records.len(), 1);
        assert_eq!(aggregated.records[0].count, 2);
        assert_eq!(aggregated.records[0].ml_counts[0].count, 2);
        assert_eq!(aggregated.records[0].ml_successor_counts[0].successor_counts[0].count, 2);
    }
}
