# Dependency-Aware DFR Bounds Experiments

This repository contains the Dockerized Rust artifact for reduced-parameter
dependency-aware DFR experiments, including:

- exact same-key diagnostics
- hierarchical/profile-based certificate refinement
- the revised `M3_rank`, `M3_cert`, and `M3_counter` experiment flow from
  [`spec/new_m3_definition_and_experiment_plan.txt`](spec/new_m3_definition_and_experiment_plan.txt)

## Build

```bash
./build.sh
```

The default image tag is `dependency-aware-dfrcert:v01`.

## Core Commands

`m3-experiment` runs one exact or sampled same-key M3 experiment and writes:

- `per_key_profiles.csv`
- `per_key_m3.csv`
- `model_ranking_summary.csv`
- `certificate_bound_summary.csv`
- `static_key_bounds.csv`
- `bad_key_recall_summary.csv`
- `bin_summary.csv`
- `worst_key_mechanisms.csv`
- `failure_taxonomy.csv`
- `run_metadata.json`
- `experiment_summary.md`
- `manifest.json`

`m3-threshold-sweep` runs a small sanity/threshold sweep and writes:

- `threshold_sweep.csv`
- `threshold_sweep_summary.md`
- `manifest.json`

## Toy Validation

Run the low-memory toy validation for the new M3 definition:

```bash
./scripts/run_m3_new_toy_validation.sh
```

Equivalent Docker command:

```bash
docker run --rm \
  -v "$PWD/configs:/work/configs:ro" \
  -v "$PWD/artifacts:/work/artifacts" \
  dependency-aware-dfrcert:v01 \
  m3-experiment --config configs/m3_new_toy_r31_d5_t1_k32.toml \
  --out artifacts/m3_new_toy_r31_d5_t1_k32
```

Results will be in:

```text
artifacts/m3_new_toy_r31_d5_t1_k32/
```

The quick summary is:

```text
artifacts/m3_new_toy_r31_d5_t1_k32/experiment_summary.md
```

## New M3 Paper-Facing Runs

Experiment A:

```bash
./scripts/run_m3_experiment_a.sh
```

Equivalent:

```bash
docker run --rm \
  -v "$PWD/configs:/work/configs:ro" \
  -v "$PWD/artifacts:/work/artifacts" \
  dependency-aware-dfrcert:v01 \
  m3-experiment --config configs/m3_new_r31_d5_t1_k512.toml \
  --out artifacts/m3_new_r31_d5_t1_k512
```

Results:

```text
artifacts/m3_new_r31_d5_t1_k512/
```

Experiment B:

```bash
./scripts/run_m3_experiment_b.sh
```

Equivalent:

```bash
docker run --rm \
  -v "$PWD/configs:/work/configs:ro" \
  -v "$PWD/artifacts:/work/artifacts" \
  dependency-aware-dfrcert:v01 \
  m3-experiment --config configs/m3_new_r31_d5_t2_k512.toml \
  --out artifacts/m3_new_r31_d5_t2_k512
```

Results:

```text
artifacts/m3_new_r31_d5_t2_k512/
```

Threshold sweep for Experiment C:

```bash
./scripts/run_m3_sweep_r47_d5_t2.sh
```

Equivalent:

```bash
docker run --rm \
  -v "$PWD/configs:/work/configs:ro" \
  -v "$PWD/artifacts:/work/artifacts" \
  dependency-aware-dfrcert:v01 \
  m3-threshold-sweep --config configs/m3_sweep_r47_d5_t2_k64.toml \
  --out artifacts/m3_sweep_r47_d5_t2_k64
```

Sweep results:

```text
artifacts/m3_sweep_r47_d5_t2_k64/
```

Main Experiment C after schedule selection:

```bash
./scripts/run_m3_experiment_c.sh
```

Equivalent:

```bash
docker run --rm \
  -v "$PWD/configs:/work/configs:ro" \
  -v "$PWD/artifacts:/work/artifacts" \
  dependency-aware-dfrcert:v01 \
  m3-experiment --config configs/m3_new_r47_d5_t2_k256.toml \
  --out artifacts/m3_new_r47_d5_t2_k256
```

Note:
- `configs/m3_new_r47_d5_t2_k256.toml` ships with `thresholds = [4, 4, 3, 3]`.
- If the sweep selects another schedule, update that config before running Experiment C.

Threshold sweep for Experiment D:

```bash
./scripts/run_m3_sweep_r31_d5_t3.sh
```

Experiment D:

```bash
./scripts/run_m3_experiment_d.sh
```

Threshold sweep for optional Experiment E:

```bash
./scripts/run_m3_sweep_r47_d7_t2.sh
```

Optional Experiment E:

```bash
./scripts/run_m3_experiment_e.sh
```

## Memory Notes

The new `m3-experiment` path is designed to minimize memory use for the large
exact runs:

- per-error `failure_taxonomy.csv` rows are streamed directly to disk
- only compact per-key aggregates are kept in memory
- no giant in-memory decode-event table is retained
- exact FP/FN counter proxies are computed on the fly during the decode loop

If you still want a more conservative runtime profile, run one experiment at a
time and optionally limit visible Rayon threads:

```bash
RAYON_NUM_THREADS=4 ./scripts/run_m3_experiment_b.sh
```

## Key Result Locations

For any completed `m3-experiment`, start with:

- `experiment_summary.md`
- `model_ranking_summary.csv`
- `certificate_bound_summary.csv`
- `static_key_bounds.csv`
- `bad_key_recall_summary.csv`
- `worst_key_mechanisms.csv`

For any completed `m3-threshold-sweep`, start with:

- `threshold_sweep_summary.md`
- `threshold_sweep.csv`

## Missing Medium-Size and Large Diagnostic Experiments

The following scripts extend the paper-facing runs beyond the already completed
`r=31` and `r=47` exact experiments. They are intended to support the paper's
three-layer experimental structure:

1. exact reduced certification;
2. medium-size exact scaling;
3. large profile-only diagnostics.

The wrapper now defaults to the selected main same-key experiments:

```bash
./scripts/run_missing_size_experiments.sh
# equivalent:
./scripts/run_missing_size_experiments.sh main
```

This executes the full model comparisons using the sweep-selected schedules:

```text
r=61, d=7, t=2, thresholds=5;4;4;3, exact errors, 256 keys
r=89, d=9, t=2, thresholds=5;5;4;3, exact errors, 128 keys
r=89, d=9, t=3, thresholds=6;5;5;4;3, sampled errors, 128 keys
```

Calibration sweeps are still available, but they are no longer the default:

```bash
./scripts/run_missing_size_experiments.sh sweeps
```

Inspect sweep outputs at:

```text
artifacts/m3_sweep_r61_d7_t2_k64/threshold_sweep_summary.md
artifacts/m3_sweep_r89_d9_t2_k64/threshold_sweep_summary.md
artifacts/m3_sweep_r89_d9_t3_k32/threshold_sweep_summary.md
```

For a profile-only larger diagnostic that avoids DFR claims, run:

```bash
./scripts/run_missing_size_experiments.sh profiles
```

To run both selected main experiments and the profile-only diagnostic:

```bash
./scripts/run_missing_size_experiments.sh selected
```

To run sweeps, main experiments, and the profile-only diagnostic in one call:

```bash
./scripts/run_missing_size_experiments.sh full
# alias:
./scripts/run_missing_size_experiments.sh all
```

Exact/sample sizes:

```text
C(122,2) = 7381 errors/key for r=61,t=2
C(178,2) = 15753 errors/key for r=89,t=2
C(178,3) = 923176 possible errors/key for r=89,t=3, sampled to 10000/key
```

The profile-only diagnostic executes:

```text
r=127, d=11, t=2, 512 sampled keys, profile computation only
```

The profile-only diagnostic writes key/profile/bin CSVs without estimating
`p_H`. It is intended only to inspect whether high-risk structural coordinates
scale; it must not be presented as a production-parameter DFR proof.
