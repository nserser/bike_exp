# Experiment Coverage Against `spec/exp.pdf`

This note maps the current repo configuration set to the experiment plan in
[exp.pdf](/home/bkakria.anis/Téléchargements/bike_exp/spec/exp.pdf:1) and
records the places where configuration files are sufficient versus the places
where more code is still needed.

## Covered by Configs and Current Code

### Threshold calibration

Use:

- `configs/calibrate_toy_r13_d3_t3.toml`

Command:

```bash
docker run --rm \
  -v "$PWD/configs:/work/configs:ro" \
  -v "$PWD/artifacts:/work/artifacts" \
  dependency-aware-dfrcert:v01 \
  calibrate --config configs/calibrate_toy_r13_d3_t3.toml \
  --out artifacts/calibrate_toy_r13_d3_t3
```

### Experiment 1: exact toy full-domain certification

Available exact-toy configs:

- `configs/exact_toy_r11_d3_t2.toml`
- `configs/exact_toy_r11_d3_t3.toml`
- `configs/exact_toy_r13_d3_t2.toml`
- `configs/exact_toy_r13_d3_t3.toml`
- `configs/exact_toy_r17_d3_t3.toml`
- `configs/exact_toy_r19_d3_t3.toml`

Recommended practical anchor runs:

- `exact_toy_r13_d3_t2`
- `exact_toy_r13_d3_t3`

Representative command:

```bash
docker run --rm \
  -v "$PWD/configs:/work/configs:ro" \
  -v "$PWD/artifacts:/work/artifacts" \
  dependency-aware-dfrcert:v01 \
  run --config configs/exact_toy_r13_d3_t3.toml \
  --out artifacts/exact_toy_r13_d3_t3
```

### Experiments 2-5: profiles, direct profile-bin bounds, finite-state bounds, baseline models

These are emitted by the existing `run` pipeline for every exact-toy and
exact-sampled config above. The current code supports `M0`, `M1`, `M2`, and
`M3`, and the configs now include `bin_budgets = [8, 16, 32, 64, 128]` to help
with the state-compression sweep discussed later in `exp.pdf`.

### Experiment 6: exact sampled reduced-parameter experiment

The primary `exp.pdf` grid is now represented by explicit configs:

- `configs/exact_sampled_r31_d5_t3.toml`
- `configs/exact_sampled_r31_d5_t4.toml`
- `configs/exact_sampled_r31_d5_t5.toml`
- `configs/exact_sampled_r31_d7_t3.toml`
- `configs/exact_sampled_r31_d7_t4.toml`
- `configs/exact_sampled_r31_d7_t5.toml`
- `configs/exact_sampled_r47_d5_t3.toml`
- `configs/exact_sampled_r47_d5_t4.toml`
- `configs/exact_sampled_r47_d5_t5.toml`
- `configs/exact_sampled_r47_d7_t3.toml`
- `configs/exact_sampled_r47_d7_t4.toml`
- `configs/exact_sampled_r47_d7_t5.toml`

These use `mode = "stratified_keys"` so the selected keys are biased toward the
structural extremes described in `exp.pdf`, while still being deterministic
under the recorded seed.

Representative command:

```bash
docker run --rm \
  -v "$PWD/configs:/work/configs:ro" \
  -v "$PWD/artifacts:/work/artifacts" \
  dependency-aware-dfrcert:v01 \
  run --config configs/exact_sampled_r31_d5_t4.toml \
  --out artifacts/exact_sampled_r31_d5_t4
```

### Experiment 7: rare-event near-codeword conditioning

Use the same exact-sampled config as the source run and feed its artifact
directory back into `rare-event`. Example:

```bash
docker run --rm \
  -v "$PWD/configs:/work/configs:ro" \
  -v "$PWD/artifacts:/work/artifacts" \
  dependency-aware-dfrcert:v01 \
  rare-event --config configs/exact_sampled_r31_d5_t4.toml \
  --inputs artifacts/exact_sampled_r31_d5_t4 \
  --out artifacts/rare_event_r31_d5_t4
```

## Not Covered by Configs Alone

The following `exp.pdf` items still need implementation work, not just more
TOML files:

- Experiment 1 at `r = 17` and `r = 19` is present as config, but the paper
  explicitly allows cyclic-symmetry reduction when full enumeration is too
  large. The current code only supports full keyspace enumeration, so these runs
  are likely impractical without new symmetry-reduced enumeration logic.
- Experiment 8 held-out generalization:
  `train_fraction` exists in config, but the current code does not yet create
  train/test splits or emit `heldout_generalization.csv`.
- Experiment 9 confidence-bound larger reduced experiment:
  the current `run` path performs exact error enumeration per selected key and
  does not yet emit one-sided sampled-error confidence bounds such as
  `p_upper_confidence`.
- Experiment 5 ablation submodels `M3a`, `M3b`, and `M3c` are not yet separate
  model names in the implementation.
- Experiment 10 filter what-if report:
  no dedicated `filter_whatif.csv` writer exists yet.
- Experiment 11 dedicated state-compression summary report:
  the configs now expose the requested bin budgets, but no
  `state_compression_sensitivity.csv` aggregation command exists yet.
- Experiment 12 runtime and reproducibility report:
  there is no `runtime_report.csv` emitter yet.

## Bottom Line

After this config expansion, the repo has enough configuration coverage to run
the currently implemented parts of the `exp.pdf` matrix:

- threshold calibration,
- exact toy anchor runs,
- exact sampled reduced-parameter runs,
- profile extraction,
- direct-bin bounds,
- finite-state bounds,
- rare-event diagnostics.

It does **not** yet have enough implementation coverage to reproduce the full
paper-facing experiment list end to end.
