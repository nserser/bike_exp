# Missing Size Experiment Scripts

This artifact adds scripts and configs for the missing medium-size and larger
structural diagnostics discussed in the paper-review process.

## Important update

The script now defaults to the full **main same-key experiments**, not to sweeps.
The previous default ran only threshold sweeps, which explained why the first
execution produced calibration outputs but no full M0/M1/M2/M3 comparison files.

The selected schedules from the completed missing-size sweeps are now encoded in
the main configs:

| Regime | Main config | Selected thresholds | Role |
| --- | --- | --- | --- |
| `r=61,d=7,t=2` | `configs/m3_new_r61_d7_t2_k256.toml` | `5;4;4;3` | medium exact tail-sensitive run |
| `r=89,d=9,t=2` | `configs/m3_new_r89_d9_t2_k128.toml` | `5;5;4;3` | larger exact run |
| `r=89,d=9,t=3` | `configs/m3_new_r89_d9_t3_k128_sampled.toml` | `6;5;5;4;3` | sampled stress run |

## Modes

Run the selected main experiments:

```bash
./scripts/run_missing_size_experiments.sh
# equivalent:
./scripts/run_missing_size_experiments.sh main
```

This executes:

```text
r=61, d=7, t=2, thresholds=5;4;4;3, exact errors, 256 keys
r=89, d=9, t=2, thresholds=5;5;4;3, exact errors, 128 keys
r=89, d=9, t=3, thresholds=6;5;5;4;3, sampled errors, 128 keys
```

Run only the threshold calibration sweeps:

```bash
./scripts/run_missing_size_experiments.sh sweeps
```

Run only the large profile-only diagnostic:

```bash
./scripts/run_missing_size_experiments.sh profiles
```

Run main experiments plus the profile-only diagnostic:

```bash
./scripts/run_missing_size_experiments.sh selected
```

Run sweeps, main experiments, and the profile-only diagnostic in one call:

```bash
./scripts/run_missing_size_experiments.sh full
# alias:
./scripts/run_missing_size_experiments.sh all
```

## Added sweeps

- `configs/m3_sweep_r61_d7_t2_k64.toml`
- `configs/m3_sweep_r89_d9_t2_k64.toml`
- `configs/m3_sweep_r89_d9_t3_k32.toml`

The sweeps select non-saturated threshold schedules using the existing preferred
mean range. They are now calibration scripts, not the default experiment path.

## Added medium-size main experiments

- `configs/m3_new_r61_d7_t2_k256.toml`
- `configs/m3_new_r89_d9_t2_k128.toml`
- `configs/m3_new_r89_d9_t3_k128_sampled.toml`

The first two runs use exact error enumeration for `t=2`; the `t=3` run samples
10000 errors per key because the exact space has 923176 errors per key.

## Added large profile-only diagnostic

- `configs/large_profile_r127_d11_t2_k512.toml`
- `scripts/run_large_profile_r127_d11.sh`

This uses the existing `profiles` command and does not estimate `p_H`. It is for
structural-coordinate scaling only.

## Paper interpretation

Use medium exact experiments to strengthen the reduced/medium-parameter evidence
for `M3_cert` and `M3_counter`.

Use the profile-only diagnostic only to support conservative scaling discussion:
structural features can be sampled at larger sizes, but these runs do not certify
production DFR.
