# Missing-size script update

This update addresses the issue where `scripts/run_missing_size_experiments.sh`
only produced threshold-sweep outputs when executed with no arguments.

## Changes

1. Default mode changed from `sweeps` to `main`.
2. Added explicit modes:
   - `sweeps`: threshold calibration only.
   - `main`: full same-key experiments for selected schedules.
   - `profiles`: large profile-only diagnostic.
   - `selected`: main + profiles.
   - `full`/`all`: sweeps + main + profiles.
3. Updated main-run threshold configs to the selected schedules from the sweep
   results:
   - `configs/m3_new_r61_d7_t2_k256.toml`: `[5,4,4,3]`
   - `configs/m3_new_r89_d9_t2_k128.toml`: `[5,5,4,3]`
   - `configs/m3_new_r89_d9_t3_k128_sampled.toml`: `[6,5,5,4,3]`
4. Updated `README.md` and `MISSING_SIZE_EXPERIMENTS.md` with the corrected
   execution workflow.

## Recommended command

Run the missing-size main experiments with:

```bash
./build.sh
./scripts/run_missing_size_experiments.sh main
```

Run calibration + main + profile diagnostic with:

```bash
./scripts/run_missing_size_experiments.sh full
```
