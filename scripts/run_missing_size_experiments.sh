#!/usr/bin/env bash
set -euo pipefail

# Runs the missing medium-size and large diagnostic experiments.
#
# Default mode is `main`: it runs the full same-key experiments using the
# sweep-selected schedules that are already recorded in the m3_new_* configs.
# Use `sweeps` only when recalibrating thresholds. Use `profiles` for the
# profile-only larger diagnostic. Use `full`/`all` to run sweeps, main, and
# profiles in one invocation.
#
# Selected schedules currently encoded in configs:
#   r=61, d=7, t=2: thresholds = 5;4;4;3
#   r=89, d=9, t=2: thresholds = 5;5;4;3
#   r=89, d=9, t=3: thresholds = 6;5;5;4;3

MODE="${1:-main}"

print_plan() {
  cat <<'PLAN'
Missing-size experiment modes:
  sweeps     Re-run threshold calibration sweeps only.
  main       Run full same-key M0/M1/M2/M3_* experiments for selected schedules.
  profiles   Run the large profile-only diagnostic r=127,d=11,t=2.
  selected   Run main + profiles.
  full/all   Run sweeps + main + profiles.

Selected main experiments:
  r=61,d=7,t=2, thresholds=5;4;4;3, 256 keys, exact errors.
  r=89,d=9,t=2, thresholds=5;5;4;3, 128 keys, exact errors.
  r=89,d=9,t=3, thresholds=6;5;5;4;3, 128 keys, sampled errors.
PLAN
}

run_sweeps() {
  ./scripts/run_m3_sweep_r61_d7_t2.sh
  ./scripts/run_m3_sweep_r89_d9_t2.sh
  ./scripts/run_m3_sweep_r89_d9_t3.sh
}

run_main() {
  ./scripts/run_m3_experiment_f_r61_d7_t2.sh
  ./scripts/run_m3_experiment_g_r89_d9_t2.sh
  ./scripts/run_m3_experiment_h_r89_d9_t3_sampled.sh
}

run_profiles() {
  ./scripts/run_large_profile_r127_d11.sh
}

case "$MODE" in
  help|-h|--help)
    print_plan
    ;;
  sweeps)
    print_plan
    run_sweeps
    ;;
  main)
    print_plan
    run_main
    ;;
  profiles)
    print_plan
    run_profiles
    ;;
  selected)
    print_plan
    run_main
    run_profiles
    ;;
  full|all)
    print_plan
    run_sweeps
    run_main
    run_profiles
    ;;
  *)
    print_plan >&2
    echo "Unknown mode: $MODE" >&2
    exit 2
    ;;
esac
