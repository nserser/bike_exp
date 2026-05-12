#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-dependency-aware-dfrcert:v01}"
CONFIG_DIR="${CONFIG_DIR:-$PWD/configs}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$PWD/artifacts}"

run_case() {
  local config_name="$1"
  local out_dir="$2"
  docker run --rm \
    -v "$CONFIG_DIR:/work/configs:ro" \
    -v "$ARTIFACT_DIR:/work/artifacts" \
    "$IMAGE_TAG" \
    diagnostics --config "configs/${config_name}" --out "artifacts/${out_dir}"
}

run_case "diagnostics_samekey_r31_d5_t1_m3hier.toml" "diagnostics_samekey_r31_d5_t1_m3hier"
run_case "diagnostics_samekey_r31_d5_t2_m3hier.toml" "diagnostics_samekey_r31_d5_t2_m3hier"
