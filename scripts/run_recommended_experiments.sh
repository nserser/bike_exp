#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_TAG="${IMAGE_TAG:-dependency-aware-dfrcert:v01}"
DOCKER_BIN="${DOCKER_BIN:-docker}"

run_step() {
  local command_name="$1"
  local config_name="$2"
  local out_dir="$3"
  local inputs_dir="${4:-}"

  echo
  echo "==> ${command_name} :: ${config_name}"

  if [[ -n "${inputs_dir}" ]]; then
    "${DOCKER_BIN}" run --rm \
      -v "${ROOT_DIR}/configs:/work/configs:ro" \
      -v "${ROOT_DIR}/artifacts:/work/artifacts" \
      "${IMAGE_TAG}" \
      "${command_name}" \
      --config "configs/${config_name}.toml" \
      --inputs "artifacts/${inputs_dir}" \
      --out "artifacts/${out_dir}"
  else
    "${DOCKER_BIN}" run --rm \
      -v "${ROOT_DIR}/configs:/work/configs:ro" \
      -v "${ROOT_DIR}/artifacts:/work/artifacts" \
      "${IMAGE_TAG}" \
      "${command_name}" \
      --config "configs/${config_name}.toml" \
      --out "artifacts/${out_dir}"
  fi
}

main() {
  mkdir -p "${ROOT_DIR}/artifacts"

  run_step calibrate "calibrate_toy_r13_d3_t3" "calibrate_toy_r13_d3_t3"
  run_step run "exact_toy_r13_d3_t2" "exact_toy_r13_d3_t2"
  run_step run "exact_toy_r13_d3_t3" "exact_toy_r13_d3_t3"
  run_step run "exact_sampled_r31_d5_t4" "exact_sampled_r31_d5_t4"
  run_step rare-event "exact_sampled_r31_d5_t4" "rare_event_r31_d5_t4" "exact_sampled_r31_d5_t4"

  echo
  echo "Recommended experiment sequence completed."
}

main "$@"
