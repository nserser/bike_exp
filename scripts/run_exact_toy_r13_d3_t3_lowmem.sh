#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_TAG="${IMAGE_TAG:-dependency-aware-dfrcert:v01}"

for model in m0 m1 m2 m3; do
  run_id="exact_toy_r13_d3_t3_${model}"
  echo "==> launching ${run_id}"
  docker run -d \
    --name "dfrcert_${run_id}" \
    -v "${ROOT_DIR}/configs:/work/configs:ro" \
    -v "${ROOT_DIR}/artifacts:/work/artifacts" \
    "${IMAGE_TAG}" \
    run --config "configs/${run_id}.toml" \
    --out "artifacts/${run_id}"
done

echo
echo "Started detached low-memory split runs:"
echo "  dfrcert_exact_toy_r13_d3_t3_m0"
echo "  dfrcert_exact_toy_r13_d3_t3_m1"
echo "  dfrcert_exact_toy_r13_d3_t3_m2"
echo "  dfrcert_exact_toy_r13_d3_t3_m3"
