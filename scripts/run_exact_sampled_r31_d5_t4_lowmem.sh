#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_TAG="${IMAGE_TAG:-dependency-aware-dfrcert:v01}"
CERTIFY_THREADS="${DFRCERT_CERTIFY_THREADS:-2}"
INTRA_KEY_PROGRESS="${DFRCERT_INTRA_KEY_PROGRESS:-1}"
RAYON_THREADS="${RAYON_NUM_THREADS:-2}"

run_one() {
  local model="$1"
  local run_id="exact_sampled_r31_d5_t4_${model}"
  local container_name="dfrcert_${run_id}"

  if docker ps -a --format '{{.Names}}' | grep -Fxq "${container_name}"; then
    echo "Container ${container_name} already exists. Remove it first with:"
    echo "  docker rm -f ${container_name}"
    return 1
  fi

  echo "==> launching ${run_id}"
  docker run -d \
    --name "${container_name}" \
    -e "DFRCERT_CERTIFY_THREADS=${CERTIFY_THREADS}" \
    -e "DFRCERT_INTRA_KEY_PROGRESS=${INTRA_KEY_PROGRESS}" \
    -e "RAYON_NUM_THREADS=${RAYON_THREADS}" \
    -v "${ROOT_DIR}/configs:/work/configs:ro" \
    -v "${ROOT_DIR}/artifacts:/work/artifacts" \
    "${IMAGE_TAG}" \
    run --config "configs/${run_id}.toml" \
    --out "artifacts/${run_id}" >/dev/null

  docker logs -f "${container_name}"
  local exit_code
  exit_code="$(docker wait "${container_name}")"
  if [[ "${exit_code}" != "0" ]]; then
    echo "Container ${container_name} exited with status ${exit_code}" >&2
    return 1
  fi
}

for model in m0 m1 m2 m3; do
  run_one "${model}"
done

echo
echo "Completed sequential low-memory sampled runs:"
echo "  exact_sampled_r31_d5_t4_m0"
echo "  exact_sampled_r31_d5_t4_m1"
echo "  exact_sampled_r31_d5_t4_m2"
echo "  exact_sampled_r31_d5_t4_m3"
echo
echo "Runner defaults:"
echo "  RAYON_NUM_THREADS=${RAYON_THREADS}"
echo "  DFRCERT_CERTIFY_THREADS=${CERTIFY_THREADS}"
echo "  DFRCERT_INTRA_KEY_PROGRESS=${INTRA_KEY_PROGRESS}"
