#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_TAG="${IMAGE_TAG:-dependency-aware-dfrcert:v01}"
CONFIG_PATH="${1:-configs/diagnostics_tiny.toml}"
OUT_DIR="${2:-artifacts/diagnostics}"

docker run --rm \
  -v "${ROOT_DIR}/configs:/work/configs:ro" \
  -v "${ROOT_DIR}/artifacts:/work/artifacts" \
  "${IMAGE_TAG}" \
  diagnostics --config "${CONFIG_PATH}" --out "${OUT_DIR}"
