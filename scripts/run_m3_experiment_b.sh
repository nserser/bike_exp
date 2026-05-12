#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-dependency-aware-dfrcert:v01}"

docker run --rm \
  -v "$PWD/configs:/work/configs:ro" \
  -v "$PWD/artifacts:/work/artifacts" \
  "$IMAGE_TAG" \
  m3-experiment --config configs/m3_new_r31_d5_t2_k512.toml \
  --out artifacts/m3_new_r31_d5_t2_k512
