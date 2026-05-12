#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-dependency-aware-dfrcert:v01}"

docker run --rm \
  -v "$PWD/configs:/work/configs:ro" \
  -v "$PWD/artifacts:/work/artifacts" \
  "$IMAGE_TAG" \
  m3-threshold-sweep --config configs/m3_sweep_r89_d9_t3_k32.toml \
  --out artifacts/m3_sweep_r89_d9_t3_k32
