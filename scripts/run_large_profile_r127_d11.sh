#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-dependency-aware-dfrcert:v01}"

docker run --rm \
  -v "$PWD/configs:/work/configs:ro" \
  -v "$PWD/artifacts:/work/artifacts" \
  "$IMAGE_TAG" \
  profiles --config configs/large_profile_r127_d11_t2_k512.toml \
  --out artifacts/large_profile_r127_d11_t2_k512
