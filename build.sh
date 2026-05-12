#!/usr/bin/env bash
set -euo pipefail

export IMAGE_TAG=dependency-aware-dfrcert:v01
docker build -t "$IMAGE_TAG" .
echo "Built $IMAGE_TAG"
echo "Run example:"
echo 'docker run --rm -v "$PWD/artifacts:/work/artifacts" "$IMAGE_TAG" run --config configs/exact_toy_r11_d3_t3.toml --out artifacts/exact_toy_r11_d3_t3'
