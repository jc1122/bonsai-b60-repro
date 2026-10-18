#!/usr/bin/env bash
set -euo pipefail
REPRO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPRO_ROOT"
python3 scripts/prepare.py
python3 scripts/download.py runtime
docker build --tag bonsai-b60-repro:2025.3-neo26.35 --file Dockerfile .
docker run --rm --network none --cpus "${BUILD_JOBS:-2}" \
  --mount "type=bind,src=$REPRO_ROOT,dst=/repro" \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e BUILD_JOBS="${BUILD_JOBS:-2}" bonsai-b60-repro:2025.3-neo26.35 \
  bash /repro/scripts/compile.sh
