#!/usr/bin/env bash
# Container build only: no GPU is exposed and no server is started.
set -euo pipefail
if [[ ${SETVARS_COMPLETED:-0} != 1 ]]; then
  set +u
  source /opt/intel/oneapi/setvars.sh >/dev/null
  set -u
fi
cd /repro
cmake -S .work/llama.cpp -B .work/llama.cpp/build-sycl \
  -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build .work/llama.cpp/build-sycl --target llama-server llama-bench -j "${BUILD_JOBS:-2}"
mkdir -p .work/guard
gcc -shared -fPIC -Wall -Wextra -Werror -O2 -Iguard/include guard/gpu_budget.c \
  -ldl -pthread -o .work/guard/gpu_budget.so
python3 scripts/guard_test.py --compiler gcc
python3 scripts/build_receipt.py
