#!/usr/bin/env python3
"""Run a separate foreground Docker server; does not stop or replace any service."""
import argparse
import os
import pathlib
import shlex
import subprocess
from common import ROOT, require, sha256, versions

ENVIRONMENT = {
    "ONEAPI_DEVICE_SELECTOR": "level_zero:gpu",
    "GGML_SYCL_PTQ1_T2": "all", "GGML_SYCL_FA_DEC_DPAS": "1",
    "GGML_SYCL_FA_ONEDNN_MAX_KV": "98304", "GGML_SYCL_T2_W8A8_MIN": "1024",
    "LLAMA_ARG_SPEC_DRAFT_UBATCH": "512", "HARNESS_ORACLE": "1",
    "LLAMA_GDN_REPLAY_MULTI": "1", "LLAMA_GDN_REPLAY_COMPACT": "1",
    "LLAMA_ARG_SPEC_DRAFT_WINDOW": "0", "GGML_SYCL_T2_MULTI": "0",
    "GGML_SYCL_PTQ1_T2_NORMQ": "0", "GGML_SYCL_PTQ1_NATIVE": "0",
}


def command(model, context, port, name, image, compact=True, prompt_cache="off"):
    require(context in (16384, 262144), "only the documented three-slot profiles are supported")
    require(compact or context == 16384, "compare padded storage on the smaller profile first")
    require(1 <= port <= 65535, "invalid port")
    require(prompt_cache in ("off", "ram6"), "unsupported prompt-cache profile")
    base = "/repro/.work/llama.cpp/build-sycl/bin"
    result = ["docker", "run", "--rm", "--name", name, "--device", "/dev/dri",
              "--publish", "127.0.0.1:%d:8080" % port,
              "--mount", "type=bind,src=%s,dst=/repro,readonly" % ROOT,
              "--mount", "type=bind,src=%s,dst=/model/model.gguf,readonly" % model]
    environment = dict(ENVIRONMENT, LLAMA_GDN_REPLAY_COMPACT="1" if compact else "0")
    for key, value in environment.items():
        result += ["--env", key + "=" + value]
    result += ["--env", "BATCH_GPU_BUDGET_BYTES=24696061952", image, "bash", "-lc",
               'if [ "${SETVARS_COMPLETED:-0}" != 1 ]; then source /opt/intel/oneapi/setvars.sh >/dev/null; fi; '
               'export LD_LIBRARY_PATH="' + base + ':/usr/local/lib:${LD_LIBRARY_PATH:-}"; '
               'export GGML_BACKEND_PATH="' + base + '"; '
               'export LD_PRELOAD=/repro/.work/guard/gpu_budget.so; exec "$@"',
               "bonsai-repro", base + "/llama-server", "-m", "/model/model.gguf",
               "-ngl", "99", "-c", str(context * 3), "-np", "3", "-ub", "1024", "-b", "2048",
               "-ctk", "q4_0", "-ctv", "q4_0", "--host", "0.0.0.0", "--port", "8080",
               "--alias", "bonsai", "--jinja", "--metrics", "--flash-attn", "on", "--fit", "off",
               "--no-context-shift", "--no-kv-unified", "--cache-ram", "6144" if prompt_cache == "ram6" else "0", "--ctx-checkpoints", "1",
               "--timeout", "3600", "--spec-type", "draft-mtp", "--spec-draft-n-max", "4",
               "-ctkd", "q4_0", "-ctvd", "q4_0"]
    if prompt_cache == "ram6":
        result.append("--no-cache-idle-slots")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path)
    parser.add_argument("--context", type=int, choices=[16384, 262144], default=16384)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--name", default="bonsai-b60-repro")
    parser.add_argument("--image", default="bonsai-b60-repro:2025.3-neo26.35")
    parser.add_argument("--compact", choices=["on", "off"], default="on")
    parser.add_argument("--prompt-cache", choices=["off", "ram6"], default="off",
                        help="off reproduces the original measurements; ram6 archives evicted states in6GiB host RAM")
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    model = versions()["model"]
    path = (args.model or ROOT / "models" / model["filename"]).resolve()
    argv = command(path, args.context, args.port, args.name, args.image, args.compact == "on", args.prompt_cache)
    if args.print_command:
        print(shlex.join(argv))
        return
    require(path.is_file() and path.stat().st_size == model["bytes"] and sha256(path) == model["sha256"],
            "model checksum mismatch; run scripts/download.py model")
    require((ROOT / ".work/llama.cpp/build-sycl/bin/llama-server").is_file(), "build the runtime first")
    require((ROOT / ".work/guard/gpu_budget.so").is_file(), "build the allocation guard first")
    available = next(int(line.split()[1]) * 1024 for line in pathlib.Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    minimum = 16 if args.prompt_cache == "ram6" else 4
    require(available >= minimum * 2**30, "less than %d GiB host RAM available" % minimum)
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    main()
