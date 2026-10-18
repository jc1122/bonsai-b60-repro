#!/usr/bin/env python3
"""Re-run the source/serialization regression on CPU buffers, without a GPU."""
import argparse
import os
import pathlib
import shutil
import subprocess
import time
from common import ROOT, require, save, sha256
from prepare import prepare


def extract_body(source, signature):
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 0
    for end in range(opening, len(source)):
        depth += (source[end] == "{") - (source[end] == "}")
        if depth == 0:
            return source[opening:end + 1]
    raise RuntimeError("unterminated production build_rs body")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    require(1 <= args.jobs <= 32, "invalid jobs count")
    source = prepare(ROOT / ".work/llama.cpp")
    baseline = ROOT / ".work/replay-control"
    if not baseline.exists():
        subprocess.run(["git", "clone", "--no-hardlinks", str(source), str(baseline)], check=True)
    prepare(baseline, "replay")
    library = ROOT / ".work/cpu-backend/bin"
    subprocess.run(["cmake", "-S", str(source), "-B", str(library.parent),
                    "-DCMAKE_BUILD_TYPE=Release", "-DGGML_SYCL=OFF", "-DGGML_NATIVE=OFF",
                    "-DGGML_OPENMP=OFF", "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF"], check=True)
    subprocess.run(["cmake", "--build", str(library.parent), "--target", "ggml-cpu", "-j", str(args.jobs)], check=True)
    output = ROOT / ".work/state-check"
    output.mkdir(parents=True, exist_ok=True)
    signature = "ggml_tensor * llm_graph_context::build_rs(\n        ggml_tensor * s,"
    body = extract_body((source / "src/llama-graph.cpp").read_text(), signature)
    declaration = ("static ggml_tensor * fixture_build_rs(ggml_context * ctx0, ggml_cgraph * gf, "
                   "ggml_tensor * s, ggml_tensor * state_copy_main, ggml_tensor * state_copy_extra, "
                   "int32_t state_size, int32_t n_seqs, uint32_t n_rs, uint32_t rs_head, uint32_t rs_size, "
                   "int32_t rs_zero, const std::function<ggml_tensor *(ggml_context *, ggml_tensor *, ggml_tensor *)> & get_state_rows) ")
    (output / "production_build_rs.inc").write_text(declaration + body + "\n")
    fixture = ROOT / "tests/fixtures"
    results = []
    for name, tree, compact in [("red-padded-baseline", baseline, 0), ("green-compact", source, 1)]:
        arm = output / name
        arm.mkdir(exist_ok=True)
        production = tree / "src/llama-memory-recurrent.cpp"
        text = production.read_text()
        require(text.count('#include "llama-model.h"') == 1, "unexpected model include")
        (arm / "memory.cpp").write_text(text.replace('#include "llama-model.h"', '#include "model_cpu_adapter.h"'))
        shutil.copyfile(tree / "src/llama-memory-recurrent.h", arm / "llama-memory-recurrent.h")
        command = ["c++", "-std=c++17", "-O1", "-g", "-Wall", "-Wextra", "-Werror", "-Wno-unused-parameter",
                   "-ffunction-sections", "-fdata-sections", "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                   "-DHAS_COMPACT=" + str(compact), "-I" + str(arm), "-I" + str(fixture), "-I" + str(output)]
        for include in ["include", "ggml/include", "ggml/src", "src"]:
            command += ["-I" + str(source / include)]
        command += [str(arm / "memory.cpp"), str(fixture / "compact_memory_fixture.cpp"),
                    str(source / "src/llama-hparams.cpp"), str(source / "src/llama-batch.cpp"),
                    str(source / "src/llama-memory.cpp"), "-L" + str(library), "-lggml-cpu", "-lggml-base",
                    "-Wl,--gc-sections", "-Wl,-rpath," + str(library), "-pthread", "-o", str(arm / "fixture")]
        started = time.monotonic()
        subprocess.run(command, check=True, timeout=180)
        env = dict(os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1")
        result = subprocess.run([str(arm / "fixture")], env=env, capture_output=True, text=True, timeout=120)
        row = {"arm": name, "exit_code": result.returncode, "stdout": result.stdout,
               "stderr": result.stderr, "source_sha256": sha256(production), "seconds": time.monotonic() - started}
        results.append(row)
        save(output / (name + ".json"), row)
        if compact:
            require(result.returncode == 0, "compact fixture failed: " + result.stderr[-3000:])
        else:
            require(result.returncode == 1 and "compact allocation active" in result.stderr,
                    "baseline did not fail the intended layout assertion: " + result.stderr[-3000:])
    save(output / "result.json", {"success": True, "gpu_access": False,
                                 "scope": "CPU-buffer layout/serialization adapter; not model math qualification",
                                 "results": results})
    print("CPU state regression: expected baseline failure and compact pass.")


if __name__ == "__main__":
    main()
