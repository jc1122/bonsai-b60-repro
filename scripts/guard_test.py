#!/usr/bin/env python3
"""Exercise the allocation guard against a fake Level Zero library; no GPU needed."""
import argparse
import os
import subprocess
from common import ROOT, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="cc")
    args = parser.parse_args()
    build = ROOT / ".work/guard-test"
    build.mkdir(parents=True, exist_ok=True)
    flags = [args.compiler, "-Wall", "-Wextra", "-Werror", "-O2", "-I" + str(ROOT / "guard/include")]
    sources = ROOT / "guard"
    commands = [
        flags + ["-shared", "-fPIC", str(sources / "gpu_budget.c"), "-ldl", "-pthread", "-o", "gpu_budget.so"],
        flags + ["-shared", "-fPIC", str(sources / "guard_fake.c"), "-o", "libguard_fake.so"],
        flags + [str(sources / "guard_test.c"), "-L.", "-l:libguard_fake.so", "-Wl,-rpath,$ORIGIN", "-pthread", "-o", "guard-test"],
    ]
    for command in commands:
        subprocess.run(command, cwd=build, check=True)
    env = dict(os.environ, LD_PRELOAD=str(build / "gpu_budget.so"), BATCH_GPU_BUDGET_BYTES="128")
    for _ in range(10):
        result = subprocess.run([str(build / "guard-test")], env=env, capture_output=True, text=True, timeout=15)
        require(result.returncode == 0, result.stdout + result.stderr)
    print("Allocation guard: ten fake-driver trials passed (no GPU).")


if __name__ == "__main__":
    main()
