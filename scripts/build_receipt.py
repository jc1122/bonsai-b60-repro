#!/usr/bin/env python3
"""Record rebuilt binaries; hashes need not equal the historical compiler output."""
import subprocess
from common import ROOT, save, sha256, versions

binary = ROOT / ".work/llama.cpp/build-sycl/bin"
files = {p.name: sha256(p) for p in sorted(binary.iterdir()) if p.is_file() and not p.is_symlink()}
save(ROOT / ".work/build-receipt.json", {
    "source_commit": versions()["source"]["commit"],
    "patch_sha256": sha256(ROOT / "patches/replay-compact-v2.patch"),
    "compiler": subprocess.check_output(["icpx", "--version"], text=True),
    "binary_sha256": files,
    "guard_sha256": sha256(ROOT / ".work/guard/gpu_budget.so"),
    "inference_tested": False,
})
