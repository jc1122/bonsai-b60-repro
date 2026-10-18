#!/usr/bin/env python3
"""Fetch the pinned public runtime and apply the recorded patch, without a build."""
import argparse
import pathlib
import subprocess
from common import ROOT, require, sha256, versions


def prepare(destination, mode="compact"):
    lock = versions()
    destination = pathlib.Path(destination).resolve()
    patch_name = "patches/replay-compact-v2.patch" if mode == "compact" else "patches/replay-multi.patch"
    patch = ROOT / patch_name
    require(sha256(patch) == lock["patches"][patch_name], "patch checksum mismatch")
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "--branch", lock["source"]["tag"],
                        lock["source"]["url"], str(destination)], check=True)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(destination), *args], text=True).strip()
    require(git("rev-parse", "HEAD") == lock["source"]["commit"], "unexpected source commit")
    status = git("status", "--porcelain", "--untracked-files=no")
    if not status:
        subprocess.run(["git", "-C", str(destination), "apply", "--check", str(patch)], check=True)
        subprocess.run(["git", "-C", str(destination), "apply", str(patch)], check=True)
    expected = lock["candidate_source_sha256"] if mode == "compact" else lock["replay_source_sha256"]
    changed = set(git("diff", "--name-only").splitlines())
    require(changed <= set(expected), "unexpected tracked edits; refusing to overwrite source")
    for name, digest in {**expected, **lock["unchanged_arithmetic_sha256"]}.items():
        require(sha256(destination / name) == digest, "source checksum mismatch: " + name)
    subprocess.run(["git", "-C", str(destination), "diff", "--check"], check=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, default=ROOT / ".work/llama.cpp")
    parser.add_argument("--mode", choices=["compact", "replay"], default="compact")
    args = parser.parse_args()
    print(prepare(args.source, args.mode))
