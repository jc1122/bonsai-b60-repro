#!/usr/bin/env python3
"""Download public model or runtime dependencies and verify their locked hashes."""
import argparse
import pathlib
import urllib.request
from common import ROOT, require, sha256, versions


def download(url, path, size, digest):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "bonsai-b60-repro/1"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
            require(temporary.stat().st_size == size and sha256(temporary) == digest,
                    "download verification failed: " + path.name)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    require(path.stat().st_size == size and sha256(path) == digest,
            "existing file differs from lock: " + str(path))
    print("verified", path.name, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["model", "runtime"])
    parser.add_argument("--directory", type=pathlib.Path)
    args = parser.parse_args()
    lock = versions()
    if args.kind == "runtime":
        directory = args.directory or ROOT / "runtime/debs"
        for item in lock["runtime"]["packages"]:
            download(item["url"], directory / item["name"], item["bytes"], item["sha256"])
    else:
        model = lock["model"]
        directory = args.directory or ROOT / "models"
        url = "https://huggingface.co/{repository}/resolve/{revision}/{filename}".format(**model)
        download(url, directory / model["filename"], model["bytes"], model["sha256"])
