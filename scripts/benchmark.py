#!/usr/bin/env python3
"""Run the original populated-context workload against an explicitly chosen server."""
import argparse
import json
import pathlib
import statistics
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from common import require, save, sha256
import server_workload as workload
from launch import ENVIRONMENT

PHYSICAL_VRAM = 25669140480  # Measured physical size of the tested B60, not advertised 24 GiB.
FDINFO_SCRIPT = '''from pathlib import Path
for path in Path("/proc/1/fdinfo").iterdir():
    try:
        text = path.read_text()
    except FileNotFoundError:
        continue
    lines = [line for line in text.splitlines() if line.startswith("drm-")]
    if lines:
        print("\\n".join(lines))
'''


class Client:
    def __init__(self, base_url):
        parsed = urllib.parse.urlparse(base_url)
        require(parsed.scheme in ("http", "https") and parsed.hostname and parsed.port,
                "supply a server origin with an explicit port")
        require(parsed.path in ("", "/") and not parsed.query and not parsed.fragment and not parsed.username,
                "supply an origin, without credentials or /v1")
        self.base_url = base_url.rstrip("/")

    def api(self, path, body=None, timeout=30):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base_url + path, data=data,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def completion(self, body, timeout):
        request = urllib.request.Request(self.base_url + "/completion", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        pieces, final, terminal = [], None, False
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                item = json.loads(data)
                require("error" not in item, "server SSE error: " + str(item.get("error")))
                pieces.append(item.get("content", ""))
                if item.get("stop") is True:
                    final, terminal = item, True
        require(terminal and final is not None, "SSE ended without a terminal completion")
        return {"content": "".join(pieces), "final": final}


def drm_residency(text):
    clients, current = {}, None
    for line in text.splitlines():
        if line.startswith("drm-driver:"):
            current = {}
        if current is not None and ":" in line:
            key, value = line.split(":", 1)
            current[key] = value.strip()
            if "drm-client-id" in current:
                clients[current["drm-client-id"]] = current
    def amount(key):
        return sum(int(v.get(key, "0").split()[0]) * 1024 for v in clients.values())
    require(bool(clients), "no DRM clients visible; GPU residency is unverified")
    return {"resident_vram_bytes": amount("drm-resident-vram0"),
            "resident_system_bytes": amount("drm-resident-system"), "unique_clients": len(clients)}


def resources(container):
    result = subprocess.run(["docker", "exec", container, "python3", "-c", FDINFO_SCRIPT],
                            capture_output=True, text=True, timeout=15, check=True)
    row = drm_residency(result.stdout)
    row["unix"] = time.time()
    row["host_mem_available_bytes"] = next(int(line.split()[1]) * 1024
        for line in pathlib.Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--container", default="bonsai-b60-repro")
    parser.add_argument("--context", type=int, choices=[16384, 262144], default=16384)
    parser.add_argument("--output-tokens", type=int, default=1024)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    require(1 <= args.output_tokens <= 1024, "output must be between 1 and 1024 tokens")
    require(not args.output.exists(), "use a fresh output directory")
    args.output.mkdir(parents=True)
    client = Client(args.base_url)
    slots = client.api("/slots")
    require(len(slots) == 3 and all(s.get("n_ctx") == args.context for s in slots),
            "server must expose exactly three slots of the requested context")
    require(not any(s.get("is_processing") for s in slots), "server already has active requests")
    save(args.output / "slots-before.json", slots)
    # Confirm telemetry belongs to the same isolated container reached over HTTP.
    inspect = json.loads(subprocess.check_output(["docker", "inspect", args.container], text=True))[0]
    ports = inspect["NetworkSettings"]["Ports"].get("8080/tcp") or []
    origin = urllib.parse.urlparse(args.base_url)
    require(origin.hostname in ("127.0.0.1", "localhost") and any(
        p["HostIp"] == "127.0.0.1" and p["HostPort"] == str(origin.port) for p in ports),
        "HTTP origin does not match the selected container's loopback port")
    save(args.output / "run-config.json", {"context_each": args.context, "sessions": 3,
        "input_tokens_each": args.context - 4096, "output_tokens_each": args.output_tokens,
        "physical_vram_bytes": PHYSICAL_VRAM, "container_id": inspect["Id"],
        "image_id": inspect.get("Image"),
        "runtime_environment": {k: v for entry in inspect.get("Config", {}).get("Env", [])
                                for k, _, v in [entry.partition("=")] if k in ENVIRONMENT},
        "benchmark_sha256": sha256(__file__), "workload_sha256": sha256(workload.__file__),
        "synthetic_workload": True, "coding_quality_scored": False})
    samples, errors, stop = [], [], threading.Event()
    def sample():
        row = resources(args.container)
        samples.append(row)
        require(row["resident_system_bytes"] == 0, "GPU has resident System allocations")
        require(row["resident_vram_bytes"] <= PHYSICAL_VRAM, "physical VRAM limit exceeded")
        require(row["host_mem_available_bytes"] >= 4 * 2**30, "host RAM floor crossed")
    sample()
    def monitor():
        while not stop.wait(2):
            try:
                sample()
            except Exception as exc:
                errors.append(str(exc))
                break
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    result = {"success": False}
    try:
        recorder = workload._Recorder(client.api, args.output)
        items = workload.make_prompts(recorder.call, args.context, args.output / "source_prompts")
        waves = []
        for phase, warm in [("cold", False), ("warm1", True), ("warm2", True)]:
            require(not errors, "resource monitoring failed: " + str(errors))
            waves.append(workload._wave(recorder, client.completion, args.output, phase,
                                        items, args.output_tokens, args.timeout, warm=warm))
        sample()
        require(not errors, "resource monitoring failed: " + str(errors))
        rate = statistics.mean(w["aggregate_tokens_per_s"] for w in waves[1:])
        result.update(success=True, waves=waves, warm_aggregate_tok_s=rate,
                      warm_mean_session_share_tok_s=rate / 3,
                      cold_full_batch_seconds=waves[0]["wall_s"],
                      coding_quality_scored=False)
    except BaseException as exc:
        result["error"] = str(exc)
        raise
    finally:
        stop.set()
        thread.join(timeout=20)
        save(args.output / "resources.json", samples)
        result.update(resource_errors=errors,
            max_resident_vram_bytes=max(s["resident_vram_bytes"] for s in samples),
            physical_headroom_bytes_min=PHYSICAL_VRAM - max(s["resident_vram_bytes"] for s in samples),
            max_resident_system_bytes=max(s["resident_system_bytes"] for s in samples))
        if errors:
            result["success"] = False
        save(args.output / "result.json", result)
        print(json.dumps(result, indent=2))
    require(result["success"], "benchmark failed; inspect result.json")


if __name__ == "__main__":
    main()
