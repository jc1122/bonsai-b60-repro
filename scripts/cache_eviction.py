#!/usr/bin/env python3
"""Measure cross-slot prompt restoration on a separate, already running server."""
import argparse
import hashlib
import http.client
import json
import pathlib
import socket
import subprocess
import threading
import time
import urllib.parse

from benchmark import Client, PHYSICAL_VRAM, resources
from common import require, save
import server_workload as workload


class EvictionClient(Client):
    def __init__(self, base_url):
        super().__init__(base_url)
        origin = urllib.parse.urlparse(base_url)
        require(origin.scheme == "http", "use the local HTTP server")
        self.host, self.port = origin.hostname, origin.port
        self.lock, self.active = threading.Lock(), None
        self.cancelled = threading.Event()

    def abort(self):
        # Interrupt our own blocked HTTP read, causing server-side cancellation.
        self.cancelled.set()
        with self.lock:
            if self.active and self.active.sock:
                try: self.active.sock.shutdown(socket.SHUT_RDWR)
                except OSError: pass

    def completion(self, body, timeout):
        require(not self.cancelled.is_set(), "resource guard cancelled this client")
        connection = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        with self.lock: self.active = connection
        started = time.monotonic(); first = None; pieces = []; tokens = []; final = None
        try:
            connection.request("POST", "/completion", json.dumps(body), {"Content-Type": "application/json"})
            require(not self.cancelled.is_set(), "resource guard cancelled this request")
            response = connection.getresponse()
            require(response.status == 200, "completion HTTP failure: %s" % response.status)
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:"): continue
                data = line[5:].strip()
                if data == "[DONE]": break
                item = json.loads(data)
                require("error" not in item, "SSE error: " + str(item.get("error")))
                if first is None and (item.get("tokens") or item.get("content")):
                    first = time.monotonic()
                pieces.append(item.get("content", ""))
                if item.get("stop"):
                    final = item
                    if item.get("tokens"): tokens = item["tokens"]
                else: tokens.extend(item.get("tokens", []))
            require(final is not None and first is not None, "unfinished or empty SSE")
            return {"content": "".join(pieces), "tokens": tokens, "final": final,
                    "ttft_s": first - started, "wall_s": time.monotonic() - started}
        finally:
            with self.lock: self.active = None
            connection.close()


def request_body(prompt, slot, output_tokens):
    return {"prompt": prompt, "id_slot": slot, "n_predict": output_tokens,
            "ignore_eos": True, "cache_prompt": True, "stream": True, "return_tokens": True,
            "seed": 271828, "temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0,
            "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}


def check_pair(initial, resumed, input_tokens, expect_reuse):
    require(len(initial["tokens"]) == len(resumed["tokens"]) == 128, "missing generated token IDs")
    require(initial["tokens"] == resumed["tokens"] and initial["content"] == resumed["content"],
            "restored greedy output differs")
    timing = resumed["final"]["timings"]
    if expect_reuse:
        require(timing.get("cache_n", 0) >= input_tokens - 2048 and timing["prompt_n"] <= 2048,
                "evicted prefix was not restored")
        require(resumed["ttft_s"] < initial["ttft_s"] * .25, "restore was not clearly faster")
        require(timing.get("draft_n", 0) > 0 and timing.get("draft_n_accepted", 0) > 0,
                "MTP did not perform accepted draft work")
    else:
        require(timing["prompt_n"] >= input_tokens - 2048, "control unexpectedly reused evicted input")
    return {"input_tokens": input_tokens, "initial_ttft_s": initial["ttft_s"],
            "resumed_ttft_s": resumed["ttft_s"], "cache_n": timing.get("cache_n"),
            "prompt_n": timing["prompt_n"], "greedy_equivalent": True,
            "content_sha256": hashlib.sha256(initial["content"].encode()).hexdigest(),
            "resume_slot": 1, "original_slot": 0, "all_slots_replaced": True,
            "draft_n": timing.get("draft_n"), "draft_n_accepted": timing.get("draft_n_accepted")}


def swap_counters():
    return {k: int(v) for k, v in (s.split() for s in pathlib.Path("/proc/vmstat").read_text().splitlines())
            if k in ("pswpin", "pswpout")}


def memory_sample(container):
    row = resources(container)
    code = "from pathlib import Path;print(next(s.split()[1] for s in Path('/proc/1/status').read_text().splitlines() if s.startswith('VmSwap:')))"
    row["process_swap_bytes"] = int(subprocess.check_output(
        ["docker", "exec", container, "python3", "-c", code], text=True, timeout=10)) * 1024
    row["swap_counters"] = swap_counters()
    row["memory_PSI_avg10"] = {parts[0]: float(parts[1].split("=")[1]) for parts in
        (line.split() for line in pathlib.Path("/proc/pressure/memory").read_text().splitlines())}
    return row


def memory_gate(row, baseline):
    require(row["resident_system_bytes"] == 0, "GPU reports resident System memory")
    require(0 < row["resident_vram_bytes"] <= PHYSICAL_VRAM, "physical VRAM gate failed")
    require(row["host_mem_available_bytes"] >= 6 * 2**30, "host RAM reserve breached")
    require(row["process_swap_bytes"] == 0, "inference process swapped")
    require(row["swap_counters"]["pswpout"] == baseline["pswpout"], "new host swap-outs")
    require(0 <= row["swap_counters"]["pswpin"] - baseline["pswpin"] <= 4096, "historical swap-ins exceeded16MiB")
    require(set(row["memory_PSI_avg10"]) == {"some", "full"}, "missing memory pressure counters")
    require(all(v == 0 for v in row["memory_PSI_avg10"].values()), "host memory pressure")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--container", default="bonsai-b60-repro")
    parser.add_argument("--input-tokens", type=int, choices=[12288, 100352], default=12288)
    parser.add_argument("--expect-reuse", action="store_true")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "use a fresh output directory")
    client = EvictionClient(args.base_url)
    info = json.loads(subprocess.check_output(["docker", "inspect", args.container], text=True))[0]
    bindings = info["NetworkSettings"]["Ports"].get("8080/tcp") or []
    require(client.host in ("127.0.0.1", "localhost") and any(
        p["HostIp"] == "127.0.0.1" and p["HostPort"] == str(client.port) for p in bindings),
        "HTTP origin and telemetry container differ")
    slots = client.api("/slots")
    require(len(slots) == 3 and all(not x["is_processing"] for x in slots), "need three idle slots")
    require(all(x.get("speculative") for x in slots), "expected MTP-enabled slots")
    context = min(x["n_ctx"] for x in slots)
    require(context > args.input_tokens + 1024, "input does not fit")
    args.output.mkdir(parents=True)
    # Keep only non-secret launch metadata.
    save(args.output / "config.json", {"context_each": context, "input_tokens": args.input_tokens,
         "expect_reuse": args.expect_reuse, "container_id": info["Id"], "image_id": info["Image"],
         "synthetic_workload": True, "coding_quality_scored": False})
    samples, faults, stop = [], [], threading.Event(); baseline = swap_counters()
    def monitor():
        while not stop.is_set():
            try:
                row = memory_sample(args.container); samples.append(row); memory_gate(row, baseline)
            except Exception as exc:
                faults.append(str(exc)); client.abort(); return
            stop.wait(2)
    thread = threading.Thread(target=monitor, daemon=True); thread.start()
    result = {"success": False}
    def call(label, prompt, slot, predicted):
        require(not faults, str(faults))
        body = request_body(prompt, slot, predicted)
        save(args.output / "requests" / (label + "-input.json"), body)
        value = client.completion(body, 900)
        require(len(value["tokens"]) == predicted and value["final"]["timings"]["predicted_n"] == predicted,
                "wrong output count")
        require(not value["final"].get("truncated"), "context truncated")
        require(value["final"].get("id_slot") == slot, "request ran in the wrong slot")
        save(args.output / "requests" / (label + ".json"), value)
        require(not faults, str(faults))
        return value
    try:
        short = workload.make_prompts(client.api, context, args.output / "short", targets=[12288] * 3)
        prompt = short[0]["fixed_prompt"]
        if args.input_tokens != 12288:
            prompt = workload.make_prompts(client.api, context, args.output / "long",
                                           targets=[args.input_tokens, 12288, 12288])[0]["fixed_prompt"]
        initial = call("initial", prompt, 0, 128)
        for slot, item in enumerate(short):
            blocker = list(item["fixed_prompt"]); blocker[0] = 100 + slot
            call("evict%d" % slot, blocker, slot, 16)
        replaced = client.api("/slots")
        require(len(replaced) == 3 and all(not x["is_processing"] for x in replaced), "evictions did not finish")
        save(args.output / "slots-before-resume.json", replaced)
        resumed = call("resumed", prompt, 1, 128)
        result.update(check_pair(initial, resumed, args.input_tokens, args.expect_reuse), success=True)
    except BaseException as exc:
        result["error"] = str(exc)
        raise
    finally:
        stop.set(); thread.join(timeout=20)
        if thread.is_alive(): faults.append("resource monitor did not stop")
        try:
            row = memory_sample(args.container); samples.append(row); memory_gate(row, baseline)
        except Exception as exc: faults.append(str(exc))
        result["resource_errors"] = faults
        if faults: result["success"] = False
        save(args.output / "resources.json", samples)
        save(args.output / "result.json", result)
        print(json.dumps(result, indent=2))
    require(result["success"], "eviction test failed")


if __name__ == "__main__": main()
