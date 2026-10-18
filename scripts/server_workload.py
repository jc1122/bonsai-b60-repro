"""Ready-server HTTP checks; importing this module performs no work.

run_server_checks(api, completion, output_dir, ctx, items=None, *, base_url,
                  authorized=False, timeout=7200, cancellation_oracle=True)

api(path, body=None, timeout=30) returns decoded JSON. completion(body, timeout)
returns either a final /completion JSON response or speed_base's streaming wrapper
{content, final, ...}. Both callbacks MUST use the orchestrator's explicit port.
base_url is explicit and is used only to disconnect the cancellation SSE stream.
The caller owns server launch, GPU monitoring, authorization and recovery.
"""
import concurrent.futures
import hashlib
import http.client
import json
import pathlib
import random
import socket
import statistics
import threading
import time
import urllib.parse


def _save(path, data):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


class _Recorder:
    def __init__(self, api, output):
        self.api = api
        self.output = pathlib.Path(output)
        self.lock = threading.Lock()
        self.counter = 0

    def call(self, path, body=None, timeout=30):
        with self.lock:
            number = self.counter
            self.counter += 1
        label = "%05d_%s" % (number, path.strip("/").replace("/", "_"))
        record = {"path": path, "method": "POST" if body is not None else "GET", "timeout_s": timeout}
        if body is not None:
            # Preserve exact payloads in separate files, including large tokenize text.
            body_file = self.output / "http" / (label + "_body.json")
            _save(body_file, body)
            record["body_file"] = str(body_file.relative_to(self.output))
        started = time.monotonic()
        try:
            response = self.api(path, body, timeout=timeout)
            _save(self.output / "http" / (label + "_response.json"), response)
            record["response_file"] = "http/" + label + "_response.json"
            return response
        except Exception as exc:
            record["error"] = repr(exc)
            raise
        finally:
            record["wall_s"] = time.monotonic() - started
            _save(self.output / "http" / (label + ".json"), record)


def make_prompts(api, ctx, output_dir, targets=None):
    """Copy of speed_base's populated deterministic generator, with explicit API.

    No import of speed_base, PORT default or model request occurs until called.
    Returns entries with fixed_prompt token lists and saves each complete entry.
    """
    output = pathlib.Path(output_dir)
    targets = list(targets or [ctx - 4096] * 3)
    _require(len(targets) == 3, "exactly three prompt targets required")
    marker = "__BENCH_BODY_98B82__"
    template = api("/apply-template", {
        "messages": [{"role": "user", "content": marker}],
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "medium"},
    }, timeout=30)["prompt"]
    _require(template.count(marker) == 1, "template marker is not unique")
    pre, post = template.split(marker)

    def tokens(text):
        return api("/tokenize", {"content": text, "add_special": False}, timeout=180)["tokens"]

    result = []
    for job, target in enumerate(targets):
        rng = random.Random(110001 + job)
        nonce = "job_%02d_a93f" % job
        a, b, k = rng.randint(3, 19), rng.randint(2, 7), rng.randint(1, 31)
        prefix = tokens(pre + "Batch coding job %s. The following are files from a synthetic Python repository. "
                        "Configuration constants occur in different files.\n```python\n" % nonce)
        tail = tokens("\n```\nFor %s, create a fresh Python test module implementing "
                      "solve(x)=((x+A_%s)*B_%s)^C_%s, using the three configuration constants in the repository. "
                      "The ^ operator is Python integer bitwise XOR; do not exponentiate. Include 160 individually "
                      "named test functions with varied inputs and explain tricky boundary cases. "
                      "Produce the complete long module.\n" % (nonce, nonce, nonce, nonce) + post)
        tags = [tokens("\n# file config_%s_%s.py\n%s_%s = %d\n" % (nonce, name, name, nonce, value))
                for name, value in zip("ABC", [a, b, k])]
        need = target - len(prefix) - len(tail) - sum(map(len, tags))
        _require(need > 0, "prompt target too small")
        lines = ["def unrelated_%s_%d(x): return (x + %d) %% %d\n" %
                 (nonce, i, rng.randrange(100000), rng.randrange(101, 9999))
                 for i in range(need // 15 + 500)]
        filler = tokens("".join(lines))
        _require(bool(filler), "empty filler tokenization")
        while len(filler) < need:
            filler += filler
        middle, start = [], 0
        for index, end in enumerate([need // 10, need // 2, need * 9 // 10, need]):
            middle += filler[start:end]
            start = end
            if index < 3:
                middle += tags[index]
        ids = prefix + middle + tail
        _require(len(ids) == target, "populated prompt token count mismatch")
        _require(len(ids) + 1024 < ctx, "prompt leaves insufficient decode space")
        entry = {"job": job, "nonce": nonce, "expected": [a, b, k],
                 "input_tokens": len(ids), "fixed_prompt": ids}
        _save(output / ("%d.json" % job), entry)
        result.append(entry)
    return result


def _body(prompt, slot, predicted, cache, greedy=False):
    return {"prompt": prompt, "id_slot": slot, "n_predict": predicted,
            "ignore_eos": True, "cache_prompt": cache, "stream": True,
            "return_tokens": True, "seed": 1100 + slot, "temperature": 0.0 if greedy else 0.1,
            "top_p": 1.0 if greedy else 0.95, "top_k": 1 if greedy else 50, "min_p": 0.0,
            "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}


def _normalized(response):
    return response if isinstance(response, dict) and "final" in response else {
        "content": response.get("content", ""), "final": response}


def _record_completion(completion, output, label, body, timeout, warm=False):
    path = output / "requests" / (label + ".json")
    prompt_file = output / "prompts" / (label + ".json")
    _save(prompt_file, body["prompt"])
    record = {"label": label, "request": {key: value for key, value in body.items() if key != "prompt"},
              "prompt_file": str(prompt_file.relative_to(output)), "input_tokens": len(body["prompt"]),
              "effective_n_predict": body["n_predict"], "started_monotonic": time.monotonic()}
    _save(path, record)
    try:
        response = _normalized(completion(body, timeout))
        final = response["final"]
        # Server final JSON echoes a very long detokenized prompt: preserve separately.
        if isinstance(final.get("prompt"), str):
            echo = output / "prompts" / (label + "_server_echo.txt")
            echo.write_text(final["prompt"], encoding="utf-8")
            final = dict(final)
            final["prompt"] = {"file": str(echo.relative_to(output))}
            response = dict(response, final=final)
        record["response"] = response
        timings = final.get("timings", {})
        predicted = timings.get("predicted_n")
        record["predicted_n"] = predicted
        record["cache_n"] = timings.get("cache_n")
        record["prompt_n"] = timings.get("prompt_n")
        record["truncated"] = bool(final.get("truncated"))
        checks = {"predicted_n_exact": predicted == body["n_predict"],
                  "not_truncated": not record["truncated"],
                  "slot_matches": final.get("id_slot", body["id_slot"]) == body["id_slot"]}
        settings = final.get("generation_settings", {})
        if "n_predict" in settings:
            checks["server_n_predict_matches"] = settings["n_predict"] == body["n_predict"]
        if warm:
            cache_n, prompt_n = timings.get("cache_n"), timings.get("prompt_n")
            checks["prefix_cache_reused"] = (isinstance(cache_n, (int, float)) and
                                             cache_n >= max(1, len(body["prompt"]) - 2048))
            checks["bounded_prefix_replay"] = isinstance(prompt_n, (int, float)) and prompt_n <= 2048
        record["checks"] = checks
        record["ok"] = all(checks.values())
    except Exception as exc:
        record["error"] = repr(exc)
        record["ok"] = False
    record["ended_monotonic"] = time.monotonic()
    record["wall_s"] = record["ended_monotonic"] - record["started_monotonic"]
    _save(path, record)
    return record


def _wave(recorder, completion, output, phase, items, predicted, timeout, warm=False, require_concurrency=True):
    barrier = threading.Barrier(3)

    def worker(item):
        barrier.wait(timeout=30)
        return _record_completion(completion, output, "%s_s%d" % (phase, item["job"]),
                                  _body(item["fixed_prompt"], item["job"], predicted, warm), timeout, warm)

    samples, records = [], []
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(worker, item) for item in items}
        while pending:
            done, pending = concurrent.futures.wait(pending, timeout=0.5,
                                                    return_when=concurrent.futures.FIRST_COMPLETED)
            records.extend(future.result() for future in done)
            if pending:
                try:
                    slots = recorder.call("/slots", timeout=5)
                    samples.append({"elapsed_s": time.monotonic() - started,
                                    "active": sum(bool(slot.get("is_processing")) for slot in slots)})
                except Exception as exc:
                    samples.append({"elapsed_s": time.monotonic() - started, "error": repr(exc)})
    summary = {"phase": phase, "effective_n_predict": predicted, "expected_requests": 3,
               "ok_requests": sum(row["ok"] for row in records), "wall_s": time.monotonic() - started,
               "peak_active_slots": max((sample.get("active", 0) for sample in samples), default=0),
               "request_files": ["requests/%s.json" % row["label"] for row in records]}
    summary["concurrency_required"] = require_concurrency
    summary["ok"] = summary["ok_requests"] == 3 and (not require_concurrency or summary["peak_active_slots"] == 3)
    if records:
        summary["median_request_s"] = statistics.median(row["wall_s"] for row in records)
        summary["generated_tokens"] = sum(row.get("predicted_n", 0) or 0 for row in records)
        summary["aggregate_tokens_per_s"] = summary["generated_tokens"] / summary["wall_s"]
    _save(output / "waves" / (phase + "_samples.json"), samples)
    _save(output / "waves" / (phase + ".json"), summary)
    _require(summary["ok"], "wave %s failed; inspect persisted requests" % phase)
    return summary


def _cancel_after_three(base_url, output, body, timeout):
    parsed = urllib.parse.urlparse(base_url)
    _require(parsed.scheme in ("http", "https") and parsed.hostname and parsed.port,
             "cancellation requires an explicit HTTP(S) hostname and port")
    _require(parsed.path in ("", "/") and not parsed.query, "base_url must be a server origin")
    cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = cls(parsed.hostname, parsed.port, timeout=timeout)
    record = {"request": {key: value for key, value in body.items() if key != "prompt"},
              "prompt_file": "prompts/cancel.json", "events": [], "target_streamed_tokens": 3}
    _save(output / "prompts" / "cancel.json", body["prompt"])
    response = None
    started = time.monotonic()
    try:
        connection.request("POST", "/completion", json.dumps(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        _require(response.status == 200, "cancellation HTTP status %d" % response.status)
        seen = 0
        for line in response:
            if not line.startswith(b"data:"):
                continue
            raw = line[5:].strip()
            if raw == b"[DONE]":
                break
            event = json.loads(raw)
            record["events"].append(event)
            _require(not event.get("error"), "cancellation stream returned an error")
            if event.get("tokens"):
                seen += len(event["tokens"])
            reported = event.get("tokens_predicted")
            observed = reported if isinstance(reported, int) else seen
            if observed >= 3:
                record["observed_streamed_tokens"] = observed
                record["received_token_ids_count"] = seen
                record["exactly_three_observed"] = observed == 3
                record["disconnect_before_final"] = not event.get("stop", False)
                break
        _require(record.get("observed_streamed_tokens", 0) >= 3, "did not observe three generated tokens before disconnect")
        _require(record.get("disconnect_before_final"), "stream completed before cancellation")
        record["ok"] = True
    except Exception as exc:
        record.update(ok=False, error=repr(exc))
    finally:
        # Force a real client disconnect rather than leaving a buffered response open.
        if connection.sock is not None:
            try:
                connection.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if response is not None:
            response.close()
        connection.close()
        record["wall_s"] = time.monotonic() - started
        _save(output / "cancellation.json", record)
    _require(record["ok"], "cancellation check failed; inspect cancellation.json")
    return record


def _wait_idle(recorder, slot_id, timeout=60):
    deadline = time.monotonic() + timeout
    observations = []
    while time.monotonic() < deadline:
        slots = recorder.call("/slots", timeout=5)
        slot = next((slot for slot in slots if slot.get("id") == slot_id or slot.get("id_slot") == slot_id), None)
        _require(slot is not None, "cancelled slot missing from /slots")
        observations.append(slot)
        if not slot.get("is_processing"):
            return observations
        time.sleep(0.25)
    raise RuntimeError("cancelled slot did not become idle")


def run_server_checks(api, completion, output_dir, ctx, items=None, *, base_url,
                      authorized=False, timeout=7200, cancellation_oracle=True):
    """Run only against an already-ready, caller-owned three-slot server.

    No service/container/process management. The authorization guard fails before
    any HTTP call. Functional oracle requests use 32 greedy tokens; capacity waves
    use exactly 512 (16k context) or 1024 (65k context), including unequal depths.
    On failure evidence is retained and the exception propagates to the caller.
    """
    _require(authorized is True, "root release required: pass authorized=True only after explicit release")
    _require(ctx in (16384, 65536), "supported context per slot is 16384 or 65536")
    output = pathlib.Path(output_dir)
    _require(not (output / "result.json").exists(), "refusing to overwrite an existing workload result")
    output.mkdir(parents=True, exist_ok=True)
    recorder = _Recorder(api, output)
    result = {"ctx_per_slot": ctx, "slots": 3, "status": "running", "waves": [],
              "explicit_server_origin": base_url,
              "capacity_n_predict": 512 if ctx == 16384 else 1024,
              "warm_assertion": "cache_n >= prompt_tokens-2048 and prompt_n <=2048; server logs prove checkpoint mechanism"}
    _save(output / "result.json", result)
    try:
        _require(recorder.call("/health", timeout=10).get("status") == "ok", "server is not healthy")
        slots = recorder.call("/slots", timeout=10)
        _require(len(slots) == 3, "server does not expose exactly three slots")
        _require(not any(slot.get("is_processing") for slot in slots), "server slots are not idle")
        _save(output / "props.json", recorder.call("/props", timeout=10))
        if items is None:
            items = make_prompts(recorder.call, ctx, output / "source_prompts")
        items = sorted(items, key=lambda item: item["job"])
        _require([item["job"] for item in items] == [0, 1, 2], "prompt jobs must be 0,1,2")
        for item in items:
            _require(len(item["fixed_prompt"]) == ctx - 4096, "uniform prompt must fill ctx-4096 tokens")
            _save(output / "source_prompts" / ("%d.json" % item["job"]), item)
        predicted = result["capacity_n_predict"]
        for phase in ("cold", "warm1", "warm2"):
            result["waves"].append(_wave(recorder, completion, output, phase, items, predicted,
                                         timeout, warm=phase != "cold"))
            _save(output / "result.json", result)
        target = ctx - 4096
        unequal = make_prompts(recorder.call, ctx, output / "unequal_source_prompts",
                               [target, target * 65 // 100, target * 35 // 100])
        for phase in ("unequal_cold", "unequal_warm1", "unequal_warm2"):
            result["waves"].append(_wave(recorder, completion, output, phase, unequal, predicted,
                                         timeout, warm=phase != "unequal_cold"))
            _save(output / "result.json", result)
        suffix = recorder.call("/tokenize", {"content": "\n# alternate deterministic prefix restore regression\n",
                                             "add_special": False}, timeout=30)["tokens"]
        _require(bool(suffix), "empty alternate suffix")
        alternate = []
        for item in unequal:
            changed = dict(item)
            # With --ctx-checkpoints 1 the retained checkpoint is four tokens
            # before prompt end. A one-token branch can reuse that checkpoint;
            # branching 32 tokens earlier would demand an older checkpoint.
            replacement = next((token for token in suffix if token != item["fixed_prompt"][-1]), None)
            _require(replacement is not None, "alternate suffix has no distinct final token")
            changed["fixed_prompt"] = item["fixed_prompt"][:-1] + [replacement]
            alternate.append(changed)
        result["waves"].append(_wave(recorder, completion, output, "prefix_restore", alternate, 32, timeout,
                                     warm=True, require_concurrency=False))
        # Slot 2 has the shortest populated context, keeping fresh-oracle work bounded.
        prompt = alternate[2]["fixed_prompt"]
        body = _body(prompt, 2, 128, True, greedy=True)
        result["cancellation"] = _cancel_after_three(base_url, output, body, timeout)
        idle = _wait_idle(recorder, 2)
        _save(output / "cancel_idle_observations.json", idle)
        live = _record_completion(completion, output, "after_cancel_same_slot", _body(prompt, 2, 32, True, True), timeout, True)
        _require(live["ok"], "same-slot request after cancellation failed")
        result["after_cancel_healthy"] = True
        if cancellation_oracle:
            fresh = _record_completion(completion, output, "after_cancel_fresh_reset", _body(prompt, 2, 32, False, True), timeout)
            _require(fresh["ok"], "fresh-reset cancellation oracle request failed")
            live_content = live["response"].get("content", "")
            fresh_content = fresh["response"].get("content", "")
            _require(bool(live_content) and bool(fresh_content), "oracle requires aggregated nonempty streaming content")
            comparison = {"same_slot": 2, "sampling": "greedy", "predicted_n": 32,
                          "exact_content_match": live_content == fresh_content,
                          "live_sha256": hashlib.sha256(live_content.encode()).hexdigest(),
                          "fresh_sha256": hashlib.sha256(fresh_content.encode()).hexdigest()}
            _save(output / "cancellation_oracle.json", comparison)
            result["cancellation_oracle"] = comparison
            _require(comparison["exact_content_match"], "same-slot output differs from fresh-reset greedy oracle")
        else:
            result["cancellation_oracle"] = {"status": "skipped_by_caller", "verified": False}
        _require(recorder.call("/health", timeout=10).get("status") == "ok", "final health check failed")
        result["status"] = "complete"
        return result
    except Exception as exc:
        result.update(status="failed", error=repr(exc))
        raise
    finally:
        _save(output / "result.json", result)
