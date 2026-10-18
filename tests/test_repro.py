"""Offline regressions for workload validation and reported measurement arithmetic."""
import io
import json
import pathlib
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import contextlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from common import ROOT, sha256, versions
from benchmark import Client, drm_residency
from launch import command, ENVIRONMENT
import benchmark
import server_workload as workload


class CompletionStream:
    def __init__(self, rows):
        self.rows = [b"data: " + json.dumps(x).encode() + b"\n" for x in rows]
    def __enter__(self): return iter(self.rows)
    def __exit__(self, *args): return False


class ReproTests(unittest.TestCase):
    def test_locked_patch_integrity(self):
        for name, digest in versions()["patches"].items():
            self.assertEqual(sha256(ROOT / name), digest)

    def test_guard_header_integrity(self):
        self.assertEqual(sha256(ROOT / "guard/include/level_zero/ze_api.h"),
                         versions()["level_zero_header"]["sha256"])

    def test_launch_profile(self):
        argv = command(pathlib.Path("/model.gguf"), 262144, 18080, "repro", "image")
        self.assertEqual(argv[argv.index("-c") + 1], "786432")
        self.assertEqual(argv[argv.index("-np") + 1], "3")
        self.assertIn("127.0.0.1:18080:8080", argv)
        self.assertEqual(ENVIRONMENT["LLAMA_GDN_REPLAY_COMPACT"], "1")
        self.assertNotIn("--privileged", argv)

    def test_unsupported_geometry_rejected(self):
        with self.assertRaises(RuntimeError): command(pathlib.Path("/x"), 362144, 18080, "repro", "image")

    def test_padded_control_is_limited_to_small_profile(self):
        argv = command(pathlib.Path("/x"), 16384, 18080, "repro", "image", compact=False)
        self.assertIn("LLAMA_GDN_REPLAY_COMPACT=0", argv)
        with self.assertRaises(RuntimeError):
            command(pathlib.Path("/x"), 262144, 18080, "repro", "image", compact=False)

    def test_credentials_in_origin_rejected(self):
        with self.assertRaises(RuntimeError): Client("http://user:password@localhost:8080")

    def test_api_path_rejected(self):
        with self.assertRaises(RuntimeError): Client("http://localhost:8080/v1")

    def test_stream_requires_done_event(self):
        client = Client("http://127.0.0.1:8080")
        with patch("urllib.request.urlopen", return_value=CompletionStream([{"content": "partial"}])):
            with self.assertRaisesRegex(RuntimeError, "terminal"):
                client.completion({}, 1)

    def test_stream_preserves_pieces(self):
        rows = [{"content": "a", "stop": False}, {"content": "b", "stop": True, "timings": {"predicted_n": 2}}]
        with patch("urllib.request.urlopen", return_value=CompletionStream(rows)):
            response = Client("http://127.0.0.1:8080").completion({}, 1)
        self.assertEqual(response["content"], "ab")
        self.assertEqual(response["final"]["timings"]["predicted_n"], 2)

    def test_stream_server_error_fails(self):
        with patch("urllib.request.urlopen", return_value=CompletionStream([{"error": "failure"}])):
            with self.assertRaisesRegex(RuntimeError, "SSE error"):
                Client("http://127.0.0.1:8080").completion({}, 1)

    def test_drm_duplicate_client_not_double_counted(self):
        block = "drm-driver: xe\ndrm-client-id: 42\ndrm-resident-vram0: 100 KiB\ndrm-resident-system: 0\n"
        self.assertEqual(drm_residency(block * 2)["resident_vram_bytes"], 102400)

    def test_drm_system_spill_is_visible(self):
        block = "drm-driver: xe\ndrm-client-id: 1\ndrm-resident-vram0: 10 KiB\ndrm-resident-system: 64 KiB\n"
        self.assertEqual(drm_residency(block)["resident_system_bytes"], 65536)

    def test_absent_gpu_evidence_fails(self):
        with self.assertRaises(RuntimeError): drm_residency("")

    def test_closed_non_gpu_descriptor_does_not_discard_valid_drm_evidence(self):
        class Descriptor:
            def __init__(self, text=None): self.text = text
            def read_text(self):
                if self.text is None: raise FileNotFoundError("descriptor closed between enumeration and read")
                return self.text
        class Directory:
            def iterdir(self):
                return iter([Descriptor(), Descriptor("pos: 0\ndrm-driver: xe\ndrm-client-id: 7\ndrm-resident-vram0: 64 KiB\n")])
        output = io.StringIO()
        with patch("pathlib.Path", return_value=Directory()), contextlib.redirect_stdout(output):
            exec(benchmark.FDINFO_SCRIPT, {})
        self.assertEqual(drm_residency(output.getvalue())["resident_vram_bytes"], 65536)

    def record(self, final, warm=False):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory)
            body = workload._body([1] * 4096, 1, 1024, warm)
            return workload._record_completion(lambda *_: final, output, "fixture", body, 1, warm)

    def test_short_output_fails(self):
        r = self.record({"content": "", "timings": {"predicted_n": 8}, "id_slot": 1})
        self.assertFalse(r["ok"])

    def test_wrong_slot_fails(self):
        r = self.record({"content": "", "timings": {"predicted_n": 1024}, "id_slot": 2})
        self.assertFalse(r["ok"])

    def test_truncation_fails(self):
        r = self.record({"content": "", "timings": {"predicted_n": 1024}, "id_slot": 1, "truncated": True})
        self.assertFalse(r["ok"])

    def test_warm_full_recompute_fails(self):
        r = self.record({"content": "", "timings": {"predicted_n": 1024, "cache_n": 0, "prompt_n": 4096}}, True)
        self.assertFalse(r["ok"])

    def test_warm_cached_completion_passes(self):
        r = self.record({"content": "", "timings": {"predicted_n": 1024, "cache_n": 4095, "prompt_n": 1}}, True)
        self.assertTrue(r["ok"])

    def test_reported_rates_recompute_from_raw_waves(self):
        result = json.loads((ROOT / "results/full-context.json").read_text())
        rates = []
        for phase in ["cold", "warm1", "warm2"]:
            wave = json.loads((ROOT / "results/waves" / (phase + ".json")).read_text())
            rows = [json.loads((ROOT / "results" / p).read_text()) for p in wave["request_files"]]
            self.assertEqual(len(rows), 3)
            self.assertEqual({r["request"]["id_slot"] for r in rows}, {0, 1, 2})
            self.assertTrue(all(r["ok"] and r["input_tokens"] == 258048 and r["predicted_n"] == 1024 for r in rows))
            recomputed = sum(r["predicted_n"] for r in rows) / wave["wall_s"]
            self.assertAlmostEqual(recomputed, wave["aggregate_tokens_per_s"], places=10)
            if phase != "cold": rates.append(recomputed)
        self.assertAlmostEqual(sum(rates) / 2, result["warm_aggregate_tok_s"], places=10)

    def test_reported_memory_recomputes_from_samples(self):
        rows = [json.loads(line) for line in (ROOT / "results/residency-samples.jsonl").read_text().splitlines()]
        result = json.loads((ROOT / "results/full-context.json").read_text())
        self.assertEqual(max(r["resident_vram_bytes"] for r in rows), result["max_resident_vram_bytes"])
        self.assertEqual(max(r["resident_system_bytes"] for r in rows), 0)
        self.assertEqual(result["physical_vram_bytes"] - result["max_resident_vram_bytes"],
                         result["physical_headroom_bytes_min"])

    def test_complete_three_wave_cli_with_synthetic_server(self):
        """Exercise prompt creation, nine concurrent requests, monitoring and saved CLI output."""
        class FakeClient:
            def __init__(self, base_url):
                self.active, self.calls, self.lock = 0, [], threading.Lock()
            def api(self, path, body=None, timeout=30):
                if path == "/slots":
                    with self.lock:
                        return [{"id": i, "n_ctx": 16384, "is_processing": i < self.active} for i in range(3)]
                if path == "/apply-template": return {"prompt": "prefix " + body["messages"][0]["content"] + " suffix"}
                if path == "/tokenize": return {"tokens": [ord(c) for c in body["content"]]}
                raise AssertionError(path)
            def completion(self, body, timeout):
                with self.lock:
                    self.active += 1
                    self.calls.append(body)
                try:
                    time.sleep(0.7)  # Leave time for the original occupancy poll.
                    cache = body["cache_prompt"]
                    return {"content": "synthetic", "final": {"stop": True, "id_slot": body["id_slot"],
                        "timings": {"predicted_n": body["n_predict"],
                                    "cache_n": len(body["prompt"]) - 1 if cache else 0,
                                    "prompt_n": 1 if cache else len(body["prompt"])}}}
                finally:
                    with self.lock: self.active -= 1
        client = FakeClient("unused")
        inspection = [{"Id": "synthetic-container", "NetworkSettings": {"Ports": {
            "8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "18080"}]}}}]
        sample = {"unix": 0, "resident_vram_bytes": 1024, "resident_system_bytes": 0,
                  "host_mem_available_bytes": 8 * 2**30, "unique_clients": 1}
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "run"
            argv = ["benchmark.py", "--output", str(output), "--output-tokens", "128"]
            with patch.object(sys, "argv", argv), patch.object(benchmark, "Client", return_value=client), \
                 patch.object(benchmark.subprocess, "check_output", return_value=json.dumps(inspection)), \
                 patch.object(benchmark, "resources", return_value=sample), contextlib.redirect_stdout(io.StringIO()):
                benchmark.main()
            report = json.loads((output / "result.json").read_text())
            self.assertTrue(report["success"])
            self.assertEqual(len(client.calls), 9)
            self.assertEqual([w["peak_active_slots"] for w in report["waves"]], [3, 3, 3])
            self.assertTrue(all(len(c["prompt"]) == 12288 for c in client.calls))
            self.assertEqual(sum(c["cache_prompt"] for c in client.calls), 6)
            self.assertFalse(report["coding_quality_scored"])


if __name__ == "__main__":
    unittest.main()
