import copy
import pathlib
import json
import statistics
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from cache_eviction import EvictionClient, check_pair, memory_gate
from launch import command


class CacheTests(unittest.TestCase):
    def test_published_cache_measurements_recompute(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        value = json.loads((root / "results/ram-prompt-cache.json").read_text())
        long = value["long"]
        self.assertAlmostEqual(long["control_resume_ttft_s"] / long["candidate_resume_ttft_s"], long["speedup"])
        self.assertEqual(long["cached_tokens"] + long["replayed_tokens"], long["input_tokens"])
        rates = value["warm_throughput"]; confirmation = rates["confirmation"]
        ctrl = rates["primary_control"] + confirmation["control_warm_values"]
        cand = rates["primary_candidate"] + confirmation["candidate_warm_values"]
        self.assertAlmostEqual(statistics.mean(ctrl), rates["pooled_control_mean"])
        self.assertAlmostEqual(statistics.mean(cand), rates["pooled_candidate_mean"])
        self.assertAlmostEqual(100 * (statistics.mean(cand) / statistics.mean(ctrl) - 1), rates["pooled_relative_change_percent"])
        self.assertEqual(value["resources"]["max_resident_system_bytes"], 0)
        self.assertEqual(value["resources"]["swap_start"]["pswpout"], value["resources"]["swap_end"]["pswpout"])
        self.assertTrue(value["adoption"]["restart_verified"])

    def pair(self):
        initial = {"tokens": list(range(128)), "content": "synthetic", "ttft_s": 158,
                   "final": {"timings": {"cache_n": 0, "prompt_n": 100352}}}
        resumed = copy.deepcopy(initial)
        resumed.update(ttft_s=.735, final={"timings": {"cache_n": 100348, "prompt_n": 4,
                                                    "draft_n": 119, "draft_n_accepted": 97}})
        return initial, resumed

    def test_cross_slot_hit(self):
        initial, resumed = self.pair()
        value = check_pair(initial, resumed, 100352, True)
        self.assertTrue(value["greedy_equivalent"])
        self.assertEqual(value["prompt_n"], 4)

    def test_hash_or_token_corruption_is_rejected(self):
        for key, value in [("content", "wrong"), ("tokens", list(range(127)) + [999])]:
            initial, resumed = self.pair(); resumed[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                check_pair(initial, resumed, 100352, True)

    def test_recomputation_is_not_a_cache_hit(self):
        initial, resumed = self.pair(); resumed["final"]["timings"].update(cache_n=0, prompt_n=100352)
        with self.assertRaises(RuntimeError): check_pair(initial, resumed, 100352, True)
        check_pair(initial, resumed, 100352, False)

    def test_resource_failure_cancels_future_requests(self):
        client = EvictionClient("http://127.0.0.1:18080"); client.abort()
        with self.assertRaisesRegex(RuntimeError, "cancelled"): client.completion({}, 1)

    def test_ram_profile_preserves_reproduction_default(self):
        args = (pathlib.Path("/model.gguf"), 262144, 18080, "repro", "image")
        original = command(*args); cached = command(*args, prompt_cache="ram6")
        self.assertEqual(original[original.index("--cache-ram") + 1], "0")
        self.assertEqual(cached[cached.index("--cache-ram") + 1], "6144")
        self.assertEqual(cached[cached.index("--ctx-checkpoints") + 1], "1")
        self.assertEqual(cached[-1], "--no-cache-idle-slots")
        cached[cached.index("--cache-ram") + 1] = "0"; cached.pop()
        self.assertEqual(cached, original)

    def test_ram_monitor_keeps_swap_and_spill_failures(self):
        base = {"pswpin": 2, "pswpout": 3}
        row = {"resident_system_bytes": 0, "resident_vram_bytes": 25000000000,
               "host_mem_available_bytes": 20 * 2**30, "process_swap_bytes": 0,
               "swap_counters": dict(base), "memory_PSI_avg10": {"some": 0, "full": 0}}
        memory_gate(row, base)
        for key, value in [("resident_system_bytes", 873529344), ("process_swap_bytes", 33906688),
                           ("swap_counters", {"pswpin": 2, "pswpout": 4}),
                           ("memory_PSI_avg10", {"some": 1.63, "full": 1.63}),
                           ("memory_PSI_avg10", {})]:
            altered = copy.deepcopy(row); altered[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError): memory_gate(altered, base)


if __name__ == "__main__": unittest.main()
