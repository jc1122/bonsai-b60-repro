# Host-RAM prompt caching

The optional `ram6` profile archives evicted conversation state in **6 GiB of
host RAM**. Active inference keeps its existing GPU allocation: three windows of
262,144 tokens, Q4 KV, compact recurrent storage and MTP4. This is conversation
restoration; it does not compute new prompts on CPU or add active GPU slots.

On 2026-10-18, a synthetic 100,352-token conversation resumed after replacement
of all three slots in **0.735 s to first token**, versus **157.701 s** with RAM
caching disabled. It restored into a different slot, reused 100,348 tokens and
replayed four. The 128 generated tokens and output text matched the fresh run
exactly. MTP continued accepting draft tokens. The shorter 12,288-token case
resumed in 0.272 s versus 13.393 s.

These are finite greedy checks. They do not establish coding accuracy at 256K
or guarantee identical outputs for every prompt.

The profile was adopted after qualification. Service restart, all three window
allocations, unchanged mapped binaries, clean kernel checks and a routed
Responses API request passed. The public eviction script below also ran on the
real server at 12K, resuming in 0.265 s with identical output and no resource
errors. The 100K figures came from the deployment harness using the same model,
settings and populated-prompt generator. A rollback to the preceding
cache-disabled 262K profile was prepared and its live dry run passed.

## Reproduce eviction and restoration

Use a separate server with no other clients. The launcher does not manage system
services. Keep the original full-context throughput recipe with its default
cache settings when reproducing the published 54.26 tokens/s result.

```bash
# Terminal1: control, with the already prepared model/runtime
python3 scripts/launch.py --context 262144 --prompt-cache off

# Terminal2, after startup
python3 scripts/cache_eviction.py --input-tokens 100352 --output runs/eviction-control
```

Stop that foreground container with Ctrl+C, then launch the candidate:

```bash
# Terminal1
python3 scripts/launch.py --context 262144 --prompt-cache ram6

# Terminal2
python3 scripts/cache_eviction.py --input-tokens 100352 --expect-reuse --output runs/eviction-ram6
```

The candidate adds `--cache-ram 6144 --ctx-checkpoints 1
--no-cache-idle-slots`. It saves the selected slot when replacing a conversation,
without eagerly duplicating other idle slots that still retain state on GPU.
The launcher requires at least 16 GiB available host RAM before starting this
profile. For a faster first check, use `--input-tokens 12288`.

The test sends A on slot0, replaces slots0/1/2 with distinct 12K B/C/D requests,
then resumes A on slot1. B/C/D are deliberately smaller than A. Exact requests,
generated token IDs, timings, pre-resume slot contents and resource samples are
saved locally. The script checks output equality, bounded prefix replay, MTP,
six GiB available host RAM, no inference-process swap, no new host swap-outs,
zero memory-pressure averages and zero GPU resident System memory. A resource
failure cancels its own HTTP stream. It does not stop or replace the server.
Full kernel-journal/MCE validation and recovery remain deployment-specific.

## Throughput and memory

To measure short inputs while retaining the full GPU window allocation:

```bash
python3 scripts/benchmark.py --context 262144 --input-tokens 12288 \
  --output-tokens 512 --output runs/ram6-throughput
```

Our five warm waves per profile averaged 143.31 tokens/s for the control and
140.98 for the candidate, **1.6% lower**. Ranges overlapped: 138.22–145.05 and
137.53–144.95. A fixed three-wave confirmation had medians of 143.85 and 138.37
tokens/s, **3.8% lower**. Both comparisons passed the predeclared 5% regression
screen; a small steady throughput cost remains possible. We claim a large
eviction-latency benefit, not a generation-speed increase.

The successful long-input trial retained at least 23.15 GiB available RAM,
reported zero GPU resident System memory, zero inference-process swap and no
new host swap-outs. Historical swap-ins were recorded separately. See the
[measurement record](../results/ram-prompt-cache.json) for exact values and scope.

The first 8 GiB/default-idle-cache candidate with eight checkpoints failed during
100K eviction: it acquired 833 MiB GPU resident System memory, 32.3 MiB process
swap and memory pressure. It was aborted and the original profile restored.
There was no container RAM cap or new MCE; the underlying cause remains
unresolved. Reducing the archive, retaining one checkpoint and disabling eager
idle copies avoided that failure in the successful trial. These changes were
tested together, so we do not attribute the fix to one flag alone.

## Checkpoints, batch jobs and limits

Active-slot checkpoints consume RAM outside the archive budget. Observed
checkpoints were roughly 195–292 MiB in this build. At the larger observed size,
32 checkpoints across three slots would use about 27 GiB before the archive or
other host processes. Do not copy the default checkpoint count blindly.

One checkpoint was sufficient for the tested repeated prompt and its four-token
replay. It does **not** guarantee reuse of older edits or branches: a checkpoint
must precede the point where the new prompt diverges. An identical repository
prefix can help batch jobs when such a checkpoint is retained. More checkpoints
and an explicit prefix-warming strategy need their own RAM and correctness tests.

The archive has finite capacity. A larger state can evict older entries; an entry
larger than the budget is skipped. This trial does not prove that three full 262K
conversations can all be archived simultaneously. The windows are still shared
by only three active GPU slots.

CPU preparation of queued requests is a separate possibility: run their prefill
on CPU, then hand off compatible attention KV, recurrent state and draft state.
Loading/tokenizing text alone does not compute those tensors. This fork has no
ready-made CPU-prefill-to-GPU queue, and its ordinary slot-export path saves only
the target context, so it is not a verified MTP handoff. No CPU-prefill speed or
cross-backend correctness result is claimed here.

For the distinction between prefix reuse, checkpoints and archive capacity, see
the upstream [Bonsai prompt-cache guide](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/PROMPT-CACHE.md).
Its experimental speculative path differs from this pinned Torchit MTP runtime.
