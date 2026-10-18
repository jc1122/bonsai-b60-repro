# What the measurements establish

## Fixed workload and timing

The full-context result used three independent token arrays of 258,048 input
tokens each, allocated in three 262,144-token slots. Synthetic Python repository
files carry different deterministic constants for each request. The task asks for
a long test module. This supplies a repeatable coding-shaped generation load;
output is not compiled or scored for this capacity test.

The original `server_workload.py` is included unchanged. It uses the server's
chat template and tokenizer, and inserts tokenized deterministic filler until
each input has exactly the requested token count. The benchmark wrapper supplies
HTTP/SSE transport and Docker residency monitoring. The source generator avoids
private repositories, documents or conversations.

Each request uses a fixed slot, `ignore_eos=true`, a forced 1,024 output tokens,
seed `1100 + slot`, temperature 0.1, top-p 0.95, top-k 50, min-p 0 and repetition
penalty 1. Reasoning is enabled in the rendered template. Counts/rates include
reasoning tokens; they are not an answer-only speed measure.

Three worker threads begin at a barrier. `/slots` is polled during the wave and
must show all three slots processing simultaneously. Each slot runs the same
synthetic input again in two warm waves. Warm validation requires `cache_n >=
input_tokens - 2048` and `prompt_n <= 2048`: limited recurrent checkpoint replay
is permitted, full re-ingestion is not.

Aggregate rate is the sum of actual generated tokens divided by wall time from
before launching the three workers until all responses and polling complete.
It is not the sum of isolated per-request server timers. The published warm mean
is the arithmetic mean of the two complete warm-wave rates. Dividing by three
provides a mean session share, not a fairness or latency guarantee.

| Phase | Output tokens | Complete wall time | Aggregate tokens/s |
|---|---:|---:|---:|
| Cold | 3,072 | 2,476.5787 s | 1.2404 |
| Warm 1 | 3,072 | 57.5143 s | 53.4128 |
| Warm 2 | 3,072 | 55.7451 s | 55.1080 |

The 54.2604 tokens/s headline therefore assumes already-ingested histories and
working prefix reuse. It does not imply fast loading of new 258K jobs. A coding
agent that substantially rewrites an earlier prefix may incur more replay or
prefill work than this repeated-input test.

## Memory accounting

The original run collected 1,272 resource samples. Maximum observed process
resident VRAM was 25,260,539,904 bytes against measured physical VRAM of
25,669,140,480 bytes, leaving 408,600,576 bytes or 389.671875 MiB. Maximum
`drm-resident-system` was zero. Host MemAvailable stayed above 27,743,105,024 bytes.

`/proc/1/fdinfo` inside the container supplies the xe DRM accounting. Repeated file
descriptors are deduplicated by DRM client. GTT allocations can still exist; zero
resident System is the particular measured field, not a claim of zero host RAM
use or zero traffic across PCIe. Sampling is not an exhaustive trace of every
transient allocation or unrelated GPU client.

The included Level Zero preload guard limits this process's device and physical
allocations to 24,696,061,952 bytes. It is allocation accounting, not a reservation
of all remaining VRAM, and does not account for every kind of allocation by every
process. Actual residency must still be observed. The original guard source is
included with a fake-driver concurrency/accounting regression.

The reproduction monitor checks residency and a 4 GiB host RAM floor. It never
stops or changes another server. A failed request or measurement is a failed run,
not a throughput datapoint. Keep unrelated GPU workloads out of a near-full test.

## Compact-state correctness

In the three-cell MTP4 layout, recurrent allocation changed from 994,443,264 to
598,302,720 bytes: 377.7890625 MiB saved. The patch separates dense replay record
banks from padded recurrent state. It does not change the SYCL arithmetic kernels.

Historical qualification included 252 logical state/checkpoint comparisons across
Q4 and F16, four complete 133-comparison forward pairs, cache/cancellation tests,
and the populated full-context run. The CPU test here re-exercises 43,913 layout
and serialization assertions with CPU buffers under ASan/UBSan. It is explicitly
an adapter, not a GPU math proof. These scopes must not be conflated.

The published source patch is opt-in and narrowly qualified for the tested
three-cell geometry. A fallback for other layouts does not qualify a new session
count. Extending the model beyond its native 262,144 window is also outside this
experiment. Earlier replay/draft-window changes helped the final fit; the entire
capacity increase cannot be attributed solely to compaction.

## Interpretation limits

This is one hardware/runtime setup, two complete warm full-context waves and a
synthetic workload. It is neither a statistical speed record nor a long-duration
multi-user soak, and it does not measure scored long-context coding quality.
The separate 603-task quality suite and later negative kernel findings have their
own profiles. They must not be silently pooled with this result.

Published short-prompt Bonsai rates and n-gram-assisted edits can be much faster.
Compare actual occupied context, model artifact, draft policy, concurrency,
sampling and cold/warm status before calculating a percentage advantage.
