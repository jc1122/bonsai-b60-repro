# Bonsai2 on a 24 GB Intel Arc Pro B60

[![CPU checks](https://github.com/jc1122/bonsai-b60-repro/actions/workflows/validate.yml/badge.svg)](https://github.com/jc1122/bonsai-b60-repro/actions/workflows/validate.yml)

Reproduce a measured **three-session, nearly full 256K-context** Bonsai2 deployment.
This repository contains the pinned runtime patch, build/launch scripts, the original
synthetic workload, selected measurement records and CPU regression checks.

| Measured on 17 October 2026 | Result |
|---|---:|
| Allocated input-plus-output window | 3 × 262,144 tokens |
| Actual input, independently populated | 3 × 258,048 tokens |
| Generated output per wave | 3 × 1,024 tokens |
| Warm generation, two-wave mean | **54.26 tokens/s aggregate** |
| Aggregate divided by three | **18.09 tokens/s mean session share** |
| Cold ingestion plus generation, complete batch | **41 minutes 17 seconds** |
| Minimum observed physical VRAM headroom | **389.7 MiB** |
| Compact recurrent-state allocation saving | **377.8 MiB** |

The throughput figure is **warm prefix-cache reuse**, including bounded checkpoint
replay. It does not describe cold end-to-end throughput. The workload verifies fit,
generation, concurrency and cache use; it does not score coding accuracy at 256K.
See [measurement details](docs/methodology.md) and [machine-readable results](results/full-context.json).

## What changed

The base is [Torchit1/llama.cpp](https://github.com/Torchit1/llama.cpp), commit
`7adb2dd207bda1dda4510c498165666e9c51fe89` (`arc-b580-v0.10.1`), using
[sudoingx's public MTP-lean GGUF](https://huggingface.co/sudoingx/Ternary-Bonsai-2-27B-PTQ1_0-MTP-GGUF).
Their existing ternary/XMX, attention and speculative-decoding work supplies the
inference foundation. **Q4 KV caches, MTP and Intel matrix kernels are upstream work.**

Our opt-in patch extends recurrent rollback-by-replay to the tested three-slot
geometry and stores replay records separately from padded state planes. The compact
layout saved 396,140,544 bytes in the measured configuration. It changes libllama
bookkeeping, retaining the SYCL arithmetic backend. No isolated speed gain from
compaction is claimed. Compact storage activates only for exactly three eligible
cells and three maximum sequences; unsupported layouts retain padded storage.

All source, model, container-base and package pins are in [versions.json](versions.json).
The combined patch is [patches/replay-compact-v2.patch](patches/replay-compact-v2.patch).
The replay-only control is included for the CPU red-green regression.

## Requirements

- Linux x86-64 host with a working Intel **xe** driver and an Arc Pro B60 exposed at `/dev/dri`.
- Docker, Git and Python 3.10 or newer; CMake and a C++ compiler for the optional host CPU checks.
- Roughly 32 GB host RAM and adequate disk space for the Intel/vLLM build images, source and 6.3 GB model.
- An otherwise available GPU for the near-full profile. Our minimum margin was only 389.7 MiB;
  another GPU client or a different runtime can change whether it fits.

The tested host was a Ryzen 9 5950X with 32 GiB RAM, a B60 with 25,669,140,480
physical VRAM bytes, and a Gen4 ×8 connection. The user-space stack was oneAPI
2025.3.0, NEO 26.35.39758.10, IGC 2.41.5 and Level Zero loader 1.32.0.
[Hardware disclosures](results/hardware.json) describe observations, not separately
benchmarked tuning recommendations. These scripts do not alter firmware, drivers,
system services or CPU/GPU clock settings.

## Build and download

```bash
git clone https://github.com/jc1122/bonsai-b60-repro.git
cd bonsai-b60-repro
python3 scripts/download.py model
bash scripts/build.sh
```

The build downloads six checksum-verified public Intel packages and combines them
with digest-pinned public compiler and loader images. It checks out the exact public
runtime commit, applies the recorded patch, verifies affected source hashes and
builds with two CPU workers by default. **No GPU is exposed during compilation.**
`BUILD_JOBS=4 bash scripts/build.sh` changes build parallelism.

The resulting image is `bonsai-b60-repro:2025.3-neo26.35`. The API-only build disables
embedded web-UI provisioning so compilation works without network access inside
the build container. Fresh binary/image hashes can differ from the historical
assembly; this is a pinned source/environment recipe, not a bit-reproducible binary
distribution. See [build and validation scope](docs/reproducibility.md).

## First run: smaller context

In one terminal:

```bash
python3 scripts/launch.py --context 16384
```

Wait for the server to report ready. In another terminal:

```bash
python3 scripts/benchmark.py --context 16384 --output-tokens 128 --output runs/smoke
```

The launcher runs a separate foreground container, publishes only
`127.0.0.1:18080`, and mounts the model read-only. It does not replace or stop an
existing server. Stop this container with Ctrl+C in its terminal before launching
the larger profile. Each benchmark needs a fresh output directory.

## Full three-session reproduction

```bash
# Terminal 1; this reserves all three windows even before requests arrive.
python3 scripts/launch.py --context 262144

# Terminal 2, after the server is ready:
python3 scripts/benchmark.py --context 262144 --output runs/full-262144
```

Allow around 45 minutes or more on comparable hardware. The benchmark creates
three deterministic, distinct synthetic repository inputs using the server's
tokenizer, fills each to `context - 4096`, and submits concurrent requests with
fixed slot IDs. It performs one cold and two cached waves, each forcing 1,024 output
tokens. It rejects wrong counts, truncation, missing three-way occupancy, warm
full-prompt recomputation and failed residency measurements. Raw selected records,
slot polling and resource samples are written under the chosen output directory.

Use `/completion`, `/slots`, `/tokenize` and `/apply-template` directly; this test
does not need an API router or a coding-agent client. `--base-url` and `--container`
must refer to the same local Docker instance. Other GPUs/backends require adapting
and requalifying the geometry and telemetry checks.

## Checks and results

For evicted conversations, an optional 6 GiB host-RAM archive reduced a tested
100K resume from 157.7 s to 0.735 s with identical greedy output. The three GPU
slots and context allocation stay the same. Short-context steady throughput
averaged 1.6% lower, with overlapping ranges; this is a resume-latency benefit.
See [RAM-cache settings, measurements and reproduction](docs/ram-prompt-cache.md).
The default launcher preserves the original cache-disabled benchmark profile.

```bash
python3 -m unittest discover -s tests -v
python3 scripts/guard_test.py
python3 scripts/check_state.py --jobs 2
```

The last command builds CPU ggml and runs the actual recurrent-memory source
through a CPU-buffer adapter under ASan/UBSan. The replay-only control must fail
the compact-layout assertion; the patched source must pass 43,913 checks. This
does not replace the historical native Q4/F16 state/logit qualification.
GitHub CI runs these CPU checks; it does not run a B60 performance benchmark.

To inspect the recurrent allocation difference on hardware, use the smaller
profile sequentially with `--compact off` and `--compact on`. Wait for each
server to load, record its `RS buffer size` lines, then stop it before the next
launch. The corresponding main-model recurrent allocation was 948.375 MiB
padded and 570.586 MiB compact in our geometry. Other buffers also exist; compare
the corresponding model allocations, not a single total-GPU snapshot. The
launcher restricts this padded control to the smaller profile because the
near-full windows may exceed its memory budget without compaction.

Selected original [wave records](results/waves), [request records](results/requests)
and [residency samples](results/residency-samples.jsonl) let the offline tests
recalculate the headline rates and memory margin. [Provenance](results/provenance.json)
explains the field selection and retains original-record hashes.

Our separate 603-task suite scored Bonsai 512/603 and Qwen 521/603, with eight and
two output cutoffs respectively. Statistical equivalence was not established;
those results do not measure 256K coding quality. Q4/F16 bookkeeping comparisons
also do not isolate the quality cost of Q4 KV. [Quality summary](results/quality-summary.json).

Repaired M15 and TinyGDN showed no robust serving benefit beyond control drift;
ordinary active CPU-KV offloading was about 8.1× slower in an earlier probe.
[Negative results and scope](results/negative-results.json). We claim neither a
speed record nor the first three-slot Bonsai server.

## Credits and license

Thanks to Torchit1 and the ggml/llama.cpp authors, sudoingx for the public MTP-lean
artifact, PrismML for Bonsai, and Intel's runtime/compiler developers.
[Related practitioner reports](docs/references.md) helped guide the experiments.

Repository code and patch additions are MIT licensed; upstream notices are
retained in [UPSTREAM_LICENSE](UPSTREAM_LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Weights and Intel container/package binaries are downloaded from their original
publishers and remain subject to their respective licenses; they are not included
in this Git repository.
