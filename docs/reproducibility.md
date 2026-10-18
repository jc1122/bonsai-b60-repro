# Build and validation scope

## Source and artifact pins

`versions.json` identifies the public Torchit1 commit, model repository/revision,
GGUF size/hash, two container digests, six Intel package hashes, source patch
hashes and expected modified/unchanged source hashes. `prepare.py` refuses an
unexpected commit or tracked source edits rather than resetting a checkout.

The model is `Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf`, 6,297,658,848 bytes
(6.30 decimal GB / 5.87 GiB), SHA256
`1e33c571a5ce7a9a3e42474d66192923d5a6d77da7fb3a22986dc809522b5685`.
PTQ1_0 describes the model weights. Q4_0 describes the separate target/draft KV
caches in this serving profile.

The historical local container image ID is retained as provenance, not presented
as a public pullable image. The Dockerfile reconstructs its compiler, NEO/IGC and
loader dependencies from public inputs. It adds an API-only build workflow and
does not redistribute the historical binary assembly. Container IDs, compiler
output paths, CPU target selection and generated build metadata can change
binary hashes. `.work/build-receipt.json` records newly built binaries.

Two CPU workers are the conservative default. Source compilation may require
several gigabytes of RAM and substantial disk space for the pinned build images.
The vLLM image is used only as a source of three Level Zero loader/layer files;
this experiment runs llama.cpp, not vLLM.

## Serving profile

The launcher implements:

```text
-ngl 99 -c 786432 -np 3 -ub 1024 -b 2048
-ctk q4_0 -ctv q4_0 --flash-attn on --fit off
--no-context-shift --no-kv-unified --cache-ram 0 --ctx-checkpoints 1
--spec-type draft-mtp --spec-draft-n-max 4 -ctkd q4_0 -ctvd q4_0
```

It also sets the tested Intel T2/DPAS attention controls, draft ubatch 512,
replay-multi/compact on, and draft window zero. The exact environment is in
`scripts/launch.py`. Several controls belong to this fork/patch; they are not
portable promises for arbitrary upstream llama.cpp builds. `HARNESS_ORACLE=1`
is retained from the historical environment; diagnostic harness controls do not
turn this ordinary server benchmark into a raw-logit qualification.

The smaller profile changes allocated context to `3 × 16384`, retaining the
three-cell shape. Selecting another slot count is deliberately not exposed as
a supported switch. `--print-command` displays a launch without starting it.
`--compact off` selects the replay-only padded-layout control at 16K; use it
sequentially with the default compact profile to compare startup RS allocation.

## What the automated checks prove

The repository checks patch/header integrity, launch geometry, SSE completion
handling, invalid short/truncated/wrong-slot output, cache-reuse failures,
deduplicated DRM accounting and arithmetic reconstructed from the historical
records. A fake Level Zero driver tests the allocation guard without a GPU.

`check_state.py` builds ggml CPU and compiles the actual recurrent-memory source
with a test-only model-device adapter. The replay-only control fails at the
intended compact-layout assertion; the candidate passes 43,913 checks, including
layout/serialization behaviour. ASan and UBSan are enabled for the fixture.
GitHub's CPU job does not verify GPU kernel correctness or performance.

The recorded native Q4/F16 comparisons and measured full-context run occurred
before publication. Their selected results are included with original-record
hashes. Publishing this package does not itself repeat a 41-minute B60 run, and
the new CLI wrapper's offline tests are distinct from the historical native run.

## Reporting a reproduction

Include GPU model/physical memory, host/kernel/driver, `versions.json`, rebuilt
binary hashes, allocated and actual input lengths, forced output count, all
three wave times, concurrency/cache checks and complete resource samples.
Report failed attempts separately. Use a synthetic reproduction when opening an
issue; generated `runs/` may contain model output and local container metadata.

The repository intentionally excludes weights, package binaries, credentials,
machine inventory, router/client configurations, private source data and original
host journals. No installed production service is required to reproduce it.
