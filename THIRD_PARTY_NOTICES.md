# Third-party notices

The patches modify MIT-licensed llama.cpp/ggml source. Copyright and license
notices from those authors are retained in `UPSTREAM_LICENSE`. Patched source is
downloaded from the pinned Torchit1 repository rather than vendored wholesale.

`guard/include/level_zero/ze_api.h` is the Intel Level Zero v1.12-r1.12.15 API header,
Copyright (C) 2019-2021 Intel Corporation, SPDX-License-Identifier: MIT. Its notice
is retained in the file; the MIT terms are reproduced in `LICENSE`.

The model and downloaded Intel runtime/compiler/container binaries retain their
publishers' licenses. The model repository currently identifies Apache-2.0;
consult its pinned revision and the original upstream model notices. No weights,
compiler binaries, runtime packages or container image blobs are redistributed
in this repository.
