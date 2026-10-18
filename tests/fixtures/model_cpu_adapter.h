#pragma once

#include "llama-arch.h"
#include "llama-hparams.h"
#include "ggml-backend.h"

// Test-only model adapter. The reported SYCL registration uses CPU buffers.
class llama_model {
public:
    llm_arch arch = LLM_ARCH_QWEN35;
    llama_hparams hparams = {};
    ggml_backend_dev_t device = nullptr;

    ggml_backend_dev_t dev_layer(int) const { return device; }
};
