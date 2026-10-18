# Upstream work and comparable reports

- [Torchit1 Intel runtime](https://github.com/Torchit1/llama.cpp) and
  [B580 guide](https://github.com/Torchit1/llama.cpp/blob/arc-b580/docs/bonsai-arc-b580.md):
  existing ternary/XMX, attention, Q4-cache and speculative-decoding work. Short
  code/rename/edit rates are workload-dependent and should not be compared
  directly with three nearly full contexts.
- [sudoingx's MTP GGUF](https://huggingface.co/sudoingx/Ternary-Bonsai-2-27B-PTQ1_0-MTP-GGUF):
  source of the pinned public lean artifact; the model card also reports context
  depth and concurrency measurements on NVIDIA cards, using different artifacts.
- [NInfer practitioner report](https://www.reddit.com/r/LocalLLM/comments/1wqab4z/ternary_bonsai_2_27b_at_up_to_532_toks_on_one_rtx/):
  earlier three-lane serving and a large shared pool on RTX 4090. The advertised
  aggregate rate does not establish three simultaneously filled 262K histories.
- [Independent three-slot comment](https://www.reddit.com/r/LocalLLM/comments/1wqab4z/ternary_bonsai_2_27b_at_up_to_532_toks_on_one_rtx/pc3qa10/):
  configured three maximum-context slots, explicitly measured quoted speeds
  with one request.
- [Intel Arc discussion](https://www.reddit.com/r/IntelArc/comments/1wzdgu5/bonsai_2_27b_on_a_12_gb_arc_b580_128k_context/):
  practical context-fit discussion and a request for measurements on other cards.

These references were inspected in October 2026. We do not claim first
three-session serving, universal hardware superiority, or state-of-the-art speed.
