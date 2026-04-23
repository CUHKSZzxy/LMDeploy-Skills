---
name: code-navigation
description: Use when you need to locate a specific subsystem, module, or key file in the LMDeploy codebase — covers pytorch backend, vl/, serve/, and top-level orchestration files.
---

# LMDeploy Project Structure

> Valid as of: LMDeploy main, 2026-04-23. Re-run `tree lmdeploy/ -L 2` to check for drift.

```text
lmdeploy/
├── cli/                        # Command line interface implementations
├── lib/                        # Shared libraries/binary assets
├── lite/                       # Quantization Toolkit
│   ├── apis/                   # Calibration, AWQ, and SmoothQuant entry points
│   ├── modeling/               # GPTQ/quantized model specific logic
│   ├── quantization/           # Scaling calculation (activations/weights)
│   └── utils/                  # Quantization helper functions (cal_qparams.py)
├── metrics/                    # Statistics and performance monitoring
├── monitoring/                 # Monitoring configs (Docker/Grafana)
├── pytorch/                    # PyTorch inference backend
│   ├── adapter/                # LoRA and adapter logic
│   ├── backends/               # Kernel/Operator Dispatchers (FP8, AWQ, CUDA)
│   ├── check_env/              # Environment/GPU capability sanity checks
│   ├── configurations/         # Per-model engine configurations (Llama, etc.)
│   ├── devices/                # Device management (CUDA)
│   ├── disagg/                 # Disaggregated prefill/decode logic
│   ├── engine/                 # Main Scheduler and Execution Loop
│   ├── kernels/                # Triton/CUDA Kernels (w8a8_triton_kernels.py)
│   ├── models/                 # Model Patches: Replacing HF layers with kernels
│   │   └── utils/model.py      # Shared helpers: get_multimodal_mask, DeployModelMixinV1
│   ├── multimodal/             # Multi-modal input types for Pytorch engine
│   ├── nn/                     # Reusable PyTorch modules
│   ├── paging/                 # PagedAttention: KV cache block management
│   ├── spec_decode/            # Speculative decoding logic
│   ├── strategies/             # Execution and dispatch strategies
│   ├── third_party/            # External dependencies/repos
│   ├── tools/                  # Internal engine debugging tools
│   ├── transformers/           # HF Transformers integration depth
│   └── weight_loader/          # Sharded/quantized weight loading engine
├── serve/                      # Serving: OpenAI-compatible API and gRPC
│   ├── core/async_engine.py    # Async inference loop, request logging
│   └── processors/multimodal.py  # Old/new preprocess dispatch (_uses_new_preprocess)
├── turbomind/                  # C++ TurboMind inference backend
├── vl/                         # Vision-Language (VL) Support and Image Processing
│   ├── constants.py            # Modality enum (IMAGE/VIDEO/AUDIO/TIME_SERIES)
│   ├── engine.py               # VL engine: apply_chat_template (absorbs has_input_ids)
│   ├── media/                  # Image/Video/Audio loaders and base classes
│   └── model/                  # VLM preprocessing per model family
│       ├── base.py             # VisionModel base: preprocess, get_override_size,
│       │                       #   get_expanded_input_ids, MultimodalSpecialTokens
│       │                       #   ATTR_NAME_TO_MODALITY registry, VISION_MODELS registry
│       └── qwen3.py            # Reference new-style VLM implementation (Qwen3-VL)
├── api.py                      # High-level entry for model interaction
├── archs.py                    # Registry: Maps architectures to runtime patches
├── messages.py                 # Core Types: GenerationConfig, EngineConfig
├── model.py                    # Chat Templates: CRITICAL for conversation logic
├── pipeline.py                 # Main Orchestrator: Engine + Tokenizer
└── tokenizer.py                # Wrapper for HF/SentencePiece tokenizers
```
