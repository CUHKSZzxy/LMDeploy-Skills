---
name: support-new-model
description: Add a new LLM or VLM to LMDeploy's PyTorch backend.
---

# Tutorial: Adding a New Model to LMDeploy (PyTorch Backend)

This guide walks through adding a new LLM or VLM to LMDeploy's PyTorch backend.

______________________________________________________________________

## Before Writing Any Code

**Study the reference implementations before touching any files.**

1. Read the HF model's `config.json` to understand: `model_type`, `architectures`, layer counts, hidden dims, number of attention heads, MoE parameters (if applicable).
2. Identify which category the model falls into:
   - **LLM only** — pure text model
   - **VLM** — text + vision (needs an additional preprocessor in `vl/model/`)
3. Find the closest existing model in LMDeploy and read it thoroughly:

| Reference model        | File(s)                                                                                                                                   |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| LLM (dense)            | `lmdeploy/pytorch/models/qwen3.py`                                                                                                        |
| LLM (MoE)              | `lmdeploy/pytorch/models/qwen3_moe.py`                                                                                                    |
| VLM preprocessor       | `lmdeploy/vl/model/qwen3.py`                                                                                                              |
| VLM (composite config) | `lmdeploy/pytorch/models/qwen3_omni_moe_thinker.py` + `lmdeploy/pytorch/configurations/qwen3_omni.py` + `lmdeploy/vl/model/qwen3_omni.py` |

______________________________________________________________________

## Key Files Quick Reference

| File                                         | Purpose                                                         |
| -------------------------------------------- | --------------------------------------------------------------- |
| `lmdeploy/pytorch/models/<model>.py`         | Attention, MLP, DecoderLayer, Model, ForCausalLM                |
| `lmdeploy/pytorch/models/module_map.py`      | HF class name → LMDeploy class path mapping                     |
| `lmdeploy/pytorch/configurations/<model>.py` | Config builder — only needed for non-standard/nested HF configs |
| `lmdeploy/vl/model/<model>.py`               | VLM: image/video preprocessing *(VLM only)*                     |
| `lmdeploy/vl/model/base.py`                  | `VisionModel` base class + `VISION_MODELS` registry             |
| `lmdeploy/archs.py`                          | VLM: arch name → task mapping *(VLM only)*                      |

______________________________________________________________________

## Step-by-Step: LLM (PyTorch Backend)

### Step 1 — Create the PyTorch model file

**File:** `lmdeploy/pytorch/models/<model_name>.py`

Implement the following class hierarchy (innermost → outermost):

1. **`<Model>Attention`** — QKV projection, rotary embedding, attention forward
2. **`<Model>MLP`** — gate-up linear, activation, down projection
3. **`<Model>DecoderLayer`** — wraps Attention + MLP with layer norms and residual connections
4. **`<Model>Model`** — embedding table, all decoder layers, final norm, rotary embedding
5. **`<Model>ForCausalLM`** — top-level class; inherits `nn.Module`, `DeployModelMixinV1`, `CudaGraphMixin`

**Required imports:**

```python
import torch
import torch.nn as nn
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import (ApplyRotaryEmb, Attention, RMSNorm, SiluAndMul,
                                  build_rotary_embedding_from_config)
from lmdeploy.pytorch.nn.linear import (build_down_linear, build_gateup_linear,
                                         build_o_proj, build_qkv_proj)
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight
from .patch import add_prefix
from .utils.cudagraph import CudaGraphMixin
from .utils.model import DeployModelMixinV1, build_embedding
```

**Attention skeleton:**

```python
class MyModelAttention(nn.Module):
    def __init__(self, config, dtype=None, device=None, prefix=''):
        super().__init__()
        self.qkv_proj = build_qkv_proj(
            config.hidden_size,
            num_q_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_size=config.hidden_size // config.num_attention_heads,
            bias=False,
            dtype=dtype, device=device, prefix=add_prefix('qkv_proj', prefix))
        self.apply_rotary_pos_emb = ApplyRotaryEmb()
        self.attn_fwd = Attention(
            config.num_attention_heads,
            config.hidden_size // config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads)
        self.o_proj = build_o_proj(
            config.num_attention_heads,
            config.hidden_size // config.num_attention_heads,
            config.hidden_size,
            bias=False,
            dtype=dtype, device=device, prefix=add_prefix('o_proj', prefix))

    def forward(self, hidden_states, rotary_pos_emb, past_key_value, attn_metadata):
        qkv_states = self.qkv_proj(hidden_states)
        # split q, k, v; apply rotary; call attn_fwd; project output
        ...
```

**MLP skeleton:**

```python
class MyModelMLP(nn.Module):
    def __init__(self, config, dtype=None, device=None, prefix=''):
        super().__init__()
        self.gate_up_proj = build_gateup_linear(
            config.hidden_size, config.intermediate_size,
            bias=False, dtype=dtype, device=device,
            prefix=add_prefix('gate_up_proj', prefix))
        self.down_proj = build_down_linear(
            config.intermediate_size, config.hidden_size,
            bias=False, dtype=dtype, device=device,
            prefix=add_prefix('down_proj', prefix))
        self.act_fn = SiluAndMul()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))
```

**ForCausalLM skeleton (critical fields):**

```python
class MyModelForCausalLM(nn.Module, DeployModelMixinV1, CudaGraphMixin):
    # Maps packed param name → list of original HF param suffixes
    packed_modules_mapping = {
        'qkv_proj': ['q_proj', 'k_proj', 'v_proj'],
        'gate_up_proj': ['gate_proj', 'up_proj'],
    }

    def __init__(self, config, ctx_mgr=None, prefix='', **kwargs):
        super().__init__()
        self.model = MyModelModel(config, ...)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.ctx_mgr = ctx_mgr

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(self, input_ids, inputs_embeds, past_key_values, attn_metadata, **kwargs):
        hidden_states = self.model(input_ids, inputs_embeds, past_key_values, attn_metadata)
        return hidden_states

    def get_logits(self, hidden_states):
        return self.lm_head(hidden_states)

    # prepare_inputs_for_generation and load_weights: copy from qwen3.py,
    # update stacked_params_mapping to match this model's HF weight names.
```

______________________________________________________________________

### Step 2 — Register in `module_map.py`

**File:** `lmdeploy/pytorch/models/module_map.py`

Add an entry to `MODULE_MAP`. The key is the exact HF architecture class name from `config.json`'s `architectures` field:

```python
MODULE_MAP.update({
    'MyModelForCausalLM': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.my_model.MyModelForCausalLM',
})
```

______________________________________________________________________

### Step 3 — Add config builder (if needed)

**File:** `lmdeploy/pytorch/configurations/<model_name>.py`

**Skip this step** for models with a standard flat HF config — `DefaultModelConfigBuilder` handles them automatically.

Only create this file when the HF config is non-standard, e.g.:

- Nested config (e.g., Qwen3-Omni has `hf_config.thinker_config.text_config`)
- Unusual `model_type` that needs special field remapping

```python
from .builder import AutoModelConfigBuilder, DefaultModelConfigBuilder

class MyModelConfigBuilder(AutoModelConfigBuilder):
    @classmethod
    def condition(cls, hf_config):
        # Must match model_type from config.json exactly
        return hf_config.model_type == 'my_model'

    @classmethod
    def build(cls, hf_config, model_path=None, **kwargs):
        # Extract the text config if nested; patch fields if needed
        cfg = DefaultModelConfigBuilder.build(hf_config, model_path, **kwargs)
        cfg.hf_config = hf_config  # keep full config for VLM layers
        return cfg
```

Auto-discovery: subclasses of `AutoModelConfigBuilder` register themselves automatically via `__init_subclass__()` — no import needed elsewhere.

______________________________________________________________________

## Step-by-Step: VLM (additional steps)

### Step 4 — Create the VL preprocessor

**File:** `lmdeploy/vl/model/<model_name>.py`

There are two styles. Choose based on whether the model needs mixed-modality support.

---

#### New-style (recommended for new models — supports mixed image+video)

Inherit from `VisionModel` and override only `build_preprocessor` and `apply_chat_template`. The base class provides `preprocess(messages, input_prompt, mm_processor_kwargs)` which handles all modality collection, HF processor dispatch, and per-modality output items automatically.

```python
from lmdeploy.vl.model.base import VISION_MODELS, VisionModel
from lmdeploy.vl.model.base import MultimodalSpecialTokens

@VISION_MODELS.register_module()
class MyModelVLModel(VisionModel):
    _arch = ['MyModelForConditionalGeneration']

    def build_preprocessor(self):
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        tokenizer = self.processor.tokenizer
        # Set token IDs for each modality the model supports
        self.mm_special_tokens = MultimodalSpecialTokens(
            image_token_id=tokenizer.convert_tokens_to_ids('<image>'),
            video_token_id=tokenizer.convert_tokens_to_ids('<video>'),  # if supported
        )
        self.image_token_id = self.mm_special_tokens.image_token_id

    def apply_chat_template(self, messages, chat_template, sequence_start, chat_template_kwargs=None):
        # Tokenise the message into input_ids using the model's own tokenizer
        # Return a list of token IDs (not a string)
        ...
```

The engine detects new-style automatically: `_uses_new_preprocess = 'input_prompt' in sig and 'mm_processor_kwargs' in sig`.

Reference implementation: `lmdeploy/vl/model/qwen3.py`.

---

#### Old-style (for models without mixed-modality needs / backward compat)

Override `preprocess(self, messages)` directly. Called before `wrap_for_pytorch`.

```python
@VISION_MODELS.register_module()
class MyModelVLModel(VisionModel):
    _arch = ['MyModelForConditionalGeneration']

    def build_preprocessor(self): ...

    def preprocess(self, messages):
        # Process images, return updated messages with pixel_values attached
        ...
```

---

Auto-registered via `@VISION_MODELS.register_module()`. **Add an explicit import** in `lmdeploy/vl/model/builder.py`:

```python
from .my_model import MyModelVLModel  # noqa F401
```

______________________________________________________________________

### Step 5 — Register VLM arch in `archs.py`

**File:** `lmdeploy/archs.py`

Add the architecture name to the `supported_archs` set inside `check_vl_llm()` so the engine routes the model through the VLM code path:

```python
# lmdeploy/archs.py — inside check_vl_llm()
supported_archs = set([
    ...
    'MyModelForConditionalGeneration',  # add this line
])
```

______________________________________________________________________

## Checklist

**LLM (PyTorch backend):**

- [ ] `pytorch/models/<model>.py` — all 5 classes implemented (`Attention`, `MLP`, `DecoderLayer`, `Model`, `ForCausalLM`)
- [ ] `module_map.py` — HF architecture class name registered
- [ ] `packed_modules_mapping` matches HF parameter naming scheme
- [ ] `stacked_params_mapping` in `load_weights()` has correct shard indices
- [ ] `pytorch/configurations/<model>.py` — added only if HF config is non-standard
- [ ] Weights load cleanly from HF checkpoint (no missing/unexpected key errors)

**VLM (additional):**

- [ ] `vl/model/<model>.py` — `build_preprocessor` implemented; chose new-style (base `preprocess`) or old-style (override `preprocess(messages)`)
- [ ] New-style: `MultimodalSpecialTokens` populated with correct token IDs for each supported modality
- [ ] New-style: `apply_chat_template` returns token ID list (not string)
- [ ] `_arch` matches `config.json` `architectures[0]` exactly
- [ ] `image_token_id` correctly resolved from the tokenizer (not None)
- [ ] `vl/model/builder.py` — explicit import added for new model
- [ ] `archs.py` entry added

______________________________________________________________________

## Common Pitfalls

1. **Weight name mismatches** — `packed_modules_mapping` keys must match HF param name suffixes exactly. Check actual HF weight names with `list(model.state_dict().keys())[:20]` before coding.
2. **Wrong shard index order** — `stacked_params_mapping` for QKV must follow Q→0, K→1, V→2. Wrong order silently produces bad outputs.
3. **Wrong `_arch`** — must match `hf_config.architectures[0]` literally (e.g., `'Qwen3VLForConditionalGeneration'`, not `'Qwen3VL'`).
4. **`image_token_id` is None** — causes the engine to silently skip image feature injection. Always verify with `tokenizer.convert_tokens_to_ids(image_token)` returning a real token ID.
5. **Missing `role='preprocess'` append** — `to_pytorch_aux()` searches messages for exactly `role='preprocess'`; if `preprocess()` does not append it, inference will fail with a confusing error.
6. **Config builder `condition()` mismatch** — `model_type` in `condition()` must match the exact string in `config.json`, not a display name or alias.
7. **MoE routing** — MoE models need `num_experts`, `num_experts_per_tok`, and a TopK gating mechanism in the MLP. Reference `qwen3_moe.py` for the pattern.
8. **CUDA graph + dynamic control flow** — models with data-dependent branching (e.g., conditional expert dispatch) may break CUDA graph capture. Use `_no_cudagraph` guards in `CudaGraphMixin` if needed.

______________________________________________________________________

## Verification

**LLM basic test:**

```bash
python -m lmdeploy.pytorch.chat <model_path> --backend pytorch
```

**VLM basic test:**

```python
from lmdeploy import pipeline
pipe = pipeline('<model_path>')
result = pipe(('Describe this image.', 'path/to/image.jpg'))
print(result.text)
```

**Unit tests:**

```bash
pytest tests/test_lmdeploy/test_vl/     # VLM tests
pytest tests/test_lmdeploy/             # all unit tests
```

**Debug weight loading:**

```bash
LMDEPLOY_LOG_LEVEL=DEBUG python -m lmdeploy.pytorch.chat <model_path> --backend pytorch 2>&1 | grep -E "load|weight|miss"
```
