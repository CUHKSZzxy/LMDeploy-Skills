# LLM Code Skeletons

## Required Imports

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

______________________________________________________________________

## Attention Skeleton

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

______________________________________________________________________

## MLP Skeleton

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

______________________________________________________________________

## ForCausalLM Skeleton

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

## Config Builder Skeleton

Only needed for non-standard HF configs (nested config, unusual `model_type`).

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

Auto-discovery: subclasses of `AutoModelConfigBuilder` register via `__init_subclass__()` — no import needed elsewhere.
