# Common Pitfalls Catalog

Match your symptom against this catalog before debugging further.

| #   | Symptom                                                  | Root cause                                                       | Fix                                                                                                           |
| --- | -------------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| 1   | Missing/unexpected key errors on weight load             | `packed_modules_mapping` keys don't match HF param name suffixes | Check actual HF weight names: `list(model.state_dict().keys())[:20]` before coding                            |
| 2   | Wrong outputs, no error                                  | `stacked_params_mapping` for QKV in wrong shard order            | Must follow Q→0, K→1, V→2 — wrong order silently produces bad outputs                                         |
| 3   | Model not found / not routed to LMDeploy class           | Wrong `_arch` value                                              | Must match `hf_config.architectures[0]` literally (e.g. `'Qwen3VLForConditionalGeneration'`, not `'Qwen3VL'`) |
| 4   | Image features silently not injected                     | `image_token_id` is `None`                                       | Always verify: `tokenizer.convert_tokens_to_ids(image_token)` must return a real integer ID                   |
| 5   | Confusing inference error mentioning `role='preprocess'` | `preprocess()` does not append `role='preprocess'` to messages   | `to_pytorch_aux()` searches for exactly `role='preprocess'`; old-style `preprocess()` must append it          |
| 6   | Config builder never triggered                           | `condition()` model_type mismatch                                | Must match the exact string in `config.json`, not a display name or alias                                     |
| 7   | MoE model produces wrong outputs or crashes              | Missing MoE routing logic                                        | MoE models need `num_experts`, `num_experts_per_tok`, and TopK gating in MLP; reference `qwen3_moe.py`        |
| 8   | CUDA graph capture fails                                 | Data-dependent branching in model forward                        | Use `_no_cudagraph` guards in `CudaGraphMixin` for conditional expert dispatch or dynamic control flow        |
