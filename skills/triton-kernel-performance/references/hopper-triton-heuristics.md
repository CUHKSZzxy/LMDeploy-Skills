# Hopper / H100 Triton Heuristics

Load this reference when tuning LMDeploy CUDA/Triton kernels on Hopper-class GPUs
such as H100 or H800.

## Preflight

- Confirm the GPU is actually Hopper: `torch.cuda.get_device_capability()` should
  be `9.0`.
- Record GPU clock/power state when comparing small deltas. Avoid mixing runs
  from busy and idle GPUs.
- Warm up until Triton compilation and CUDA graph capture effects are gone.
- Run decode-shaped and prefill-shaped benchmarks separately; Hopper can make
  a prefill win look huge while decode is unchanged.

## Metrics To Capture

Use CUDA-event microbenchmarks first, then profiler evidence for serious claims.

Useful Nsight Compute signals:

- `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- `sm__warps_active.avg.pct_of_peak_sustained_active`
- `smsp__warps_eligible.avg.per_cycle_active`
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `lts__throughput.avg.pct_of_peak_sustained_elapsed`
- `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct`
- local-memory or spill counters when register pressure is suspected

Starter command:

```bash
ncu --set full --target-processes all --kernel-name regex:<kernel_regex> \
  --launch-skip 10 --launch-count 20 python path/to/bench.py
```

Use `nsys profile` when launch order, CPU gaps, stream overlap, or tiny-kernel
clusters matter more than one kernel's internal counters.

## Hopper-Specific Tuning Notes

- Hopper has strong Tensor Core/FP8 paths, but non-matmul KV-cache kernels are
  often memory-bound. Do not chase WGMMA ideas for scalar cache fill/readback
  kernels unless profiler evidence shows matmul-like work.
- TMA is valuable for large tiled copies into shared memory. Most LMDeploy
  KV-cache fill, flatten, and paged decode helper kernels use small strided
  gathers/scatters, so TMA is usually not the first lever.
- Prefer coalesced vectorized lanes over clever indexing. On H100, bad sector
  utilization can erase gains from fewer instructions.
- Watch register pressure when adding fused quant/dequant math. A fused kernel
  that halves launches but spills may lose on decode.
- Keep shape specialization explicit for head dims such as `64`, `128`, and
  `256`. A broad dynamic kernel can lose more from branches and masks than it
  saves in code reuse.
- For FP8 KV cache, scale loads are part of the bandwidth budget. Load each
  scale once per logical token/head group where possible and avoid repeating
  scale reads across element lanes.
- On small-batch decode, launch count and exposed latency can dominate. Fusion
  or CUDA graph compatibility may matter more than a 5% faster individual
  Triton kernel.
- On long-prefill or bulk cache conversion, memory bandwidth and vectorization
  are usually the first bottlenecks.

## Heuristic Search Order

1. Verify correctness and supported fast paths.
2. Identify whether the kernel is memory-bound, compute-bound, launch-bound, or
   hidden behind another stream.
3. If memory-bound: fix layout, coalescing, vector width, redundant metadata
   loads, and masks.
4. If compute-bound: inspect tile size, `num_warps`, `num_stages`, accumulator
   dtype, and Tensor Core eligibility.
5. If launch-bound: consider fusion, graph capture compatibility, or dispatch
   consolidation.
6. If hidden by overlap: optimize only if it changes the exposed critical path,
   launch count, or downstream scheduling.

## Qwen3 / Qwen3.5 Reminders

- Check Q/K norm, RoPE, cache write, and FP8 scale handling as one pipeline.
- Treat speculative/MTP decode as a separate path; an optimization in normal
  decode is not automatically correct there.
- For FP8 or NVFP4 checkpoints, weight-scale naming and KV-scale remapping can
  be correctness bugs, not only loading annoyances.
