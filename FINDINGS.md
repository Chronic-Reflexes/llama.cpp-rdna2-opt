# FINDINGS.md — Live Benchmark Log & Technical Findings

> **Purpose**: Living log of benchmark measurements, verified optimizations, failed experiments, and current GPU bottlenecks for `Qwen3.8-4B-Q8_0.gguf` on AMD Radeon RX 6600 (`AMD_RDNA2`, Windows 11 Vulkan).  
> **Project Goal**: Maximize both **inference performance (tokens/sec)** and **VRAM savings** while maintaining **100% mathematical/output accuracy with zero degradation to model knowledge or skills** (lossless graph, memory-layout, shader, and buffer optimizations only).  
> **Coding rules & workflow**: See [`AGENTS.md`](./AGENTS.md).

---

## 1. Benchmark Progression (`llama-bench` `pp128` / `tg32`)

| # | Configuration / Commit | `pp128` (t/s) | `tg32` (t/s) | Latency (ms/tok) | Speedup vs. Base | Status |
| :- | :--- | :---: | :---: | :---: | :---: | :--- |
| 0 | Upstream `llama.cpp` baseline (`7476e47`) | — | `3.08` | `324.7 ms` | `1.00x` | Baseline |
| 1 | `token_embd.weight` moved to `buft_list_output` (`Vulkan0`, `splits = 1`) | — | `3.08` (`11.42` burst w/ MTP) | `324.7 ms` | Eliminated ~290 ms CPU matmul | Committed (`b4897a1`) |
| 2 | Pure `DEVICE_LOCAL` VRAM (`$env:GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM="1"`) | `181.95 ± 1.40` | `7.90 ± 0.00` | `126.6 ms` | `2.56x` | Active (Runtime Env) |
| 3 | Two-Stage Split-K (`groups_z = 4` + `prealloc_split_k` + `split_k_reduce`) | `165.66 ± 3.11` | `8.66 ± 0.02` | `115.5 ms` | `2.81x` | Reverted (Barrier overhead) |
| 4 | `NUM_ROWS = 1` on `Q8_0` (`DMMV_WG_SIZE_SUBGROUP`, 32 threads/WG) | `165.32 ± 1.68` | `8.68 ± 0.02` | `115.2 ms` | `2.82x` | Superseded by #5 |
| 5 | `DMMV_WG_SIZE_LARGE` (128 threads = 4 wave32s/WG) + `NUM_ROWS = 1` | `170.40 ± 1.85` | `10.26 ± 0.03` | `97.5 ms` | `3.33x` | Committed (`b4897a1`) |
| 6 | Native `MMVQ` (`Q8_0 x Q8_1` `SDOT4`) enabled for `AMD_RDNA2` | **`170.87 ± 0.63`** | **`10.36 ± 0.01`** | **`96.5 ms`** | **`3.36x`** | **Committed (`b4897a1`)** |
| 7 | `MMVQ` + `k_per_iter = 8` for `GGML_TYPE_Q8_0` when `ne00 <= 4096` | `172.00 ± 1.66` | `10.59 ± 0.04` | `94.4 ms` | `3.44x` | Tested locally (uncommitted) |

---

## 2. Verified Lossless Wins (Keep)

1. **Pure `DEVICE_LOCAL` GDDR6 Allocations (`GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1`)**:
   * Bypassed AMD Windows ReBAR `HOST_VISIBLE | DEVICE_LOCAL` read penalty, boosting raw `tg32` from `3.08 t/s` to `7.90 t/s` (+156%) with bit-identical math.
2. **Eliminating CPU Fallback on `token_embd.weight` (`src/llama-model-loader.cpp`)**:
   * `LLM_TENSOR_LAYER_INPUT` (644 MiB `Q8_0`) previously fell back to `CPU_Mapped` because `Vulkan_Host` rejected `Q8_0`. Assigning `buft_list = buft_list_output` put 100% of all 1,905 graph nodes on `Vulkan0` (`splits = 1`).
3. **`DMMV_WG_SIZE_LARGE` (128 threads) + `NUM_ROWS = 1` for `Q8_0` (`ggml-vulkan.cpp`)**:
   * Raised `tg32` from `7.90 t/s` to `10.26 t/s` by launching 4 wave32s per workgroup (coalescing contiguous 64-byte L1 cache lines) and cutting VGPR pressure per wave.
4. **Native `MMVQ` (`Q8_0 x Q8_1` `dot4I8Packed`) on `AMD_RDNA2`**:
   * Upstream `ggml_vk_should_use_mmvq` restricted `GGML_TYPE_Q8_0` to `AMD_GCN`. Enabling `AMD_RDNA2` uses hardware `SDOT4` integer dot products and locks in `10.36 t/s` (`96.5 ms/tok`) without needing `GGML_VK_FORCE_MMVQ="1"`.

---

## 3. Failed Experiments & Dead Ends (Do Not Repeat)

* **`VK_EXT_shader_atomic_float` (`atomicAdd` on `float` buffers)**: Rejected at runtime on Windows (`atomic_float=0` from AMD Adrenalin driver).
* **Two-Stage Split-K (`groups_z = 4` + `split_k_reduce`)**: Only reached `8.66 t/s` because ~320 extra `vkCmdPipelineBarrier` drains + ~160 reduce dispatches per token negated the parallelism gain, and fused `MUL_MAT_ADD` ops (`num_additional_fused_ops > 0`) were skipped.
* **`uint32_t` Buffer Aliasing for `block_q8_0`**: `sizeof(block_q8_0)` is 34 bytes (`2 mod 4`), so even block indices are not 4-byte aligned.
* **Workgroup Sizes > 128 (`wg_size_subgroup > 128`)**: Hardcoded 4-subgroup shared memory (`tmpsh`) in `mul_mat_vec_base.glsl` caps workgroups at 128 threads.

---

## 4. Current Bottleneck Profile (`GGML_VK_PERF_LOGGER=1` at ~94 ms/tok)

* **`MUL_MAT_VEC q8_0 m=9216 n=1 k=2560`** (64 dispatches): **37.5 ms** (`80.6 GFLOPS/s`, ~42 GB/s)
* **`MUL_MAT_ADD MUL_MAT_VEC q8_0 m=2560 n=1 k=9216`** (32 dispatches): **18.0 ms** (`83.8 GFLOPS/s`, ~44 GB/s)
* **`MUL_MAT_VEC q8_0 m=8192 n=1 k=2560`** (32 dispatches): **14.8 ms** (`91.0 GFLOPS/s`, ~48 GB/s)
* **`MUL_MAT_ADD MUL_MAT_VEC q8_0 m=2560 n=1 k=4096`** (31 dispatches): **6.4 ms** (`101.8 GFLOPS/s`)
* **`MUL_MAT_VEC q8_0 m=4096 n=1 k=2560`** (24 dispatches): **6.1 ms** (`82.7 GFLOPS/s`)
* **`MUL_MAT_VEC q8_0 m=248320 n=1 k=2560`** (vocab head, 1 dispatch): **3.2 ms** (`397.1 GFLOPS/s`, ~205 GB/s — full GDDR6 speed)
* **All other ops** (`GATED_DELTA_NET`, `FLASH_ATTN_EXT`, `RMS_NORM`, `GLU`, `CONCAT`, etc.): **~8.2 ms**

---

## 5. Open Next Steps (Lossless Performance & VRAM Optimizations Only)

> **Constraint**: Do **not** use lossy weight re-quantization (e.g., `Q4_K_M`, `IQ4_XS`) or lossy KV compression. Every optimization below preserves 100% mathematical/output fidelity of `Qwen3.8-4B-Q8_0.gguf`.

1. **Benchmark End-to-End Lossless MTP Speculative Decoding (`llama-cli` / `llama-server`)**:
   * Build `llama-cli` and `llama-server` and measure real-world generation speed and VRAM footprint with `--spec-type draft-mtp --spec-draft-n-max 4` on top of the `10.36 t/s` base forward pass (~81.7% acceptance, mean length 4.27 -> expected **~25–35+ t/s** with zero accuracy change because target verification is exact).
2. **Eliminate Shared-Memory Barriers on `m = 2560..9216` (`USE_SUBGROUP_ADD_NO_SHMEM`) & Operator Fusion**:
   * Investigate running 1 wave32 (`32` threads) per row with `USE_SUBGROUP_ADD_NO_SHMEM` (`subgroupAdd` only, zero workgroup `barrier()`) or fusing `Q/K/V` and `Gate+Up` projections into single dispatches to close the ~45 GB/s vs. ~205 GB/s bandwidth gap while also eliminating intermediate activation buffers.
3. **Lossless VRAM Footprint Reductions (Fit Larger Contexts Inside 8 GB GDDR6)**:
   * **Audit Vulkan Compute & Scratch Buffer Allocations (`ggml-vulkan.cpp`)**: Inspect `prealloc_size_x`, `prealloc_size_y`, `prealloc_size_split_k`, and `vk_pool` staging allocations to trim unused or oversized persistent scratch buffers on `n = 1` / small-batch inference.
   * **Hybrid Architecture KV Cache Sizing**: Because `Qwen3.8-4B` is a hybrid Gated DeltaNet + Attention architecture (only 8 of 32 layers use full KV attention while 24 layers use fixed-size recurrent state), audit KV cache allocation (`-c` sizing and recurrent cell reservation) to reclaim unused VRAM without altering precision (`F16` KV).
   * **Sub-Allocation / Buffer Aliasing for `token_embd.weight` vs. Compute Graph**: Check whether `ggml-backend` graph allocator over-reserves `Vulkan0` compute buffers alongside the 4.28 GiB model weights on 8 GB VRAM.
