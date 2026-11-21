# Instructions for llama.cpp (Chronic-Reflexes/llama.cpp-rdna2-opt)

## Important

This repository (`Chronic-Reflexes/llama.cpp-rdna2-opt`) is a **standalone performance fork** targeting **AMD RDNA 2 (`AMD_RDNA2`) Vulkan inference on Windows** (`Qwen3.8-4B-Q8_0` and similar models). We do **not** submit PRs to upstream `ggml-org/llama.cpp`.

* **Living Benchmark & Findings Log**: Always read [`FINDINGS.md`](./FINDINGS.md) before making any backend or shader changes, and update `FINDINGS.md` after every benchmarked experiment.
* **AI Assistance**: AI agents are explicitly authorized to write code, generate commit messages, run local `git commit` / `git push` to `llama-ops` (`Chronic-Reflexes/llama.cpp-rdna2-opt`), and drive optimization iterations directly.

---

## Hardware, Build & Runtime Rules (Mandatory)

1. **Target Hardware & Driver**:
   * **GPU**: AMD Radeon RX 6600 (8 GB GDDR6, RDNA 2 / `AMD_RDNA2`, 28 CUs, 32 MB Infinity Cache, warp size `32`, `int dot: 1`, `rebar: 1`) on Windows 11 (AMD proprietary Adrenalin Vulkan driver).
   * **Runtime Environment Variable**: Always set `$env:GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM="1"` when running `llama-bench`, `llama-cli`, or `llama-server` so weights allocate in pure `DEVICE_LOCAL` GDDR6 VRAM instead of slow ReBAR `HOST_VISIBLE | DEVICE_LOCAL` heaps.
2. **Driver & Shader Guardrails**:
   * **No `VK_EXT_shader_atomic_float`**: AMD's Windows Adrenalin Vulkan driver reports `shaderBufferFloat32AtomicAdd = 0` on RX 6600. Do not use `atomicAdd` on `float` storage buffers.
   * **Shared Memory Cap (`128` threads)**: `mul_mat_vec_base.glsl` sizes `tmpsh` for at most 4 subgroups (`4 * 32 = 128` threads). Never exceed `wg_size_subgroup = 128` without updating `tmpsh` across all variants.
   * **34-Byte `block_q8_0` Alignment**: `sizeof(block_q8_0)` is 34 bytes (`2 mod 4`). Never alias `block_q8_0` buffers as `uint32_t[]` in GLSL; always use `data_a_packed16` (`i16vec2`).
   * **No Extra Barriers in Hot Loops**: Avoid inserting `ggml_vk_sync_buffers` (`vkCmdPipelineBarrier`) or multi-pass reduction dispatches inside per-layer token generation loops.
3. **Build & Benchmark Workflow**:
   * **Generator**: Always use **Ninja + `ccache`** (`-G Ninja -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache`). Never use `ccache` with Visual Studio MSBuild (MSBuild deletes `.rsp` response files before `ccache` reads them).
   * **Incremental Builds**: Build only the target being tested (`cmake --build build --target llama-bench -j`).
   * **Benchmark Baseline Command**:
     `llama-bench.exe -m "C:\Users\Partr\.ollama\models\Qwen3.8-4B-Q8_0.gguf" -ngl 99 -p 128 -n 32`
   * **Per-Op Profiling**: Use `$env:GGML_VK_PERF_LOGGER="1"` with `-p 0 -n 2` to inspect per-dispatch GFLOPS/s and microsecond timings.
   * **PowerShell Delivery**: Keep pasted PowerShell commands concise (under ~10 lines) or write a `.ps1` script so terminal pastes never truncate.

---

## Code and Commit Standards (Adapted from Upstream `llama.cpp`)

Please follow these style conventions so our diff against upstream `llama.cpp` stays clean, readable, and easy to rebase:

* **ASCII Only**: Avoid emdash (`—`), unicode arrows (`→`), or unicode math symbols (`×`, `…`); use ASCII equivalents (`-`, `->`, `x`, `...`).
* **Line Endings**: Preserve LF (`\n`) line endings when editing GLSL shaders (`*.comp`, `*.glsl`) in `ggml/src/ggml-vulkan/vulkan-shaders/`.
* **Code Comments**:
  * Keep code comments concise (1-2 lines) using simple, direct wording.
  * Explain non-obvious hardware invariants (e.g., RDNA 2 wave occupancy or driver quirks), not what the code already says.
  * Avoid redundant inline commentary or hard-wrapping comments to a fixed column width.
* **Infrastructure & Scope**:
  * Prefer reusing existing `ggml-vulkan` infrastructure over introducing whole new subsystems.
  * Before writing code, read the relevant files (`ggml/src/ggml-vulkan/ggml-vulkan.cpp`, `vulkan-shaders/*.glsl`) so changes blend naturally with surrounding patterns.
  * Llama.cpp does **not** use Minja; its Jinja engine lives in `common/jinja`.
* **Commits & Git**:
  * Keep commits focused on single, benchmark-verified changes on branch `rdna2-vulkan-q8-opt` (remote `llama-ops`).
  * Write concise commit messages with `Assisted-by: <assistant name>` when AI-assisted.

### Code Comment Examples

```cpp
// GOOD (code is self-explanatory, no comment needed)
n_ctx = read_metadata("context_length", 1024);

// BAD (too verbose, restates what the code already says)
// Populate the n_ctx from metadata key name "context_length", default to 1024 if the key doesn't exist
n_ctx = read_metadata("context_length", 1024);

// GOOD (explains a non-obvious hardware invariant)
// RDNA 2 needs 128-thread workgroups (4 wave32s) to coalesce Q8_0 L1 cache-line reads
wg_size_idx = DMMV_WG_SIZE_LARGE;

// BAD (addresses a single chat session out of context)
// Changed this from DMMV_WG_SIZE_SUBGROUP because our benchmark was 7.90 t/s earlier
wg_size_idx = DMMV_WG_SIZE_LARGE;
```

### Commit Message Examples

```text
// GOOD
ggml-vulkan : enable native MMVQ Q8_0 on AMD RDNA 2
Assisted-by: Gemini

// BAD
This commit introduces a comprehensive fix for RDNA 2 matrix-vector multiplication by changing several flags and testing different workgroup sizes.
```

---

## Useful Resources

* **Live Benchmarks & Experiment History**: [`FINDINGS.md`](./FINDINGS.md)
* **Skills**: Reusable task workflows live in the `skills/` directory.
* **Documentation**:
  * [Build documentation](docs/build.md)
  * [Server usage documentation](tools/server/README.md)
  * [Jinja engine](common/jinja)
