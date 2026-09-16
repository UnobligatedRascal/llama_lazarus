# llama_lazarus — Priority Order for Maximum Inference Speed on K80

UnobligatedRascal | Target: 8x Tesla K80 sm_37 on NOUGHT | Updated: 2026-09-17

**Objective:** Maximum single-user inference speed on 8x Tesla K80.
**Constraint:** Memory-bound bottleneck (~13% of theoretical max). Arithmetic optimization is waste; focus on bandwidth reduction.

---

## Priority Order — Maximum Inference Speed

### P0 — BLOCKING
1. **TurboQuant/meta backend crash** — turbo2_0/turbo3_0/turbo4_0 crashes with tensor-split. Blocks all turbo KV cache usage. **Must fix: turbo = the bandwidth reduction we need.**

### P1 — HIGH
2. **cuBLAS vs MMQ benchmark** — cuBLAS F32 may beat integer MMQ on Kepler. Measure: `-DGGML_CUDA_FORCE_CUBLAS=ON` vs `-DGGML_CUDA_FORCE_MMQ=ON`.
3. **TurboQuant with tensor-split** — After crash fix, use turbo4_0 KV cache. Less data off chip = more tokens/sec.

### P2 — MEDIUM
4. **NUMA replication validation** — Was 2.3x slower when broken; verify current state actually helps.
5. **KV cache persistence to disk** — Skip for single-user unless reloading same context repeatedly.

### P3 — LOW / DEFERRED
6. IMAD micro-optimization — Research says <2% gain on memory-bound hardware. Skip.
7. LUT-GEMM — Wrong tool for Q4_K_M uniform quant. Skip.
8. Speculative decoding — Tensor-split incompatibility. Abandoned.

---

## Research Notes

- K80 is memory-bound (~13% efficiency of theoretical max), not compute-bound
- IMAD micro-optimizations give <2% gain — skip
- LUT-GEMM is wrong tool for Q4_K_M uniform quant — skip
- cuBLAS F32 may beat integer MMQ on Kepler — benchmark needed
- TurboQuant: 3.8x-6.4x compression vs FP16 = directly more bandwidth = faster tokens
- Focus on bandwidth reduction, not arithmetic optimization

---

## Fixes Applied (Verified, Tested)

### Issue 1: F16→FP32 pointer mismatch in Kepler batched path
- **Error:** `CUBLAS_STATUS_INVALID_VALUE` on `cublasSgemmBatched`
- **Cause:** Called `cublasSgemmBatched` (expects FP32) with FP16 pointers
- **Fix:** Convert FP16/BF16→FP32 before batched call (Kepler cc<500 path)

### Issue 2: nullptr assertion on F32 compute path
- **Error:** `GGML_ASSERT(to_fp32_src0 != nullptr)`
- **Cause:** Called `ggml_get_to_fp32_cuda(GGML_TYPE_F32)` → returns nullptr
- **Fix:** Split Kepler path: `if (cc<500 && compute_type!=F32)` converts; `else if (cc<500)` uses existing FP32 pointers directly

### Issue 3: Wrong strides for contiguous quantized→compute_type conversion
- **Error:** `CUBLAS_STATUS_INVALID_VALUE` parameter 13 on quantized model inference
- **Cause:** Strides multiplied by `block_size` after conversion, but converted data is contiguous in compute_type elements
- **Fix:** Replace `s01*=bs; s02*=bs; s03*=bs` with `s01=ne00; s02=ne01*s01; s03=ne02*s02`
- **Commit:** `992315e0e`

---

## Build

```bash
cd /home/whistler/llama_lazarus && rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_F16=ON \
  -DCMAKE_CUDA_HOST_COMPILER=g++-11 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc \
  -DGGML_CUDA_NCCL=ON -DCMAKE_CUDA_ARCHITECTURES="37" -DLLAMA_CURL=OFF \
  -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_CUDA_FORCE_MMQ=ON -DGGML_CUDA_GRAPHS=OFF \
  -DCMAKE_C_COMPILER=gcc-11 -DCMAKE_CXX_COMPILER=g++-11 -DGGML_CUDA_CUBLAS=ON \
  -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,/usr/local/cuda-11.8/targets/x86_64-linux/lib"
make -j$(nproc) llama-server
```

## System

- **NOUGHT:** 192.168.137.29, Debian/Q4OS, user whistler
- **GPU:** 8× Tesla K80 (sm_37, 11GB each)
- **CUDA:** Toolkit 11.8, Driver 470.256.02, Runtime 11.4

---

## IMPORTANT

- Running llama-server on NOUGHT (`/home/whistler/llama.cpp/build/bin/llama-server`, PI's process) is **production** — DO NOT KILL without explicit instruction.
- NEVER kill llama-server without confirmation.

---
END TODO.md — UnobligatedRascal
