# Project Llama Lazarus - Kepler sm_37 cuBLAS Fixes

## Status: REBUILD FROM MAIN COMPLETE (2026-09-17) — PATCHES COMMITTED

**IMPORTANT**: If running llama-server as production, DO NOT KILL the existing process without validation.
**NEW BINARY** (untested, awaiting validation): `llama_lazarus/build/bin/llama-server`

---

## Fixes Applied (All Committed, Tested)

### Issue 1: F16→FP32 pointer mismatch in Kepler batched path
- **Error**: `CUBLAS_STATUS_INVALID_VALUE` on `cublasSgemmBatched`
- **Cause**: Called `cublasSgemmBatched` (expects FP32) with FP16 pointers
- **Fix**: Convert FP16/BF16→FP32 before batched call (Kepler cc<500 path)

### Issue 2: nullptr assertion on F32 compute path
- **Error**: `GGML_ASSERT(to_fp32_src0 != nullptr)`
- **Cause**: Called `ggml_get_to_fp32_cuda(GGML_TYPE_F32)` → returns nullptr
- **Fix**: Split Kepler path: `if (cc<500 && compute_type!=F32)` converts; `else if (cc<500)` uses existing FP32 pointers directly

### Issue 3: Wrong strides for contiguous quantized→compute_type conversion [LATEST]
- **Error**: `CUBLAS_STATUS_INVALID_VALUE` parameter 13 on quantized model inference
- **Cause**: Strides multiplied by `block_size` after conversion, but converted data is contiguous in compute_type elements — strides should be dimension-based
- **Fix** (lines ~1458-1464, ~1483-1489): Replace `s01*=bs; s02*=bs; s03*=bs` with `s01=ne00; s02=ne01*s01; s03=ne02*s02`
- **Commit**: `992315e0e`
- **Test**: Q4_K_M 27B, -np 1 — loads, 40 tokens generated, no CUDA errors

---

## Build

```bash
cd llama_lazarus && rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_F16=ON \
  -DCMAKE_CUDA_HOST_COMPILER=g++-11 -DCMAKE_CUDA_COMPILER=<CUDA_PATH>/bin/nvcc \
  -DGGML_CUDA_NCCL=ON -DCMAKE_CUDA_ARCHITECTURES="37" -DLLAMA_CURL=OFF \
  -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_CUDA_FORCE_MMQ=ON -DGGML_CUDA_GRAPHS=OFF \
  -DCMAKE_C_COMPILER=gcc-11 -DCMAKE_CXX_COMPILER=g++-11 -DGGML_CUDA_CUBLAS=ON \
  -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,<CUDA_LIB_PATH>"
make -j$(nproc) llama-server
```

## System

- **Target Hardware**: 8× Tesla K80 (sm_37, 11GB each)
- **CUDA**: Toolkit 11.8, Driver 470.256.02, Runtime 11.4
- **Other patches**: gemm_algo selection, CUBLAS_DEFAULT_MATH, solve_tri math mode

## Key Files

- `ggml/src/ggml-cuda/ggml-cuda.cu`: mul_mat_cublas_impl (~1395), Kepler path (~1610), strides fix (~1458, ~1483)
- `ggml/src/ggml-cuda/common.cuh`: math mode (~1506)
- `ggml/src/ggml-cuda/solve_tri.cu`: math mode (~75)

---

## Recent Activity (2026-09-17)

- Rebased patches onto latest llama.cpp main (`930e2fa59`, Sep 17 2026)
- Fixed `.git/objects` permissions (some dirs owned by root)
- Committed patches: `8a0943c21` on branch `sync-upstream-2026-09-17`
- Rebuilt `llama-server` successfully
- Build config: cc=37, cuBLAS F16, MMQ forced, FA all quants, NCCL on

---
Last updated: 2026-09-17
