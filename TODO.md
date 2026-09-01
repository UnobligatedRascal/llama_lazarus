# Project Llama Lazarus - Kepler sm_37 cuBLAS BF16 Fix

## Status
✅ PATCHED & BUILT  
⏳ TEST PENDING (old llama-server still running on all 8 GPUs, need root to kill PID 72093)

## Target System: NOUGHT Server
- **GPU**: 8× Tesla K80 (Kepler sm_37, 11GB each)
- **Driver**: 470.256.02, CUDA Version: 11.4
- **CUDA Toolkit**: 11.8 (at /usr/local/cuda-11.8)
- **OS**: Debian Q4OS (x86_64)
- **Build Path**: `/home/whistler/llama.cpp/build`

## Completed Steps

### 1. Source Patching (ggml-cuda.cu)
File: `/home/whistler/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`

**Patch 1**: Added hardware capability check for cuBLAS GEMM algorithm selection
- Location: `ggml_cuda_mul_mat_cublas_impl()` after `cc` variable declaration (~line 1515)
- Code:
  ```cpp
  const cublasGemmAlgo_t gemm_algo = (cc >= GGML_CUDA_CC_VOLTA) ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
  ```

**Patch 2**: Added BF16 hardware capability check with F32 fallback
- Location: `ggml_cuda_mul_mat_cublas()` case GGML_TYPE_BF16: block (~line 1659)
- Code:
  ```cpp
  if (!bf16_mma_hardware_available(ggml_cuda_info().devices[ctx.device].cc)) {
      GGML_LOG_WARN("BF16 not available, falling back to F32\n");
      ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(ctx, src0, src1, dst);
      break;
  }
  ```

**Patch 3**: Replaced 3× `CUBLAS_GEMM_DEFAULT_TENSOR_OP` with `gemm_algo` variable
- In `ggml_cuda_mul_mat_cublas_impl()` GEMM calls

### 2. Build Configuration
```bash
cd /home/whistler/llama.cpp && rm -rf build && mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_SCHED_MAX_COPIES=4 \
  -DGGML_CUDA_F16=ON \
  -DGGML_CUDA_PEER_MAX_BATCH_SIZE=64 \
  -DCMAKE_CUDA_HOST_COMPILER=g++-11 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc \
  -DGGML_CUDA_NCCL=ON \
  -DCMAKE_CUDA_ARCHITECTURES=37 \
  -DLLAMA_CURL=OFF \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_CUDA_FORCE_MMQ=ON \
  -DGGML_CUDA_ARCHITECTURES="37" \
  -DGGML_CUDA_GRAPHS=OFF \
  -DCMAKE_C_COMPILER=gcc-11 \
  -DCMAKE_CXX_COMPILER=g++-11 \
  -DGGML_CUDA_CUBLAS=ON \
  -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,/usr/local/cuda-11.8/targets/x86_64-linux/lib"

make -j$(nproc) llama-server
```

### 3. Key Issues Resolved
- **Linker Error**: `cudaLaunchKernelExC@libcudart.so.11.0` undefined
  - Root: System ld.so.conf had CUDA 11.4 lib path before 11.8
  - Fix: Added rpath to 11.8 lib path via CMAKE_SHARED_LINKER_FLAGS
- **cuBLAS BF16**: Kepler sm_37 lacks BF16 hardware support
  - Fix: Added runtime hardware check with F32 fallback
- **Tensor Core Ops**: sm_37 doesn't support tensor ops
  - Fix: Compute gemm_algo based on compute capability

## Verification
- ✅ Binary built: `/home/whistler/llama.cpp/build/bin/llama-server`
- ✅ Version: `0.3.0-dev (build 10729, commit 458681e1d)`
- ✅ Links libcudart.so.11.0 from CUDA 11.8 (via RUNPATH)
- ✅ No missing dependencies

## Testing Plan

### Step 1: Kill Old Server
```bash
sudo kill -9 72093
```

### Step 2: Single-GPU Test (-np 1)
```bash
cd /home/whistler/llama.cpp/build
nvidia-smi && ./bin/llama-server \
  --model /path/to/model.gguf \
  --n-gpu-layers 99 \
  --n-parallel 4 \
  --ctx-size 4096 &
# Wait for server to start
nvidia-smi  # Verify single GPU utilization
kill $!
```

### Step 3: Multi-GPU Test (-np 4)
```bash
cd /home/whistler/llama.cpp/build
nvidia-smi && ./bin/llama-server \
  --model /path/to/model.gguf \
  --n-gpu-layers 99 \
  --n-parallel 4 \
  --ctx-size 4096 \
  -np 4 &
# Wait for server to start
nvidia-smi  # Verify 4 GPU utilization
kill $!
```

### Step 4: Multi-User Load Test
```bash
# Start server with -np 4
# Run concurrent requests
for i in 1 2 3 4; do
  curl -s http://localhost:8080/completion \
    -d '{"prompt": "Hello world", "n_predict": 50}' \
    > /tmp/test_$i.txt &
done
wait
nvidia-smi  # Check memory and utilization across GPUs
```

## Rollback Options (if issues)
1. Disable NCCL: Remove `-DGGML_CUDA_NCCL=ON`
2. Disable FORCE_MMQ: Remove `-DGGML_CUDA_FORCE_MMQ=ON`
3. Disable FA_ALL_QUANTS: Remove `-DGGML_CUDA_FA_ALL_QUANTS=ON`
4. Restore from backup: `/home/whistler/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu.bak_*`

## References
- Patch script: `/tmp/llama_cpp_patch.py`
- Original TODO: `/home/whistler/llama.cpp/build/TODO.md`
- Build log: `/home/whistler/llama.cpp/build/build.log` (if created)

---
Last updated: 2026-08-31 23:40 UTC  
Next: Kill old server process, run single-GPU test, multi-GPU test, load test
