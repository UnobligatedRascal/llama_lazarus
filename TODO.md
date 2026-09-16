# llama_lazarus — Priority Order for Maximum Inference Speed

UnobligatedRascal | Target: 8x Tesla K80 sm_37 on NOUGHT

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
END TODO.md — UnobligatedRascal
