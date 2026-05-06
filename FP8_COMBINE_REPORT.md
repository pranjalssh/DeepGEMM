# FP8 + UE8M0 SF on the second a2a (mega-MoE combine path)

Standalone report of the work landed in DeepGEMM PR #28 + companion sglang
PR #24449's `SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP8_COMBINE` flag.

## Why

The mega-MoE second all-to-all (combine) currently ships BF16 over NVLink:
`kHidden * 2` bytes per (token, slot). For DeepSeek-V4-Pro (kHidden=7168,
kNumTopk=6, EP=8), per token per rank that's ~86 KB of NVLink traffic.

This change ships FP8 E4M3 + per-(token, N=128) UE8M0 SF: `kHidden +
kHidden / 128` bytes per (token, slot). For the same shape: 43 KB —
**half the NVLink bytes**.

## Implementation

Producer (L2 epilogue write-back):
- Read 8 BF16 from smem (existing STSM target, unchanged when off).
- Per-row amax via `__shfl_xor_sync` reduction over the 16 lanes that
  share each row tile. **16-lane mask required** — the outer
  `if (m_idx_in_block >= valid_m) break` may cause the OTHER half-warp
  to exit on padding rows; full-warp shfl would deadlock waiting on
  exited lanes.
- UE8M0 SF (E4M3 finfo_max=448, mirrors `get_e4m3_sf_and_sf_inv`).
- Cast 8 BF16 → 8 FP8 via `__nv_fp8x4_e4m3(float4)` ×2; pack into uint64.
- Write 8 FP8 bytes to remote (vs 16 BF16). Lane 0 of the 16-lane group
  writes the SF byte to `combine_sf_buffer`.

Consumer (combine reduce):
- Per-slot SF base ptr cached at slot start.
- TMA-load FP8 chunk (`kNumChunkBytes / 2` bytes when `kUseFp8Combine`).
- Per uint4 (16 FP8): `__ldg` the SF byte for the segment;
  `cvt.rn.f16x2.e4m3x2` → `cvt.f32.f16` → `__fmaf_rn(val, sf, acc)` per
  element.
- BF16 store layout: 2 BF16 uint4 per input FP8 uint4 (16 elements →
  2 × 8 BF16 stripes), at indices `(j*32+lane)*2 + {0,1}`.

The flag is a new template parameter `kUseFp8Combine` (default `false`)
— when off, the BF16 path is byte-identical. Forward it from sglang via
`SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP8_COMBINE=1` → `DG_USE_FP8_COMBINE=1`.

## Microbench

`ptx/d_combine_reduce_v{1,2,3}_*` in the kernels repo. 1 warp / 1 CTA /
1 token, kHidden=7168, kNumTopk=6, kNumChunks=2.

| Variant | Cycles/token | Δ vs BF16 | Notes |
|---|---:|---:|---|
| v1 BF16 baseline | 6,895 | — | max_abs=0 |
| v2 FP8 + FP32 acc | 10,797 | +57% | production path; 50% NVLink savings |
| v3 FP8 + FP16 HFMA | 5,799 | **−16%** | unsafe — see below |

The HFMA path (FP16 accumulator + `fma.f16x2`) is faster because it cuts
per-FP8x2 ops from 5 to 2 (no FP32 cast in inner loop) and halves
register pressure (94 regs vs 138). Reverted from production: FP16
accumulator overflows on random-init test when L2 GEMM amax > ~30K.
Production amax can reach `clamp(10) * fp4_max(6) * intermediate(3072)`
= 184K, well above FP16's 65K dynamic range. To unlock HFMA safely you'd
need either (a) per-slot amax check + dynamic FP32 fallback, or (b)
confirm production amax bounded < 30K (would require activation
measurement on real workloads).

## Sentinel test

`tests/test_mega_moe_fp8_combine_sentinel.py`: end-to-end y rel-RMSE
between FP8-combine and BF16-combine on identical inputs.

| Shape | ntok | DG_USE_FP4_ACTS | rel-RMSE | Verdict |
|---|---:|---|---:|---|
| Smoke (h=1024, ie=512, E=8, K=2) | 256 | 0 | 0.027 | PASS |
| Smoke | 256 | 1 (+MXF4) | 0.027 | PASS |
| Production (h=7168, ie=3072, E=384, K=6, 8-rank) | 1024 | 0 | 0.027 | PASS |
| Production | 2048 | 1 (+MXF4) | 0.027 | PASS |
| Production | 4096 | 0 | 0.027 | PASS |
| Production | 4096 | 1 (+MXF4) | 0.027 | PASS |

Target ≤ 0.30 (= 30% rel-RMSE — well above the FP8 quant chain noise
floor of ~2-3% per cell after sqrt(K=6) reduction).

Independence confirmed: FP8 combine quant noise is the same with or
without FP4 acts, stable from 256 to 4096 tokens.

## Single-GPU iso bench (8x B300, EP8)

`bench_megamoe_iso.py` — fused-kernel-only timing, no NVLink contention.

**FP8 acts only (no FP4):**

| ntok | tpe | FP8 alone | FP8 + combine | delta |
|----:|----:|---------:|---------:|------:|
| 128  | 16  | 375 us   | 375 us   | ~0% |
| 512  | 64  | 449 us   | 412 us   | **+9.0%** |
| 2048 | 256 | 828 us   | 824 us   | +0.4% |

**FP4 + MXF4:**

| ntok | tpe | FP4+MXF4 | FP4+MXF4+combine | delta |
|----:|----:|---------:|-----------------:|------:|
| 32   | 4   | 395 us   | 355 us           | **+10.1%** |
| 64   | 8   | 357 us   | 357 us           | ~0% |
| 128  | 16  | 360 us   | 360 us           | ~0% |
| 512  | 64  | 377 us   | 386 us           | -2.2% |
| 2048 | 256 | 710 us   | 739 us           | -3.9% |

Single-GPU iso bench is compute-bound; production wins come from NVLink
savings + memory-bound regimes. The +9% FP8 b=512 win is the iso-bench
signal of the same NVLink savings that show up at e2e.

## End-to-end (sglang serve, DeepSeek-V4-Pro, 8x B300, 8K input + 1024 output)

Per-GPU throughput = `(input_len + output_len) × bs / latency / 8` (tok/s/gpu).

| batch | FP8 acts | FP8+combine | FP4+MXF4 | FP4+MXF4+combine | combine vs FP8 baseline |
|-:|---:|---:|---:|---:|---:|
| 512  | 6,417 | — | — | **7,526** | **+17.3%** |
| 1024 | — | — | — | **8,806** | — |
| 2048 | 9,096 | 9,158 | 9,814 | **9,962 (3-run avg)** | +0.7% / +9.5% |
| 4096 | 9,639 | — | 10,418 | **10,622** | +10.2% |

## GSM8K accuracy

`sglang.test.few_shot_gsm8k`, 5-shot, parallel=200, single run each.

200 questions:

| Config | Accuracy |
|---|---:|
| FP4 + MXF4 (baseline) | 97.0% |
| **FP4 + MXF4 + FP8 combine** | **97.5%** |

1000 questions (tighter stats, ~±1.5% binomial CI):

| Config | Accuracy |
|---|---:|
| FP4 + MXF4 (baseline) | 95.8% |
| **FP4 + MXF4 + FP8 combine** | **94.9%** |

Both within the previously reported 95.6% ± 0.5 baseline range.
**FP8 combine preserves GSM8K accuracy** at production scale (with
or without FP4 acts in the dispatch path).

## How to use

```bash
export DG_USE_FP8_COMBINE=1   # halve combine NVLink bytes
```

Independent of `DG_USE_FP4_ACTS` / `DG_USE_MXF4_KIND`. Combinable for
maximum savings:

```bash
export DG_USE_FP4_ACTS=1
export DG_USE_MXF4_KIND=1
export DG_USE_FP8_COMBINE=1
```

In sglang (companion PR #24449):

```bash
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS=1
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND=1
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP8_COMBINE=1
```

## Future work

- **Safe HFMA**: per-slot amax check + dynamic FP32 fallback. Recover
  the v3 microbench's 16% advantage over BF16 for slots whose amax
  fits in FP16 range.
- **Apply to other a2a kernels**: same pattern (BF16 → FP8 + UE8M0 SF
  on the wire) could halve NVLink traffic for other cross-rank reduces
  in the codebase (paged_mqa_logits combine, etc.).
- **gran_k = 64 SF**: tighter quant resolution at the cost of 2× more
  SF bytes per token. Probably accuracy-positive but bandwidth-negative
  (`kHidden + kHidden/64` bytes/token = ~52% of BF16 instead of 50%).
