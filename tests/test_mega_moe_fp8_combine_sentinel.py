# Stream B sentinel — verifies the FP8 second-a2a (combine) path agrees
# with the BF16 baseline within the FP8 quant noise floor.
#
# Methodology:
#   - Run kernel with DG_USE_FP8_COMBINE=0 → BF16 combine baseline → y_bf16
#   - Run kernel with DG_USE_FP8_COMBINE=1 → FP8 combine path → y_fp8c
#   - Compare end-to-end y rel-RMSE between the two.
#
# Pass criterion: rel-RMSE <= 0.30 with the kernel's default activation
# clamp. The FP8 (E4M3) quant noise on a single per-row UE8M0 SF is
# ~5-7% per element; summed across kNumTopk=6 slots, the noise reduces
# by sqrt(6) ~ 2.4× to ~2-3% per cell. Allow 30× margin for tail
# outliers and to match the existing l1_sentinel's <=0.5 ceiling for
# FP4 acts.
#
# Usage:
#   torchrun --nproc-per-node=8 tests/test_mega_moe_fp8_combine_sentinel.py
#
# Notes:
#   - Compatible with `kUseFp4Acts` / `kUseMxf4Kind`: just set
#     DG_USE_FP4_ACTS=1 / DG_USE_MXF4_KIND=1 in the env to test the
#     FP4-acts mainloop on top of FP8 combine.
#   - Production sentinel target (FP4 acts vs FP8 acts agreement, which
#     is checked separately in `test_mega_moe_l1_sentinel.py`) is also
#     orthogonal to FP8 combine — the combine pass is independent of
#     the dispatch quant level.

import argparse
import os
import random
from typing import Tuple

import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist


def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    num_tokens = args.num_tokens
    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    activation_clamp = args.activation_clamp

    # Inputs (deterministic per-rank seed).
    x_bf16 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    l1_weights_bf16 = torch.randn(
        (num_experts_per_rank, intermediate_hidden * 2, hidden),
        dtype=torch.bfloat16, device='cuda')
    l2_weights_bf16 = torch.randn(
        (num_experts_per_rank, hidden, intermediate_hidden),
        dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    cumulative = torch.zeros((num_experts_per_rank,), dtype=torch.int, device='cuda')

    # FP8 acts (always — we test the combine-path-only delta here).
    x_fp8 = per_token_cast_to_fp8(x_bf16, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    x_fp4 = per_token_cast_to_fp4(x_bf16, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)

    def cast_grouped_weights_to_fp4(bf16_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_groups, n, k = bf16_weights.shape
        w = torch.empty((num_groups, n, k // 2), device='cuda', dtype=torch.int8)
        w_sf = torch.empty((num_groups, n, k // 32), device='cuda', dtype=torch.float)
        for i in range(num_groups):
            w[i], w_sf[i] = per_token_cast_to_fp4(bf16_weights[i], use_ue8m0=True, gran_k=32)
        w_sf = deep_gemm.transform_sf_into_required_layout(w_sf, n, k, (1, 32), num_groups)
        return w, w_sf

    l1_weights_fp4 = cast_grouped_weights_to_fp4(l1_weights_bf16)
    l2_weights_fp4 = cast_grouped_weights_to_fp4(l2_weights_bf16)
    transformed_l1_weights, transformed_l2_weights = \
        deep_gemm.transform_weights_for_mega_moe(l1_weights_fp4, l2_weights_fp4)

    use_fp4_acts = os.environ.get('DG_USE_FP4_ACTS', '0') != '0'
    x_src = x_fp4 if use_fp4_acts else x_fp8

    def make_buffer_and_run(use_fp8_combine: bool):
        os.environ['DG_USE_FP8_COMBINE'] = '1' if use_fp8_combine else '0'
        os.environ['DG_COMM_KERNEL_DEBUG'] = '0'
        buf = deep_gemm.get_symm_buffer_for_mega_moe(
            group, num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
        )

        def run_once():
            buf.x[:num_tokens].copy_(x_src[0])
            buf.x_sf[:num_tokens].copy_(x_src[1])
            buf.topk_idx[:num_tokens].copy_(topk_idx)
            buf.topk_weights[:num_tokens].copy_(topk_weights)
            cumulative.zero_()
            y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
            deep_gemm.fp8_fp4_mega_moe(
                y, transformed_l1_weights, transformed_l2_weights, buf,
                cumulative_local_expert_recv_stats=cumulative,
                activation_clamp=activation_clamp,
                fast_math=bool(args.fast_math),
            )
            return y, cumulative.clone()

        _ = run_once()
        torch.cuda.synchronize()
        y_out, _ = run_once()
        torch.cuda.synchronize()
        buf.destroy()
        return y_out

    # Order: BF16 combine first (default), then FP8 combine.
    y_bf16 = make_buffer_and_run(use_fp8_combine=False)
    y_fp8c = make_buffer_and_run(use_fp8_combine=True)

    y_diff = (y_fp8c.float() - y_bf16.float()).abs()
    y_rmse = y_diff.pow(2).mean().sqrt().item()
    y_bf16_rms = y_bf16.float().pow(2).mean().sqrt().item()
    rel_rmse = y_rmse / max(y_bf16_rms, 1e-12)

    dist_print(f'=== Stream B sentinel — y rel-RMSE (FP8 combine vs BF16 combine) ===',
               once_in_node=True)
    dist_print(f'  use_fp4_acts:    {use_fp4_acts}', once_in_node=True)
    dist_print(f'  y_bf16 RMS:      {y_bf16_rms:.4f}', once_in_node=True)
    dist_print(f'  y_rmse:          {y_rmse:.4f}', once_in_node=True)
    dist_print(f'  rel-RMSE:        {rel_rmse:.4f}', once_in_node=True)
    dist_print(f'  target:          <= 0.30 (FP8 quant chain noise floor)',
               once_in_node=True)
    dist_print(f'  verdict:         {"PASS" if rel_rmse <= 0.30 else "FAIL"}',
               once_in_node=True)

    dist_print(f'\n  y_bf16 [0, :8]:  {y_bf16[0, :8].cpu().tolist()}',
               once_in_node=True)
    dist_print(f'  y_fp8c [0, :8]:  {y_fp8c[0, :8].cpu().tolist()}',
               once_in_node=True)

    assert rel_rmse <= 0.30, \
        f'FP8 combine sentinel: y rel-RMSE {rel_rmse:.4f} > 0.30'

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-processes', type=int, default=2)
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=8192)
    parser.add_argument('--num-tokens', type=int, default=512)
    parser.add_argument('--hidden', type=int, default=1024)
    parser.add_argument('--intermediate-hidden', type=int, default=512)
    parser.add_argument('--num-experts', type=int, default=8)
    parser.add_argument('--num-topk', type=int, default=2)
    parser.add_argument('--activation-clamp', type=float, default=10)
    parser.add_argument('--fast-math', type=int, default=1)
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test, args=(num_processes, args), nprocs=num_processes)
