#!/usr/bin/env python3
import argparse
import time

import torch
from flash_attn.flash_attn_interface import flash_attn_func


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke benchmark for FlashAttention Proteus JIT forward path."
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=[64, 96, 128, 192],
        help="Head dimensions to test.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_grad_enabled(False)

    for d in args.dims:
        q = torch.randn(
            args.batch, args.seqlen, args.heads, d, device="cuda", dtype=torch.float16
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = flash_attn_func(
            q, k, v, dropout_p=0.0, softmax_scale=None, causal=False
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        out = flash_attn_func(
            q, k, v, dropout_p=0.0, softmax_scale=None, causal=False
        )
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        print(
            f"d={d}: first={(t1 - t0) * 1e3:.2f} ms "
            f"second={(t2 - t1) * 1e3:.2f} ms "
            f"mean={out.float().mean().item():.6f}"
        )


if __name__ == "__main__":
    main()
