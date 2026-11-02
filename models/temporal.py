"""Arrange spatial blocks into temporal sequences for DeMamba."""

import math

import torch


def temporal_blocks(tokens, block_size):
    """[B,T,H*W,C] -> [B*(H/N)*(W/N),T,N*N,C], N=block_size.

    Each sequence follows one spatial block across frames. Within a block,
    patches follow a column-wise snake, matching DeMamba's scan convention.
    """
    b, t, patches, c = tokens.shape
    side = math.isqrt(patches)
    if side * side != patches or block_size < 1 or side % block_size:
        raise ValueError("Patch grid must be square and divisible by block_size")
    groups = side // block_size
    blocks = tokens.reshape(b, t, groups, block_size, groups, block_size, c)
    blocks = blocks.permute(0, 2, 4, 1, 3, 5, 6).contiguous()
    blocks = blocks.reshape(b * groups * groups, t, block_size * block_size, c)
    order = [
        row * block_size + col
        for col in range(block_size)
        for row in (range(block_size) if col % 2 == 0 else range(block_size - 1, -1, -1))
    ]
    return blocks.index_select(2, torch.tensor(order, device=tokens.device))
