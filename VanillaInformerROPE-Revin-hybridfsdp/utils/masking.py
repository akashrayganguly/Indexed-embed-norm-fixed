"""
utils/masking.py

CHANGES vs latest repo:

1. Added BlockTriangularCausalMask:
   Replaces TriangularCausalMask for the (N*channel_period, m+1) restructured
   data format. Standard triangular masking blocks position i from attending to
   any position j > i, which imposes a spurious ordering among channels at the
   same real timestep. In the restructured format, channel_period consecutive
   positions all belong to the same real timestep and should be able to attend
   to each other freely.

   Block-triangular rule:
       Position i can attend to position j iff
           floor(j / channel_period) <= floor(i / channel_period)
   Within the same timestep block: all attention permitted.
   Across timestep blocks: only earlier or equal blocks visible.
   When channel_period=1: reduces exactly to TriangularCausalMask (no regression).

2. Updated ProbMask:
   Added channel_period argument. Replaces the .triu(1) base mask with the
   block-triangular pattern so that ProbAttention's sparse selected queries
   also use the correct masking structure. When channel_period=1, produces
   identical result to the original .triu(1) implementation.

TriangularCausalMask is kept unchanged for backward compatibility.
"""

import torch


class TriangularCausalMask():
    """Original triangular causal mask. Unchanged. Used when channel_period=1."""
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask


class BlockTriangularCausalMask():
    """
    Block-triangular causal mask for (N*channel_period, m+1) restructured data.

    In the restructured format, positions j and j+1 are consecutive channels at
    the same real timestep (or channel 6 transitioning to channel 0 of the next
    timestep). The standard triangular mask treats all seq_len positions as a
    flat causal sequence, preventing channel 0 at timestep t from attending to
    channels 1..6 at the same timestep t. This destroys the cross-channel
    within-timestep information that is the primary motivation for the
    restructured format.

    The block-triangular mask allows full attention within each timestep block
    and causal attention across blocks:

        blocked when: floor(j / channel_period) > floor(i / channel_period)
        i.e. key position j is in a strictly later timestep than query position i

    Shape: (B, 1, L, L), dtype bool, True = masked (blocked).

    When channel_period=1:
        floor(j/1) > floor(i/1) is equivalent to j > i,
        which is exactly TriangularCausalMask. No regression for ETT datasets.
    """
    def __init__(self, B, L, channel_period=1, device="cpu"):
        with torch.no_grad():
            # Timestep block index for each of the L positions
            # block_idx[j] = which timestep block position j belongs to
            block_idx = torch.arange(L, device=device) // channel_period  # (L,)

            # mask[i, j] = True when key block > query block
            # i.e. j is in a strictly later timestep block than i → blocked
            mask_2d = (block_idx.unsqueeze(0) > block_idx.unsqueeze(1))  # (L, L)

            # Expand to (B, 1, L, L) matching TriangularCausalMask format
            self._mask = mask_2d.unsqueeze(0).unsqueeze(0).expand(
                B, 1, L, L
            ).clone()

    @property
    def mask(self):
        return self._mask


class ProbMask():
    """
    Sparse attention mask for ProbAttention.

    CHANGE vs original:
        Added channel_period argument (default=1 preserves original behaviour).
        Base mask is now block-triangular instead of upper-triangular (.triu(1)).

        Original:
            _mask = torch.ones(L, L_K).triu(1)
            Blocked when j > i (standard causal).

        New:
            _mask[i, j] = True when floor(j/cp) > floor(i/cp)
            Blocked when j is in a strictly later timestep block than i.
            When cp=1: floor(j/1) > floor(i/1) ↔ j > i, identical to .triu(1).

        channel_period is stored in ProbAttention and passed here from
        ProbAttention._update_context.
    """
    def __init__(self, B, H, L, index, scores, device="cpu", channel_period=1):
        L_K = scores.shape[-1]

        # Block index for each query position (L rows) and key position (L_K cols)
        q_block = torch.arange(L, device=device) // channel_period    # (L,)
        k_block = torch.arange(L_K, device=device) // channel_period  # (L_K,)

        # Base block-triangular mask: (L, L_K)
        # _mask[i, j] = True when key block > query block (blocked)
        _mask = (k_block.unsqueeze(0) > q_block.unsqueeze(1))         # (L, L_K)

        # Expand to (B, H, L, L_K) and slice to selected query rows (index)
        _mask_ex = _mask[None, None, :].expand(B, H, L, L_K)
        indicator = _mask_ex[
            torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None],
            index, :
        ].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask
