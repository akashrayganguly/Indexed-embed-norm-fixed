"""
embed.py - ROPE Embeddings with Per-Sample Channel Phase Correction
             and Timestep-Granularity Positional Embeddings

FIXES vs previous version:

FIX 1 (channel phase — from previous round):
    RotaryChannelEmbeddingLearnable and RotaryChannelEmbeddingFixed accept a
    per-sample phase_offset so that position j of sample b gets the channel
    embedding for (phase_offset[b] + j) % channel_period rather than always
    starting from channel 0.

FIX 2 (positional embedding granularity — this round):
    RotaryPositionalEmbedding and RotaryPositionalEmbeddingFixed previously
    looked up table row j for window position j, giving every row in the
    (N*c, m+1) data a unique positional embedding. This was wrong: in the
    restructured data, channel_period consecutive rows all belong to the same
    real timestep and should receive the same positional embedding.

    FIX: look up table row j // channel_period instead of j.
    - Positions 0..c-1   → timestep 0 embedding  (all identical) ✓
    - Positions c..2c-1  → timestep 1 embedding  (all identical) ✓
    - etc.

    channel_period is now a constructor argument for both positional embedding
    classes. DataEmbedding already had self.channel_period and now forwards it.

    max_len is unchanged — the maximum lookup index after the fix is
    (seq_len-1) // channel_period which is always smaller than seq_len,
    so any table that was large enough before is still large enough.
    The max_len guard is updated to check the actual maximum index rather
    than seq_len to avoid false-positive errors.

    For ETT datasets where channel_period=1: j // 1 == j, exact same
    behaviour as before, no regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# =============================================================================
# UTILITY FUNCTIONS  (unchanged)
# =============================================================================

def _compute_rope_embeddings(
    max_len: int,
    d_model: int,
    base: float = 10000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert d_model % 2 == 0, f"d_model must be even, got {d_model}"
    inv_freq = 1.0 / (base ** (
        torch.arange(0, d_model, 2, dtype=dtype, device=device) / d_model
    ))
    positions = torch.arange(0, max_len, dtype=dtype, device=device)
    sinusoid_inp = torch.outer(positions, inv_freq)
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def _rotate_half_optimized(x: torch.Tensor) -> torch.Tensor:
    """
    FSDP-safe rotate_half: uses flatten(-2) instead of .view() to avoid
    failures on non-contiguous sharded tensors.
    [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def _apply_rotary_emb(
    x: torch.Tensor,
    sin_embed: torch.Tensor,
    cos_embed: torch.Tensor
) -> torch.Tensor:
    return x * cos_embed + _rotate_half_optimized(x) * sin_embed


# =============================================================================
# ROTARY POSITIONAL EMBEDDING — LEARNABLE  (FIX 2 applied)
# =============================================================================

class RotaryPositionalEmbedding(nn.Module):
    """
    Learnable Rotary Positional Embedding with timestep-granularity indexing.

    CHANGE vs original:
        Constructor now accepts channel_period (default 1 = no change for ETT).
        forward() looks up table row j // channel_period for window position j
        so that all channel_period rows belonging to the same real timestep
        receive identical positional embeddings.

    WHY channel_period=1 is safe for non-restructured data:
        j // 1 == j, so behaviour is identical to the original.
    """

    def __init__(self, d_model: int, max_len: int = 200000,
                 base: float = 10000.0, channel_period: int = 1):
        super().__init__()
        assert d_model % 2 == 0, f"d_model must be even, got {d_model}"
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        self.channel_period = channel_period

        sin_embed, cos_embed = _compute_rope_embeddings(max_len, d_model, base)
        self.sin_embed = nn.Parameter(sin_embed, requires_grad=True)
        self.cos_embed = nn.Parameter(cos_embed, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, d_model)
        Returns:
            (B, seq_len, d_model) with timestep-granularity ROPE applied
        """
        batch, seq_len, d_model = x.size()

        # Maximum table index after the fix: (seq_len-1) // channel_period
        # Always smaller than seq_len, so any previously valid max_len
        # is still valid. Guard updated to check actual max index.
        max_index = (seq_len - 1) // self.channel_period
        if max_index >= self.max_len:
            raise ValueError(
                f"Timestep index {max_index} >= max_len {self.max_len}. "
                f"Increase max_len to at least {max_index + 1}."
            )

        # ── FIX 2: index by real timestep within window, not by row ─────────
        # All channel_period rows of the same timestep get the same embedding.
        # j // channel_period: [0,0,..,0, 1,1,..,1, 2,2,..,2, ...]
        #                       ←c times→  ←c times→
        timestep_indices = torch.arange(seq_len, device=x.device) // self.channel_period
        # ─────────────────────────────────────────────────────────────────────

        sin_base = self.sin_embed[timestep_indices, :]          # (seq_len, d_model/2)
        cos_base = self.cos_embed[timestep_indices, :]

        sin_embed = sin_base.unsqueeze(0).repeat_interleave(2, dim=-1)  # (1, seq_len, d_model)
        cos_embed = cos_base.unsqueeze(0).repeat_interleave(2, dim=-1)

        # DTYPE SAFETY: FSDP may shard parameters as BF16 while x is FP32
        if sin_embed.dtype != x.dtype:
            sin_embed = sin_embed.to(x.dtype)
            cos_embed = cos_embed.to(x.dtype)

        return _apply_rotary_emb(x, sin_embed, cos_embed)

    def extra_repr(self) -> str:
        return (f'd_model={self.d_model}, max_len={self.max_len}, '
                f'channel_period={self.channel_period}')


# =============================================================================
# ROTARY POSITIONAL EMBEDDING — FIXED  (FIX 2 applied)
# =============================================================================

class RotaryPositionalEmbeddingFixed(nn.Module):
    """
    Fixed (non-learnable) Rotary Positional Embedding with timestep-granularity
    indexing. Identical logic to the learnable version; sin/cos are buffers.
    """

    def __init__(self, d_model: int, max_len: int = 200000,
                 base: float = 10000.0, channel_period: int = 1):
        super().__init__()
        assert d_model % 2 == 0, f"d_model must be even, got {d_model}"
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        self.channel_period = channel_period

        sin_embed, cos_embed = _compute_rope_embeddings(max_len, d_model, base)
        self.register_buffer("sin_embed", sin_embed, persistent=True)
        self.register_buffer("cos_embed", cos_embed, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.size()

        max_index = (seq_len - 1) // self.channel_period
        if max_index >= self.max_len:
            raise ValueError(
                f"Timestep index {max_index} >= max_len {self.max_len}. "
                f"Increase max_len to at least {max_index + 1}."
            )

        # ── FIX 2 ────────────────────────────────────────────────────────────
        timestep_indices = torch.arange(seq_len, device=x.device) // self.channel_period
        # ─────────────────────────────────────────────────────────────────────

        sin_base = self.sin_embed[timestep_indices, :]
        cos_base = self.cos_embed[timestep_indices, :]

        sin_embed = sin_base.unsqueeze(0).repeat_interleave(2, dim=-1)
        cos_embed = cos_base.unsqueeze(0).repeat_interleave(2, dim=-1)

        if sin_embed.dtype != x.dtype:
            sin_embed = sin_embed.to(x.dtype)
            cos_embed = cos_embed.to(x.dtype)

        return _apply_rotary_emb(x, sin_embed, cos_embed)

    def extra_repr(self) -> str:
        return (f'd_model={self.d_model}, max_len={self.max_len}, '
                f'channel_period={self.channel_period}')


# =============================================================================
# ROTARY CHANNEL EMBEDDING — LEARNABLE  (FIX 1, unchanged from previous round)
# =============================================================================

class RotaryChannelEmbeddingLearnable(nn.Module):
    """
    Learnable Rotary Channel Embedding with per-sample phase correction.

    forward(x, phase_offset):
        position j of sample b receives the channel embedding for
        (phase_offset[b] + j) % channel_period.
        phase_offset=None falls back to all-zeros (phase 0).
    """

    def __init__(self, c_in: int, d_model: int,
                 channel_period: int = 321, max_len: int = 2000,
                 base: float = 50000.0):
        super().__init__()
        assert d_model % 2 == 0
        self.d_model = d_model
        self.c_in = c_in
        self.channel_period = channel_period
        self.base = base
        self.max_len = max_len

        sin_embed, cos_embed = _compute_rope_embeddings(channel_period, d_model, base)
        self.sin_embed = nn.Parameter(sin_embed, requires_grad=True)
        self.cos_embed = nn.Parameter(cos_embed, requires_grad=True)

    def forward(self, x: torch.Tensor, phase_offset=None) -> torch.Tensor:
        batch, seq_len, d_model = x.size()
        j = torch.arange(seq_len, device=x.device, dtype=torch.long)

        if phase_offset is None:
            positions = j % self.channel_period
            sin_e = self.sin_embed[positions].repeat_interleave(2, dim=-1).unsqueeze(0)
            cos_e = self.cos_embed[positions].repeat_interleave(2, dim=-1).unsqueeze(0)
        else:
            if not isinstance(phase_offset, torch.Tensor):
                phase_offset = torch.tensor(
                    phase_offset, device=x.device, dtype=torch.long)
            else:
                phase_offset = phase_offset.to(device=x.device, dtype=torch.long)
            if phase_offset.dim() == 0:
                phase_offset = phase_offset.unsqueeze(0).expand(batch)

            positions = (phase_offset.unsqueeze(1) + j.unsqueeze(0)) % self.channel_period
            sin_e = self.sin_embed[positions].repeat_interleave(2, dim=-1)
            cos_e = self.cos_embed[positions].repeat_interleave(2, dim=-1)

        if sin_e.dtype != x.dtype:
            sin_e = sin_e.to(x.dtype)
            cos_e = cos_e.to(x.dtype)

        return _apply_rotary_emb(x, sin_e, cos_e)

    def extra_repr(self) -> str:
        return (f'd_model={self.d_model}, c_in={self.c_in}, '
                f'period={self.channel_period}, base={self.base}')


# =============================================================================
# ROTARY CHANNEL EMBEDDING — FIXED  (FIX 1, unchanged from previous round)
# =============================================================================

class RotaryChannelEmbeddingFixed(nn.Module):
    """
    Fixed (non-learnable) Rotary Channel Embedding with per-sample phase
    correction. Identical logic to Learnable; sin/cos are buffers.
    """

    def __init__(self, c_in: int, d_model: int,
                 channel_period: int = 321, max_len: int = 2000,
                 base: float = 50000.0):
        super().__init__()
        assert d_model % 2 == 0
        self.d_model = d_model
        self.c_in = c_in
        self.channel_period = channel_period
        self.base = base

        sin_embed, cos_embed = _compute_rope_embeddings(channel_period, d_model, base)
        self.register_buffer("sin_embed", sin_embed, persistent=True)
        self.register_buffer("cos_embed", cos_embed, persistent=True)

    def forward(self, x: torch.Tensor, phase_offset=None) -> torch.Tensor:
        batch, seq_len, d_model = x.size()
        j = torch.arange(seq_len, device=x.device, dtype=torch.long)

        if phase_offset is None:
            positions = j % self.channel_period
            sin_e = self.sin_embed[positions].repeat_interleave(2, dim=-1).unsqueeze(0)
            cos_e = self.cos_embed[positions].repeat_interleave(2, dim=-1).unsqueeze(0)
        else:
            if not isinstance(phase_offset, torch.Tensor):
                phase_offset = torch.tensor(
                    phase_offset, device=x.device, dtype=torch.long)
            else:
                phase_offset = phase_offset.to(device=x.device, dtype=torch.long)
            if phase_offset.dim() == 0:
                phase_offset = phase_offset.unsqueeze(0).expand(batch)

            positions = (phase_offset.unsqueeze(1) + j.unsqueeze(0)) % self.channel_period
            sin_e = self.sin_embed[positions].repeat_interleave(2, dim=-1)
            cos_e = self.cos_embed[positions].repeat_interleave(2, dim=-1)

        if sin_e.dtype != x.dtype:
            sin_e = sin_e.to(x.dtype)
            cos_e = cos_e.to(x.dtype)

        return _apply_rotary_emb(x, sin_e, cos_e)

    def extra_repr(self) -> str:
        return (f'd_model={self.d_model}, c_in={self.c_in}, '
                f'period={self.channel_period}, base={self.base}')


# =============================================================================
# STANDARD EMBEDDINGS  (unchanged)
# =============================================================================

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200000):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, :x.size(1), :]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in, out_channels=d_model,
            kernel_size=3, padding=padding, padding_mode='circular'
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class FixedEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        w = torch.zeros(c_in, d_model, dtype=torch.float32)
        position = torch.arange(0, c_in, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model: int, embed_type: str = 'fixed', freq: str = 'h'):
        super().__init__()
        minute_size, hour_size, weekday_size, day_size, month_size = 4, 24, 7, 32, 13
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model: int, embed_type: str = 'timeF', freq: str = 'h'):
        super().__init__()
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


# =============================================================================
# MAIN DATA EMBEDDING  (channel_period now forwarded to positional embeddings)
# =============================================================================

class DataEmbedding(nn.Module):
    """
    Combined Data Embedding with phase-corrected channel ROPE and
    timestep-granularity positional ROPE.

    CHANGES vs previous round:
        rpe and rpe_fixed now receive channel_period so that forward()
        indexes the sin/cos table by j // channel_period rather than j.
        DataEmbedding already stored self.channel_period; it is now also
        forwarded to the positional embedding constructors.

    All other logic unchanged from the previous channel-phase-correction round.
    """

    def __init__(self, c_in: int, d_model: int,
                 embed_type: str = 'fixed', freq: str = 'h',
                 dropout: float = 0.1, channel_period: int = 321,
                 max_len: int = 200000):
        super().__init__()
        self.c_in = c_in
        self.d_model = d_model
        self.channel_period = channel_period

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model, max_len=max_len)

        if embed_type != 'timeF':
            self.temporal_embedding = TemporalEmbedding(
                d_model=d_model, embed_type=embed_type, freq=freq)
        else:
            self.temporal_embedding = TimeFeatureEmbedding(
                d_model=d_model, embed_type=embed_type, freq=freq)

        # ── FIX 2: pass channel_period to both positional embeddings ─────────
        self.rpe = RotaryPositionalEmbedding(
            d_model=d_model, max_len=max_len, channel_period=channel_period)
        self.rpe_fixed = RotaryPositionalEmbeddingFixed(
            d_model=d_model, max_len=max_len, channel_period=channel_period)
        # ─────────────────────────────────────────────────────────────────────

        self.fixed_channel_embedding = RotaryChannelEmbeddingFixed(
            c_in=c_in, d_model=d_model, channel_period=channel_period)
        self.learnable_channel_embedding = RotaryChannelEmbeddingLearnable(
            c_in=c_in, d_model=d_model, channel_period=channel_period)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor,
                phase_offset=None) -> torch.Tensor:
        """
        Args:
            x:            (B, seq_len, c_in)
            x_mark:       (B, seq_len, time_features) — kept for API compat,
                          not used in ROPE mode (same as original)
            phase_offset: None | int | (B,) LongTensor
                          Channel phase for this batch window.

        Embedding pipeline:
            1. TokenEmbedding: project c_in -> d_model
            2. rpe + rpe_fixed: positional ROPE indexed by j // channel_period
               → all channels at the same real timestep get identical pos emb
            3. channel fixed + channel learnable: ROPE indexed by
               (phase_offset + j) % channel_period
               → each position correctly identifies which channel it is
        """
        x = self.value_embedding(x)

        # Step 2: positional ROPE — timestep granularity (FIX 2)
        x = self.rpe(x) + self.rpe_fixed(x)

        # Step 3: channel ROPE — phase-corrected (FIX 1)
        x = (self.fixed_channel_embedding(x, phase_offset) +
             self.learnable_channel_embedding(x, phase_offset))

        return self.dropout(x)
