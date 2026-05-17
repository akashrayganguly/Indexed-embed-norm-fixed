"""
models/embed.py

CHANGE vs latest repo (TokenEmbedding only — everything else unchanged):

TokenEmbedding: kernel_size=3, padding=1, padding_mode='circular'
             -> kernel_size=1, padding=0

REASON:
    In the (N*channel_period, m+1) restructured format, adjacent sequence
    positions j and j+1 are different channels (or a channel-boundary crossing
    into the next timestep). A Conv1d with kernel_size=3 therefore mixes three
    consecutive positions that may represent three different channels spanning
    two different timesteps. The circular padding made this worse: position 0
    received a contribution from position seq_len-1 (last channel of the last
    timestep → first channel of the first timestep), mixing temporally and
    channel-wise distant information at the very first token.

    kernel_size=1 is a pointwise independent linear projection per position:
        embedding[j] = W @ x[j] + b
    Each position's d_model representation is derived exclusively from its
    own (m+1)-dimensional feature vector. No neighbourhood mixing occurs.
    All cross-position interactions — both temporal (same channel, different
    timesteps) and cross-channel (same timestep, different channels) — are
    then handled exclusively by the attention mechanism and the channel mixing
    matrix W, which is the architecturally principled place for them.

    The Kaiming initialisation loop is unchanged and applies correctly to
    kernel_size=1 Conv1d weights.

    When channel_period=1 (ETT standard format): positions are already
    distinct timesteps of the same multivariate series, so kernel_size=1 is
    equally correct and produces a faster embedding with no information loss.

All other classes (RotaryPositionalEmbedding, RotaryPositionalEmbeddingFixed,
RotaryChannelEmbeddingLearnable, RotaryChannelEmbeddingFixed,
PositionalEmbedding, FixedEmbedding, TemporalEmbedding, TimeFeatureEmbedding,
DataEmbedding) are UNCHANGED from the latest repo.
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
    FSDP-safe rotate_half: uses flatten(-2) instead of .view().
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
# ROTARY POSITIONAL EMBEDDING — LEARNABLE  (unchanged)
# =============================================================================

class RotaryPositionalEmbedding(nn.Module):
    """Learnable Rotary Positional Embedding with timestep-granularity indexing."""

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
        batch, seq_len, d_model = x.size()

        max_index = (seq_len - 1) // self.channel_period
        if max_index >= self.max_len:
            raise ValueError(
                f"Timestep index {max_index} >= max_len {self.max_len}. "
                f"Increase max_len to at least {max_index + 1}."
            )

        timestep_indices = torch.arange(
            seq_len, device=x.device) // self.channel_period

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
# ROTARY POSITIONAL EMBEDDING — FIXED  (unchanged)
# =============================================================================

class RotaryPositionalEmbeddingFixed(nn.Module):
    """Fixed (non-learnable) Rotary Positional Embedding with timestep-granularity."""

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

        timestep_indices = torch.arange(
            seq_len, device=x.device) // self.channel_period

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
# ROTARY CHANNEL EMBEDDING — LEARNABLE  (unchanged)
# =============================================================================

class RotaryChannelEmbeddingLearnable(nn.Module):
    """Learnable Rotary Channel Embedding with per-sample phase correction."""

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

        sin_embed, cos_embed = _compute_rope_embeddings(
            channel_period, d_model, base)
        self.sin_embed = nn.Parameter(sin_embed, requires_grad=True)
        self.cos_embed = nn.Parameter(cos_embed, requires_grad=True)

    def forward(self, x: torch.Tensor, phase_offset=None) -> torch.Tensor:
        batch, seq_len, d_model = x.size()
        j = torch.arange(seq_len, device=x.device, dtype=torch.long)

        if phase_offset is None:
            positions = j % self.channel_period
            sin_e = self.sin_embed[positions].repeat_interleave(
                2, dim=-1).unsqueeze(0)
            cos_e = self.cos_embed[positions].repeat_interleave(
                2, dim=-1).unsqueeze(0)
        else:
            if not isinstance(phase_offset, torch.Tensor):
                phase_offset = torch.tensor(
                    phase_offset, device=x.device, dtype=torch.long)
            else:
                phase_offset = phase_offset.to(device=x.device, dtype=torch.long)
            if phase_offset.dim() == 0:
                phase_offset = phase_offset.unsqueeze(0).expand(batch)

            positions = (
                phase_offset.unsqueeze(1) + j.unsqueeze(0)
            ) % self.channel_period
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
# ROTARY CHANNEL EMBEDDING — FIXED  (unchanged)
# =============================================================================

class RotaryChannelEmbeddingFixed(nn.Module):
    """Fixed (non-learnable) Rotary Channel Embedding with per-sample phase correction."""

    def __init__(self, c_in: int, d_model: int,
                 channel_period: int = 321, max_len: int = 2000,
                 base: float = 50000.0):
        super().__init__()
        assert d_model % 2 == 0
        self.d_model = d_model
        self.c_in = c_in
        self.channel_period = channel_period
        self.base = base

        sin_embed, cos_embed = _compute_rope_embeddings(
            channel_period, d_model, base)
        self.register_buffer("sin_embed", sin_embed, persistent=True)
        self.register_buffer("cos_embed", cos_embed, persistent=True)

    def forward(self, x: torch.Tensor, phase_offset=None) -> torch.Tensor:
        batch, seq_len, d_model = x.size()
        j = torch.arange(seq_len, device=x.device, dtype=torch.long)

        if phase_offset is None:
            positions = j % self.channel_period
            sin_e = self.sin_embed[positions].repeat_interleave(
                2, dim=-1).unsqueeze(0)
            cos_e = self.cos_embed[positions].repeat_interleave(
                2, dim=-1).unsqueeze(0)
        else:
            if not isinstance(phase_offset, torch.Tensor):
                phase_offset = torch.tensor(
                    phase_offset, device=x.device, dtype=torch.long)
            else:
                phase_offset = phase_offset.to(device=x.device, dtype=torch.long)
            if phase_offset.dim() == 0:
                phase_offset = phase_offset.unsqueeze(0).expand(batch)

            positions = (
                phase_offset.unsqueeze(1) + j.unsqueeze(0)
            ) % self.channel_period
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
# STANDARD EMBEDDINGS
# =============================================================================

class PositionalEmbedding(nn.Module):
    """Unchanged."""
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
    """
    Token Embedding: projects c_in features to d_model dimensions.

    CHANGE: kernel_size=3, padding=1, padding_mode='circular'
         -> kernel_size=1, padding=0

    kernel_size=1 is a pointwise independent projection per sequence position.
    No cross-position mixing. All inter-position interactions handled by
    subsequent attention and channel mixing layers.
    """
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        # CHANGE: kernel_size=1, padding=0 — pointwise independent projection
        # Removed: kernel_size=3, padding=1, padding_mode='circular'
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=1,
            padding=0
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class FixedEmbedding(nn.Module):
    """Unchanged."""
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
    """Unchanged."""
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
        minute_x = (self.minute_embed(x[:, :, 4])
                    if hasattr(self, 'minute_embed') else 0.)
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    """Unchanged."""
    def __init__(self, d_model: int, embed_type: str = 'timeF', freq: str = 'h'):
        super().__init__()
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


# =============================================================================
# MAIN DATA EMBEDDING  (unchanged)
# =============================================================================

class DataEmbedding(nn.Module):
    """
    Combined Data Embedding. Unchanged from latest repo.

    The TokenEmbedding it uses now has kernel_size=1, but DataEmbedding
    itself has no changes. The rest of the pipeline (positional ROPE indexed
    by j//channel_period, channel ROPE indexed by (phase_offset+j)%cp) is
    identical to the latest repo.
    """

    def __init__(self, c_in: int, d_model: int,
                 embed_type: str = 'fixed', freq: str = 'h',
                 dropout: float = 0.1, channel_period: int = 321,
                 max_len: int = 200000):
        super().__init__()
        self.c_in = c_in
        self.d_model = d_model
        self.channel_period = channel_period

        # TokenEmbedding now uses kernel_size=1
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(
            d_model=d_model, max_len=max_len)

        if embed_type != 'timeF':
            self.temporal_embedding = TemporalEmbedding(
                d_model=d_model, embed_type=embed_type, freq=freq)
        else:
            self.temporal_embedding = TimeFeatureEmbedding(
                d_model=d_model, embed_type=embed_type, freq=freq)

        self.rpe = RotaryPositionalEmbedding(
            d_model=d_model, max_len=max_len, channel_period=channel_period)
        self.rpe_fixed = RotaryPositionalEmbeddingFixed(
            d_model=d_model, max_len=max_len, channel_period=channel_period)

        self.fixed_channel_embedding = RotaryChannelEmbeddingFixed(
            c_in=c_in, d_model=d_model, channel_period=channel_period)
        self.learnable_channel_embedding = RotaryChannelEmbeddingLearnable(
            c_in=c_in, d_model=d_model, channel_period=channel_period)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor,
                phase_offset=None) -> torch.Tensor:
        x = self.value_embedding(x)
        x = self.rpe(x) + self.rpe_fixed(x)
        x = (self.fixed_channel_embedding(x, phase_offset) +
             self.learnable_channel_embedding(x, phase_offset))
        return self.dropout(x)
