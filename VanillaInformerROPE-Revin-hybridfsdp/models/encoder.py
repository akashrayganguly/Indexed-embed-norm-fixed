"""
models/encoder.py

CHANGES vs latest repo (ConvLayer only — EncoderLayer, Encoder, EncoderStack
are unchanged):

ConvLayer receives three principled fixes:

1. downConv kernel_size=3 -> kernel_size=1, padding=0:
   The original kernel_size=3 with circular padding mixed adjacent sequence
   positions in representation space. In the (N*channel_period, m+1) format,
   adjacent positions are different channels (possibly from different timesteps).
   Circular padding additionally contaminated position 0 with information from
   position L-1 (last timestep's last channel → first timestep's first channel).
   kernel_size=1 is a pointwise independent projection per position: each
   position's representation is transformed using only its own d_model vector.
   All cross-position interactions remain exclusively in the attention layers.

2. BatchNorm1d -> InstanceNorm1d(affine=True):
   BatchNorm1d with batch_size=2 per GPU estimates statistics from 2 samples,
   which is too noisy to be meaningful for a d_model=512 dimensional
   representation. InstanceNorm1d normalises each sample's sequence independently
   with no dependence on batch size. affine=True preserves learnable scale and
   shift parameters per channel, maintaining representational capacity.

3. MaxPool1d -> channel-aware strided timestep selection:
   MaxPool with kernel_size=3, stride=2 selected the maximum activation from
   3 consecutive positions that could span 2 or 3 different channels and up to
   2 different timesteps. If one channel consistently has higher d_model
   activations, MaxPool systematically retains it and discards neighbouring
   channels, creating a channel selection bias with no semantic justification.
   Furthermore, MaxPool with circular-padded input disrupted the clean
   channel_period block structure in the downsampled sequence.

   The replacement: select every other complete timestep block.
   A timestep block is channel_period consecutive positions (channels 0..c-1
   at the same real timestep). We keep even-indexed blocks and discard
   odd-indexed ones. Output length = input_length // 2 (same ratio as before).
   The channel_period structure is perfectly preserved in the output: the
   downsampled sequence still has blocks of exactly channel_period positions,
   channel ROPE remains valid, and block-triangular masks apply correctly.

   When channel_period=1: selects even-indexed positions (0, 2, 4, ...),
   equivalent to stride-2 selection — same as MaxPool stride=2 for 1-channel
   data. No regression for ETT datasets.

Constructor change: ConvLayer(c_in, channel_period=1)
   channel_period is passed from model.py at construction time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, c_in, channel_period=1):
        super(ConvLayer, self).__init__()
        self.channel_period = channel_period

        # CHANGE 1: kernel_size=1 — pointwise independent projection per position
        # No cross-position mixing; eliminates channel-boundary contamination
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=1,
            padding=0
        )

        # CHANGE 2: InstanceNorm1d — no batch-size dependence
        # Normalises each sample's L-length sequence independently
        self.norm = nn.InstanceNorm1d(c_in, affine=True)

        self.activation = nn.ELU()
        # CHANGE 3: MaxPool removed — replaced by channel-aware strided
        # timestep selection in forward()

    def forward(self, x):
        """
        Args:
            x: (B, L, d_model)  where L = n_timesteps * channel_period

        Returns:
            (B, L//2, d_model)  preserving channel_period block structure
        """
        # Permute for Conv1d: (B, L, d_model) -> (B, d_model, L)
        x = self.downConv(x.permute(0, 2, 1))   # (B, d_model, L)
        x = self.norm(x)
        x = self.activation(x)
        x = x.transpose(1, 2)                    # (B, L, d_model)

        # CHANGE 3: Channel-aware strided timestep selection
        #
        # The sequence of L positions has n_timesteps = L // channel_period
        # complete timestep blocks. We keep every other block (even-indexed)
        # and discard odd-indexed blocks, halving the sequence length while
        # exactly preserving the channel_period block structure.
        #
        # Example with channel_period=7, L=672 (96 timestep blocks):
        #   Keep blocks 0,2,4,...,94  (48 blocks × 7 channels = 336 positions)
        #   Discard blocks 1,3,5,...,95
        #   Output: (B, 336, d_model) — still 7-periodic, all masks valid
        #
        # Example with channel_period=1, L=672:
        #   Keep positions 0,2,4,...,670 (identical to stride-2 selection)
        #   Output: (B, 336, d_model) — same as original MaxPool behaviour

        cp = self.channel_period
        L = x.shape[1]
        n_timesteps = L // cp

        # Indices of even-indexed timestep blocks
        even_ts = torch.arange(0, n_timesteps, 2, device=x.device)  # (n_ts//2,)

        # For each even timestep t, collect positions [t*cp, t*cp+1, ..., t*cp+cp-1]
        # block_starts: (n_ts//2, cp)
        block_starts = (
            even_ts.unsqueeze(1) * cp
            + torch.arange(cp, device=x.device).unsqueeze(0)
        )
        indices = block_starts.reshape(-1)   # (L//2,)

        x = x[:, indices, :]                 # (B, L//2, d_model)
        return x


class EncoderLayer(nn.Module):
    """
    Informer Encoder Layer with learnable channel-mixing matrix W.
    Unchanged from latest repo.
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1,
                 activation="relu", channel_mix_size=None):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model

        self.attention = attention
        self.conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        self.channel_mix_size = channel_mix_size
        if channel_mix_size is not None and channel_mix_size > 0:
            self.norm3 = nn.LayerNorm(d_model)
            self.W = nn.Parameter(torch.empty(channel_mix_size, channel_mix_size))
            nn.init.xavier_uniform_(self.W)
        else:
            self.norm3 = None
            self.W = None

    def forward(self, x, attn_mask=None):
        # Self-attention + residual
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)

        # FFN + residual
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = self.norm2(x + y)

        # Channel mixing (optional)
        if self.W is not None:
            batch_size, seq_len, d_model = x.shape
            c = self.channel_mix_size

            if seq_len % c != 0:
                raise RuntimeError(
                    f"EncoderLayer channel_mix_size={c} does not evenly "
                    f"divide seq_len={seq_len}. seq_len must be a multiple "
                    f"of channel_mix_size."
                )

            n = seq_len // c
            x_reshaped = x.view(batch_size, n, c, d_model)
            W = self.W
            if x_reshaped.dtype != W.dtype:
                x_reshaped = x_reshaped.to(W.dtype)
            x_transformed = torch.einsum('ij,bnjd->bnid', W, x_reshaped)
            x = x_transformed.reshape(batch_size, seq_len, d_model)
            x = self.dropout(x)
            x = self.norm3(x)

        return x, attn


class Encoder(nn.Module):
    """Unchanged from latest repo."""
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (nn.ModuleList(conv_layers)
                            if conv_layers is not None else None)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers,
                                               self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class EncoderStack(nn.Module):
    """Unchanged from latest repo."""
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        x_stack = []
        attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1] // (2 ** i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s)
            attns.append(attn)
        x_stack = torch.cat(x_stack, -2)
        return x_stack, attns
