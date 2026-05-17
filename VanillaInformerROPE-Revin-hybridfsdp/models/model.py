"""
models/model.py

CHANGES vs latest repo (both Informer and InformerStack):

1. Store self.label_len = label_len in __init__:
   Required for the dec_inp label normalization in forward(). Previously
   label_len was not stored as an instance variable.

2. Normalize decoder label portion through RevIN in forward():
   In the (N*cp, m+1) format, x_dec arrives with its label portion in
   StandardScaler space (from batch_y in _process_one_batch) and its pred
   portion as zeros. The encoder input x_enc has been RevIN-normalized.
   The decoder cross-attends enc_out (RevIN space) with its own embeddings.
   If the label portion of x_dec is in a different space from x_enc, the
   cross-attention compares representations from mismatched distributions.

   FIX: after revin(x_enc, 'norm', enc_phase), use the stored per-channel
   statistics to also normalize the label portion of x_dec via
   revin._normalize(x_dec[:, :self.label_len, :], enc_phase=dec_phase).
   The pred-portion zeros are left unchanged (zero is already a valid
   neutral value in the normalized space).

   dec_phase is used (not enc_phase) to correctly index which channel is
   at each decoder label position. Since seq_len and label_len are both
   multiples of channel_period in the current config, dec_phase == enc_phase
   numerically, but using dec_phase is semantically correct and future-proof.

3. Pass channel_period to all Attn constructors:
   FullAttention and ProbAttention now accept channel_period to construct
   the correct BlockTriangularCausalMask. Previously they received no
   channel_period and defaulted to standard triangular masking.

4. Pass channel_period to ConvLayer constructors:
   ConvLayer now accepts channel_period for channel-aware strided timestep
   selection. The construction call [ConvLayer(d_model) for ...] becomes
   [ConvLayer(d_model, channel_period=channel_period) for ...].

All four changes are mirrored identically in InformerStack.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding
from utils.RevIN import RevIN


class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2,
                 d_ff=512, dropout=0.0, attn='prob', embed='fixed', freq='h',
                 activation='gelu', output_attention=False, distil=True,
                 mix=True, device=torch.device('cuda:0'), use_revin=True,
                 channel_mix_size=None, channel_period=1, max_len=200000):
        super(Informer, self).__init__()
        self.pred_len = out_len
        # CHANGE 1: store label_len for dec_inp normalization in forward()
        self.label_len = label_len
        self.attn = attn
        self.output_attention = output_attention

        self.enc_embedding = DataEmbedding(
            c_in=enc_in, d_model=d_model, embed_type=embed, freq=freq,
            dropout=dropout, channel_period=channel_period, max_len=max_len
        )
        self.dec_embedding = DataEmbedding(
            c_in=dec_in, d_model=d_model, embed_type=embed, freq=freq,
            dropout=dropout, channel_period=channel_period, max_len=max_len
        )

        # CHANGE 3: pass channel_period to attention constructors
        # FullAttention and ProbAttention use it to build BlockTriangularCausalMask
        Attn = ProbAttention if attn == 'prob' else FullAttention

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        # Encoder self-attention: mask_flag=False (no causal mask)
                        Attn(False, factor, attention_dropout=dropout,
                             output_attention=output_attention,
                             channel_period=channel_period),  # CHANGE 3
                        d_model, n_heads, mix=False),
                    d_model, d_ff, dropout=dropout, activation=activation,
                    channel_mix_size=channel_mix_size,
                ) for _ in range(e_layers)
            ],
            # CHANGE 4: pass channel_period to ConvLayer for channel-aware pooling
            [ConvLayer(d_model, channel_period=channel_period)           # CHANGE 4
             for _ in range(e_layers - 1)] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        # Decoder self-attention: mask_flag=True → BlockTriangular
                        Attn(True, factor, attention_dropout=dropout,
                             output_attention=False,
                             channel_period=channel_period),  # CHANGE 3
                        d_model, n_heads, mix=mix),
                    AttentionLayer(
                        # Cross-attention: mask_flag=False (no causal mask)
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False,
                                      channel_period=channel_period),  # CHANGE 3
                        d_model, n_heads, mix=False),
                    d_model, d_ff, dropout=dropout, activation=activation,
                    channel_mix_size=channel_mix_size,
                ) for _ in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

        self.use_revin = use_revin
        if use_revin:
            self.revin = RevIN(enc_in, channel_period=channel_period)
            # affine=False by default in updated RevIN

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,
                enc_phase=None, dec_phase=None):
        """
        Args:
            enc_phase: (B,) LongTensor — channel at encoder position 0.
            dec_phase: (B,) LongTensor — channel at decoder position 0.
                       Used to normalize decoder label portion (CHANGE 2).
        """
        if self.use_revin:
            # Normalize encoder input — stores mean_per_channel, stdev_per_channel
            x_enc = self.revin(x_enc, 'norm', enc_phase=enc_phase)

            # CHANGE 2: normalize decoder label portion using stored statistics.
            # x_dec[:, :self.label_len, :] is the label (StandardScaler space).
            # x_dec[:, self.label_len:, :] is zeros (pred placeholder, stays 0).
            # We normalize the label using dec_phase so each decoder label
            # position is normalized by its own channel's instance statistics.
            # This puts encoder input and decoder label input in the same
            # RevIN-normalized space, enabling coherent cross-attention.
            if self.label_len > 0:
                label_normed = self.revin._normalize(
                    x_dec[:, :self.label_len, :],
                    enc_phase=dec_phase
                )
                x_dec = torch.cat(
                    [label_normed, x_dec[:, self.label_len:, :]], dim=1)

        # Encoder
        enc_out = self.enc_embedding(
            x_enc, x_mark_enc, phase_offset=enc_phase)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # Decoder
        dec_out = self.dec_embedding(
            x_dec, x_mark_dec, phase_offset=dec_phase)
        dec_out = self.decoder(dec_out, enc_out,
                               x_mask=dec_self_mask,
                               cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        if self.use_revin:
            # Denormalize: bring predictions back to StandardScaler space
            # enc_phase used here because prediction step k has channel
            # (enc_phase + k) % cp (valid since seq_len % cp == 0)
            dec_out = self.revin(
                dec_out[:, -self.pred_len:, :],
                'denorm',
                enc_phase=enc_phase
            )
            if self.output_attention:
                return dec_out, attns
            return dec_out

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]


class InformerStack(nn.Module):
    """
    InformerStack with identical changes to Informer:
    1. self.label_len stored
    2. dec_inp label normalized through RevIN
    3. channel_period passed to all attention constructors
    4. channel_period passed to ConvLayer constructors
    """
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=[3, 2, 1],
                 d_layers=2, d_ff=512, dropout=0.0, attn='prob', embed='fixed',
                 freq='h', activation='gelu', output_attention=False,
                 distil=True, mix=True, device=torch.device('cuda:0'),
                 use_revin=True, channel_mix_size=None, channel_period=1,
                 max_len=200000):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        # CHANGE 1: store label_len
        self.label_len = label_len
        self.attn = attn
        self.output_attention = output_attention
        self.use_revin = use_revin

        if use_revin:
            self.revin = RevIN(enc_in, channel_period=channel_period)

        self.enc_embedding = DataEmbedding(
            c_in=enc_in, d_model=d_model, embed_type=embed, freq=freq,
            dropout=dropout, channel_period=channel_period, max_len=max_len
        )
        self.dec_embedding = DataEmbedding(
            c_in=dec_in, d_model=d_model, embed_type=embed, freq=freq,
            dropout=dropout, channel_period=channel_period, max_len=max_len
        )

        # CHANGE 3: channel_period passed to attention constructors
        Attn = ProbAttention if attn == 'prob' else FullAttention
        inp_lens = list(range(len(e_layers)))
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            Attn(False, factor, attention_dropout=dropout,
                                 output_attention=output_attention,
                                 channel_period=channel_period),  # CHANGE 3
                            d_model, n_heads, mix=False),
                        d_model, d_ff, dropout=dropout, activation=activation,
                        channel_mix_size=channel_mix_size,
                    ) for _ in range(el)
                ],
                # CHANGE 4: channel_period to ConvLayer
                [ConvLayer(d_model, channel_period=channel_period)       # CHANGE 4
                 for _ in range(el - 1)] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers
        ]
        self.encoder = EncoderStack(encoders, inp_lens)

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        Attn(True, factor, attention_dropout=dropout,
                             output_attention=False,
                             channel_period=channel_period),  # CHANGE 3
                        d_model, n_heads, mix=mix),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False,
                                      channel_period=channel_period),  # CHANGE 3
                        d_model, n_heads, mix=False),
                    d_model, d_ff, dropout=dropout, activation=activation,
                    channel_mix_size=channel_mix_size,
                ) for _ in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,
                enc_phase=None, dec_phase=None):

        if self.use_revin:
            x_enc = self.revin(x_enc, 'norm', enc_phase=enc_phase)

            # CHANGE 2: normalize decoder label portion
            if self.label_len > 0:
                label_normed = self.revin._normalize(
                    x_dec[:, :self.label_len, :],
                    enc_phase=dec_phase
                )
                x_dec = torch.cat(
                    [label_normed, x_dec[:, self.label_len:, :]], dim=1)

        enc_out = self.enc_embedding(
            x_enc, x_mark_enc, phase_offset=enc_phase)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(
            x_dec, x_mark_dec, phase_offset=dec_phase)
        dec_out = self.decoder(dec_out, enc_out,
                               x_mask=dec_self_mask,
                               cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        if self.use_revin:
            dec_out = self.revin(
                dec_out[:, -self.pred_len:, :],
                'denorm',
                enc_phase=enc_phase
            )
            if self.output_attention:
                return dec_out, attns
            return dec_out

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]
