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
    """
    CHANGES vs previous version:
    1. RevIN constructor now receives channel_period so it can compute
       per-channel statistics (one per channel in the (N*c, m+1) structure)
       rather than mixed statistics across all channels.
    2. Both revin calls (norm and denorm) now pass enc_phase so RevIN
       knows which channel is at each sequence position.
    3. The partial denorm if/else branch is removed. RevIN._denormalize
       now handles c_out < enc_in internally by slicing mean_per_channel
       to the first c_out features. model.py is cleaner as a result.
    """
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2,
                 d_ff=512, dropout=0.0, attn='prob', embed='fixed', freq='h',
                 activation='gelu', output_attention=False, distil=True,
                 mix=True, device=torch.device('cuda:0'), use_revin=True,
                 channel_mix_size=None, channel_period=1, max_len=200000):
        super(Informer, self).__init__()
        self.pred_len = out_len
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

        Attn = ProbAttention if attn == 'prob' else FullAttention

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attn(False, factor, attention_dropout=dropout,
                             output_attention=output_attention),
                        d_model, n_heads, mix=False),
                    d_model, d_ff, dropout=dropout, activation=activation,
                    channel_mix_size=channel_mix_size,
                ) for _ in range(e_layers)
            ],
            [ConvLayer(d_model) for _ in range(e_layers - 1)] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        Attn(True, factor, attention_dropout=dropout,
                             output_attention=False),
                        d_model, n_heads, mix=mix),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False),
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
            # ── CHANGE: pass channel_period so RevIN computes per-channel
            # statistics for the (N*c, m+1) restructured data ─────────────
            self.revin = RevIN(enc_in, channel_period=channel_period)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,
                enc_phase=None, dec_phase=None):
        """
        Args:
            enc_phase: (B,) LongTensor — channel at encoder position 0.
                       Passed to RevIN so it knows which channel each row
                       belongs to when computing per-channel statistics.
            dec_phase: (B,) LongTensor — channel at decoder position 0.
                       Used for channel embedding in dec_embedding.
        """
        if self.use_revin:
            # ── CHANGE: pass enc_phase for per-channel normalization ───────
            x_enc = self.revin(x_enc, 'norm', enc_phase=enc_phase)

        # Phase-corrected encoder embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc, phase_offset=enc_phase)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # Phase-corrected decoder embedding
        dec_out = self.dec_embedding(x_dec, x_mark_dec, phase_offset=dec_phase)
        dec_out = self.decoder(dec_out, enc_out,
                               x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        if self.use_revin:
            # ── CHANGE: pass enc_phase for per-channel denormalization.
            # RevIN._denormalize handles c_out < enc_in internally —
            # prediction step k has channel (enc_phase + k) % channel_period
            # (valid since seq_len % channel_period == 0 by construction).
            # The old partial-denorm if/else branch is no longer needed. ───
            dec_out = self.revin(
                dec_out[:, -self.pred_len:, :], 'denorm', enc_phase=enc_phase
            )
            return dec_out if not self.output_attention else (dec_out, attns)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]


class InformerStack(nn.Module):
    """
    InformerStack with the same per-channel RevIN changes as Informer.
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

        Attn = ProbAttention if attn == 'prob' else FullAttention
        inp_lens = list(range(len(e_layers)))
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            Attn(False, factor, attention_dropout=dropout,
                                 output_attention=output_attention),
                            d_model, n_heads, mix=False),
                        d_model, d_ff, dropout=dropout, activation=activation,
                        channel_mix_size=channel_mix_size,
                    ) for _ in range(el)
                ],
                [ConvLayer(d_model) for _ in range(el - 1)] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers
        ]
        self.encoder = EncoderStack(encoders, inp_lens)

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        Attn(True, factor, attention_dropout=dropout,
                             output_attention=False),
                        d_model, n_heads, mix=mix),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False),
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

        enc_out = self.enc_embedding(x_enc, x_mark_enc, phase_offset=enc_phase)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec, phase_offset=dec_phase)
        dec_out = self.decoder(dec_out, enc_out,
                               x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        if self.use_revin:
            dec_out = self.revin(
                dec_out[:, -self.pred_len:, :], 'denorm', enc_phase=enc_phase
            )
            return dec_out if not self.output_attention else (dec_out, attns)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]
