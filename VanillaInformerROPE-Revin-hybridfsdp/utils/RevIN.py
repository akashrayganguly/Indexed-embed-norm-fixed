"""
RevIN (Reversible Instance Normalization) - Per-Channel FSDP & BF16 Optimized

KEY CHANGE vs original:
    Original RevIN computed mean and std over ALL seq_len rows of the input
    window regardless of which channel each row belonged to. With the
    (N*c, m+1) restructured data where channel_period consecutive rows belong
    to different channels, this produced mixed-channel statistics that were
    wrong for every channel except those whose statistics happened to match
    the mixture.

    FIX: per-channel statistics.
        For sample b, position j has channel (enc_phase[b] + j) % channel_period.
        _get_statistics now computes separate mean and std for each of the
        channel_period channels using only that channel's rows (exactly
        seq_len // channel_period rows per channel).

        _normalize applies the correct channel's statistics to each row.
        _denormalize applies the correct channel's statistics to each
        prediction step, where prediction step k has channel
        (enc_phase + k) % channel_period (valid because
        seq_len % channel_period == 0 by construction in create_args).

        _denormalize handles c_out < enc_in internally — it indexes only
        the first c_out features from mean_per_channel — so model.py no
        longer needs a separate partial denorm branch.

    FALLBACK: when channel_period <= 1 or enc_phase is None, original
        behavior is preserved exactly. ETT datasets and any call path
        that omits enc_phase use the original global statistics.

    DISTRIBUTED: all_reduce is applied to mean_per_channel and
        var_per_channel tensors so all FSDP ranks use the same statistics.
"""

import torch
import torch.nn as nn
import torch.distributed as dist


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-3, affine=True,
                 distributed=True, channel_period=1):
        """
        Reversible Instance Normalization with per-channel statistics.

        Args:
            num_features:    Number of features/columns (enc_in = m+1)
            eps:             Numerical stability epsilon (1e-3 for BF16)
            affine:          If True, learnable affine per feature
            distributed:     If True, all_reduce stats across GPUs
            channel_period:  Number of channels c in (N*c, m+1) data.
                             Set to args.channel_period in model constructor.
                             1 = original global statistics (default, ETT safe)
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.distributed = distributed
        self.channel_period = channel_period

        # Per-channel statistics (set during normalize, read during denormalize)
        self.mean_per_channel = None
        self.stdev_per_channel = None

        # Original global statistics (kept for fallback when cp <= 1)
        self.mean = None
        self.stdev = None

        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x, enc_phase=None):
        """
        Compute mean and std, either per-channel or globally.

        Args:
            x:          (B, seq_len, num_features)
            enc_phase:  (B,) LongTensor — channel index at position 0 of
                        each sample. None triggers original global behavior.
        """
        cp = self.channel_period

        if enc_phase is None or cp <= 1:
            # ── ORIGINAL GLOBAL BEHAVIOR ──────────────────────────────────
            dim2reduce = tuple(range(1, x.ndim - 1))
            mean = torch.mean(x, dim=dim2reduce, keepdim=True)
            variance = torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False)

            if self.distributed and dist.is_initialized() and dist.get_world_size() > 1:
                dist.all_reduce(mean, op=dist.ReduceOp.AVG)
                dist.all_reduce(variance, op=dist.ReduceOp.AVG)

            self.mean = mean.detach()
            self.stdev = torch.sqrt(variance + self.eps).detach()
            self.mean_per_channel = None
            self.stdev_per_channel = None
            return

        # ── PER-CHANNEL STATISTICS ────────────────────────────────────────
        B, T, F = x.shape

        # channel_at_pos[b, j] = which channel is at position j of sample b
        # = (enc_phase[b] + j) % channel_period
        j = torch.arange(T, device=x.device)                          # (T,)
        channel_at_pos = (
            enc_phase.to(x.device).unsqueeze(1) + j.unsqueeze(0)
        ) % cp                                                          # (B, T)

        # One-hot channel membership: (B, T, cp)
        # one_hot[b, j, c] = 1 if position j of sample b belongs to channel c
        one_hot = torch.zeros(B, T, cp, device=x.device, dtype=x.dtype)
        one_hot.scatter_(2, channel_at_pos.unsqueeze(-1), 1.0)

        # Count of positions per channel per sample: (B, cp)
        # With T = 96 * cp, count is exactly 96 for every (b, c).
        count = one_hot.sum(dim=1).clamp(min=1.0)                     # (B, cp)

        # Mean per channel: (B, cp, F)
        # bmm: (B, cp, T) × (B, T, F) → (B, cp, F)
        x_sum = torch.bmm(one_hot.transpose(1, 2), x)                 # (B, cp, F)
        mean_per_channel = x_sum / count.unsqueeze(-1)                 # (B, cp, F)

        # Gather mean back to sequence positions for variance computation
        # mean_at_pos[b, j, f] = mean_per_channel[b, channel_at_pos[b,j], f]
        b_idx = torch.arange(B, device=x.device)[:, None].expand(B, T)
        mean_at_pos = mean_per_channel[b_idx, channel_at_pos, :]      # (B, T, F)

        # Variance per channel: (B, cp, F)
        x_centered = x - mean_at_pos                                   # (B, T, F)
        var_sum = torch.bmm(
            one_hot.transpose(1, 2), x_centered ** 2
        )                                                              # (B, cp, F)
        var_per_channel = var_sum / count.unsqueeze(-1)                # (B, cp, F)

        # Synchronize across FSDP ranks
        if self.distributed and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(mean_per_channel, op=dist.ReduceOp.AVG)
            dist.all_reduce(var_per_channel, op=dist.ReduceOp.AVG)

        self.mean_per_channel = mean_per_channel.detach()              # (B, cp, F)
        self.stdev_per_channel = torch.sqrt(
            var_per_channel + self.eps
        ).detach()                                                     # (B, cp, F)

        # Clear global stats to avoid accidental use
        self.mean = None
        self.stdev = None

    def _normalize(self, x, enc_phase=None):
        """
        Normalize x using stored statistics.

        Args:
            x:          (B, seq_len, num_features)
            enc_phase:  (B,) LongTensor — same as passed to _get_statistics
        """
        cp = self.channel_period

        if self.mean_per_channel is None or enc_phase is None or cp <= 1:
            # ── ORIGINAL GLOBAL NORMALIZATION ─────────────────────────────
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = (x * self.affine_weight.view(1, 1, -1)
                     + self.affine_bias.view(1, 1, -1))
            return x

        # ── PER-CHANNEL NORMALIZATION ─────────────────────────────────────
        B, T, F = x.shape

        j = torch.arange(T, device=x.device)
        channel_at_pos = (
            enc_phase.to(x.device).unsqueeze(1) + j.unsqueeze(0)
        ) % cp                                                          # (B, T)

        b_idx = torch.arange(B, device=x.device)[:, None].expand(B, T)

        mean_at_pos = self.mean_per_channel[
            b_idx, channel_at_pos, :
        ]                                                              # (B, T, F)
        stdev_at_pos = self.stdev_per_channel[
            b_idx, channel_at_pos, :
        ]                                                              # (B, T, F)

        x = (x - mean_at_pos) / stdev_at_pos

        if self.affine:
            x = (x * self.affine_weight.view(1, 1, -1)
                 + self.affine_bias.view(1, 1, -1))

        return x

    def _denormalize(self, x, enc_phase=None):
        """
        Reverse normalization using stored statistics.

        Handles c_out < enc_in: if x has fewer features than enc_in,
        only the first x.shape[-1] features of mean_per_channel are used.
        This removes the need for a separate partial-denorm branch in model.py.

        Args:
            x:          (B, pred_len, c_out)  where c_out <= enc_in
            enc_phase:  (B,) LongTensor — channel at encoder position 0.
                        Prediction step k has channel
                        (enc_phase + k) % channel_period
                        (valid because seq_len % channel_period == 0).
        """
        cp = self.channel_period
        F = x.shape[-1]   # c_out (may be < num_features)

        if self.mean_per_channel is None or enc_phase is None or cp <= 1:
            # ── ORIGINAL GLOBAL DENORMALIZATION ───────────────────────────
            if self.affine:
                weight = self.affine_weight[:F].view(1, 1, -1)
                bias = self.affine_bias[:F].view(1, 1, -1)
                x = (x - bias) / (weight + self.eps * self.eps)
            # Use global stats, sliced to F features
            x = x * self.stdev[:, :, :F] + self.mean[:, :, :F]
            return x

        # ── PER-CHANNEL DENORMALIZATION ───────────────────────────────────
        B, pred_len, _ = x.shape

        # Prediction step k has channel (enc_phase + k) % cp
        # Valid because seq_len % cp == 0 → prediction starts at enc_phase
        k_idx = torch.arange(pred_len, device=x.device)
        channel_at_step = (
            enc_phase.to(x.device).unsqueeze(1) + k_idx.unsqueeze(0)
        ) % cp                                                          # (B, pred_len)

        b_idx = torch.arange(B, device=x.device)[:, None].expand(B, pred_len)

        # Slice to first F features (handles c_out < enc_in naturally)
        mean_at_step = self.mean_per_channel[
            b_idx, channel_at_step, :F
        ]                                                              # (B, pred_len, F)
        stdev_at_step = self.stdev_per_channel[
            b_idx, channel_at_step, :F
        ]                                                              # (B, pred_len, F)

        # Reverse affine
        if self.affine:
            weight = self.affine_weight[:F].view(1, 1, -1)
            bias = self.affine_bias[:F].view(1, 1, -1)
            x = (x - bias) / (weight + self.eps * self.eps)

        # Reverse standardization
        x = x * stdev_at_step + mean_at_step

        return x

    def normalize(self, x, enc_phase=None):
        self._get_statistics(x, enc_phase)
        return self._normalize(x, enc_phase)

    def denormalize(self, x, enc_phase=None):
        if self.mean_per_channel is None and self.mean is None:
            raise RuntimeError(
                "RevIN.denormalize() called before normalize(). "
                "Call normalize() first."
            )
        return self._denormalize(x, enc_phase)

    def forward(self, x, mode: str, enc_phase=None):
        """
        Args:
            x:          Input tensor
            mode:       'norm' or 'denorm'
            enc_phase:  (B,) LongTensor — channel at position 0 of encoder
                        window. None triggers original global-stats behavior.
        """
        if mode == 'norm':
            return self.normalize(x, enc_phase)
        elif mode == 'denorm':
            return self.denormalize(x, enc_phase)
        else:
            raise NotImplementedError(
                f"Mode '{mode}' not supported. Use 'norm' or 'denorm'."
            )
