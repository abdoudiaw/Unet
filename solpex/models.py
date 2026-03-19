# Copyright 2025-2026 Oak Ridge National Laboratory
# @authors: Abdourahmane (Abdou) Diaw - diawa@ornl.gov
#
# SPDX-License-Identifier: MIT

import torch
from torch import nn
import torch.nn.functional as F

def bottleneck_to_z(b: torch.Tensor) -> torch.Tensor:
    # b: (B,C,h,w) -> z: (B,C)
    return torch.flatten(F.adaptive_avg_pool2d(b, 1), 1)

class ParamToZ(nn.Module):
    def __init__(self, P: int, z_dim: int, hidden=(256,256), dropout=0.0, use_layernorm=False):
        super().__init__()
        layers = []
        in_d = P
        for h in hidden:
            layers += [nn.Linear(in_d, h), nn.SiLU()]
            if use_layernorm:
                layers += [nn.LayerNorm(h)]
            if dropout and dropout > 0:
                layers += [nn.Dropout(dropout)]
            in_d = h
        layers += [nn.Linear(in_d, z_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, p):
        return self.net(p)

def z_to_bottleneck(z: torch.Tensor, b_shape) -> torch.Tensor:
    """z: (B,C) -> b: (B,C,h,w) by broadcasting"""
    B, C, h, w = b_shape
    return z.view(B, C, 1, 1).expand(B, C, h, w)


class FiLMGenerator(nn.Module):
    """Generate per-channel scale (gamma) and shift (beta) from scalar params."""
    def __init__(self, P: int, n_channels: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(P, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, n_channels * 2),
        )
        # Initialize near-identity: gamma ≈ 1, beta ≈ 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.n_channels = n_channels

    def forward(self, params):
        """params: (B, P) -> gamma: (B, C, 1, 1), beta: (B, C, 1, 1)"""
        out = self.net(params)  # (B, 2*C)
        gamma, beta = out.split(self.n_channels, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return (1.0 + gamma), beta  # center gamma around 1


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, groups_gn=8):
        super().__init__()
        g = min(groups_gn, out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.SiLU(inplace=True),
        )
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x):
        return self.net(x) + self.shortcut(x)

class UNet(nn.Module):
    """
    UNet with ConvTranspose2d upsampling and optional FiLM conditioning.

    When P > 0, scalar control parameters are injected via FiLM (Feature-wise
    Linear Modulation) at every encoder, bottleneck, and decoder level.
    When P == 0, behaves identically to the original architecture.

    Parameters
    ----------
    depth : int
        Number of encoder/decoder levels (default 3 for backward compatibility).
        Channel multipliers are 1, 2, 4, ..., 2^depth at bottleneck.
    base_loss : str
        Not used here — kept for docs. See losses.py.
    """
    def __init__(self, in_ch=1, out_ch=1, base=32, groups_gn=8, dropout=0.0,
                 P: int = 0, film_hidden: int = 128, z_dim: int = 0,
                 depth: int = 3):
        super().__init__()
        self.dropout = dropout
        self.P = P
        self.z_dim = z_dim
        self.depth = depth
        self.pool = nn.MaxPool2d(2)

        # Build encoder levels: channel mults are 1, 2, 4, ..., 2^(depth-1)
        enc_channels = [base * (2 ** i) for i in range(depth)]
        self.encoders = nn.ModuleList()
        ch_in = in_ch
        for ch_out in enc_channels:
            self.encoders.append(DoubleConv(ch_in, ch_out, groups_gn))
            ch_in = ch_out

        # Bottleneck: 2^depth * base channels
        bot_ch = base * (2 ** depth)
        self.bot = DoubleConv(enc_channels[-1], bot_ch, groups_gn)

        # Build decoder levels (reverse order)
        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        dec_in = bot_ch
        for i in reversed(range(depth)):
            skip_ch = enc_channels[i]
            self.up_convs.append(nn.ConvTranspose2d(dec_in, skip_ch, 2, 2))
            self.decoders.append(DoubleConv(skip_ch + skip_ch, skip_ch, groups_gn))
            self.dropouts.append(nn.Dropout2d(dropout) if dropout > 0 else nn.Identity())
            dec_in = skip_ch

        self.out = nn.Conv2d(enc_channels[0], out_ch, 1)

        # FiLM conditioning layers (only when P > 0)
        if P > 0:
            self.film_enc = nn.ModuleList(
                [FiLMGenerator(P, ch, film_hidden) for ch in enc_channels]
            )
            self.film_bot = FiLMGenerator(P, bot_ch, film_hidden)
            self.film_dec = nn.ModuleList(
                [FiLMGenerator(P, enc_channels[i], film_hidden) for i in reversed(range(depth))]
            )

        # Learned latent projection (only when z_dim > 0)
        if z_dim > 0:
            self.z_proj = nn.Linear(bot_ch, z_dim)
            self.z_unproj = nn.Linear(z_dim, bot_ch)
        else:
            self.z_proj = None
            self.z_unproj = None

    @staticmethod
    def _apply_film(features, film_layer, params):
        """Apply FiLM modulation: features * gamma + beta."""
        gamma, beta = film_layer(params)
        return features * gamma + beta

    def project_z(self, b):
        """Bottleneck -> compressed latent: (B,C,h,w) -> (B, z_dim)."""
        z = bottleneck_to_z(b)
        if self.z_proj is not None:
            z = self.z_proj(z)
        return z

    def unproject_z(self, z_low, b_shape):
        """Compressed latent -> bottleneck: (B, z_dim) -> (B,C,h,w)."""
        if self.z_unproj is not None:
            z_full = self.z_unproj(z_low)
        else:
            z_full = z_low
        return z_to_bottleneck(z_full, b_shape)

    def encode(self, x, params=None):
        skips = []
        h = x
        for i, enc in enumerate(self.encoders):
            h = enc(h if i == 0 else self.pool(h))
            if params is not None and self.P > 0:
                h = self._apply_film(h, self.film_enc[i], params)
            skips.append(h)
        b = self.bot(self.pool(h))
        if params is not None and self.P > 0:
            b = self._apply_film(b, self.film_bot, params)
        return b, skips

    def decode(self, b, skips, params=None):
        h = b
        for i, (up, dec, drop) in enumerate(zip(self.up_convs, self.decoders, self.dropouts)):
            skip = skips[self.depth - 1 - i]
            h = up(h)
            # Match spatial dims when pooling lost an odd pixel
            dh = skip.shape[2] - h.shape[2]
            dw = skip.shape[3] - h.shape[3]
            if dh != 0 or dw != 0:
                # Pad if upsampled is smaller, crop if larger
                if dh > 0 or dw > 0:
                    h = F.pad(h, [0, max(dw, 0), 0, max(dh, 0)])
                if dh < 0 or dw < 0:
                    h = h[:, :, :skip.shape[2], :skip.shape[3]]
            h = drop(dec(torch.cat([h, skip], dim=1)))
            if params is not None and self.P > 0:
                h = self._apply_film(h, self.film_dec[i], params)
        return self.out(h)

    def forward(self, x, params=None, return_bottleneck: bool = False):
        b, skips = self.encode(x, params=params)
        y = self.decode(b, skips, params=params)
        return (y, b) if return_bottleneck else y

    # Convenience for surrogate test: decode with skips recomputed from x
    def decode_from_bottleneck(self, x, b_pred, params=None):
        _, skips = self.encode(x, params=params)
        return self.decode(b_pred, skips, params=params)
