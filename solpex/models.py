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
    """
    def __init__(self, in_ch=1, out_ch=1, base=32, groups_gn=8, dropout=0.0,
                 P: int = 0, film_hidden: int = 128, z_dim: int = 0):
        super().__init__()
        self.dropout = dropout
        self.P = P
        self.z_dim = z_dim
        self.pool = nn.MaxPool2d(2)

        self.enc1 = DoubleConv(in_ch, base, groups_gn)
        self.enc2 = DoubleConv(base, base*2, groups_gn)
        self.enc3 = DoubleConv(base*2, base*4, groups_gn)

        self.bot  = DoubleConv(base*4, base*8, groups_gn)

        self.up3  = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.dec3 = DoubleConv(base*4 + base*4, base*4, groups_gn)
        self.drop3 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.up2  = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.dec2 = DoubleConv(base*2 + base*2, base*2, groups_gn)
        self.drop2 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.up1  = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.dec1 = DoubleConv(base + base, base, groups_gn)
        self.drop1 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.out  = nn.Conv2d(base, out_ch, 1)

        # FiLM conditioning layers (only when P > 0)
        if P > 0:
            self.film_enc1 = FiLMGenerator(P, base, film_hidden)
            self.film_enc2 = FiLMGenerator(P, base*2, film_hidden)
            self.film_enc3 = FiLMGenerator(P, base*4, film_hidden)
            self.film_bot  = FiLMGenerator(P, base*8, film_hidden)
            self.film_dec3 = FiLMGenerator(P, base*4, film_hidden)
            self.film_dec2 = FiLMGenerator(P, base*2, film_hidden)
            self.film_dec1 = FiLMGenerator(P, base, film_hidden)

        # Learned latent projection (only when z_dim > 0)
        if z_dim > 0:
            self.z_proj = nn.Linear(base * 8, z_dim)
            self.z_unproj = nn.Linear(z_dim, base * 8)
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
        z = bottleneck_to_z(b)  # (B, base*8)
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
        e1 = self.enc1(x)
        if params is not None and self.P > 0:
            e1 = self._apply_film(e1, self.film_enc1, params)
        e2 = self.enc2(self.pool(e1))
        if params is not None and self.P > 0:
            e2 = self._apply_film(e2, self.film_enc2, params)
        e3 = self.enc3(self.pool(e2))
        if params is not None and self.P > 0:
            e3 = self._apply_film(e3, self.film_enc3, params)
        b  = self.bot(self.pool(e3))
        if params is not None and self.P > 0:
            b = self._apply_film(b, self.film_bot, params)
        return b, (e1, e2, e3)

    def decode(self, b, skips, params=None):
        e1, e2, e3 = skips
        u3 = self.up3(b)
        d3 = self.drop3(self.dec3(torch.cat([u3, e3], dim=1)))
        if params is not None and self.P > 0:
            d3 = self._apply_film(d3, self.film_dec3, params)
        u2 = self.up2(d3)
        d2 = self.drop2(self.dec2(torch.cat([u2, e2], dim=1)))
        if params is not None and self.P > 0:
            d2 = self._apply_film(d2, self.film_dec2, params)
        u1 = self.up1(d2)
        d1 = self.drop1(self.dec1(torch.cat([u1, e1], dim=1)))
        if params is not None and self.P > 0:
            d1 = self._apply_film(d1, self.film_dec1, params)
        return self.out(d1)

    def forward(self, x, params=None, return_bottleneck: bool = False):
        b, skips = self.encode(x, params=params)
        y = self.decode(b, skips, params=params)
        return (y, b) if return_bottleneck else y

    # Convenience for surrogate test: decode with skips recomputed from x
    def decode_from_bottleneck(self, x, b_pred, params=None):
        _, skips = self.encode(x, params=params)
        return self.decode(b_pred, skips, params=params)
