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
    def __init__(self, P: int, z_dim: int, hidden=(256,256), dropout=0.0):
        super().__init__()
        layers = []
        in_d = P
        for h in hidden:
            layers += [nn.Linear(in_d, h), nn.SiLU()]
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

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    """
    UNet with ConvTranspose2d upsampling.
    Refactored into encode() and decode().
    """
    def __init__(self, in_ch=1, out_ch=1, base=32, groups_gn=8):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = DoubleConv(in_ch, base, groups_gn)
        self.enc2 = DoubleConv(base, base*2, groups_gn)
        self.enc3 = DoubleConv(base*2, base*4, groups_gn)

        self.bot  = DoubleConv(base*4, base*8, groups_gn)

        self.up3  = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.dec3 = DoubleConv(base*4 + base*4, base*4, groups_gn)

        self.up2  = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.dec2 = DoubleConv(base*2 + base*2, base*2, groups_gn)

        self.up1  = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.dec1 = DoubleConv(base + base, base, groups_gn)

        self.out  = nn.Conv2d(base, out_ch, 1)

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bot(self.pool(e3))
        return b, (e1, e2, e3)

    def decode(self, b, skips):
        e1, e2, e3 = skips
        u3 = self.up3(b)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.out(d1)

    def forward(self, x, return_bottleneck: bool = False):
        b, skips = self.encode(x)
        y = self.decode(b, skips)
        return (y, b) if return_bottleneck else y

    # Convenience for surrogate test: decode with skips recomputed from x
    def decode_from_bottleneck(self, x, b_pred):
        _, skips = self.encode(x)
        return self.decode(b_pred, skips)
