# Authors: Abdourahmane (Abdou) Diaw - diawa@ornl.gov
# SPDX-License-Identifier: MIT
"""
Shared GNN building blocks for both conditional and EIRENE models.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class EdgeConv(MessagePassing):
    """Message passing layer that incorporates edge attributes (dR, dZ, distance)."""

    def __init__(self, in_ch, out_ch, edge_dim=3):
        super().__init__(aggr="mean")
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * in_ch + edge_dim, out_ch),
            nn.SiLU(),
            nn.Linear(out_ch, out_ch),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_ch + out_ch, out_ch),
            nn.SiLU(),
        )
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.update_mlp(torch.cat([x, out], dim=-1))
        return self.norm(out)

    def message(self, x_i, x_j, edge_attr):
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))


class NodeFiLM(nn.Module):
    """FiLM conditioning: broadcast global params to per-node scale+shift."""

    def __init__(self, param_dim, hidden_dim, film_hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(param_dim, film_hidden),
            nn.SiLU(),
            nn.Linear(film_hidden, hidden_dim * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.hidden_dim = hidden_dim

    def forward(self, h, params):
        """h: (N, hidden), params: (N, P) broadcast to all nodes."""
        out = self.net(params)
        gamma, beta = out.split(self.hidden_dim, dim=-1)
        return h * (1.0 + gamma) + beta
