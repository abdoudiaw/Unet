# Authors: Abdourahmane (Abdou) Diaw - diawa@ornl.gov
# SPDX-License-Identifier: MIT
"""
Conditional GNN for params → plasma field prediction.

Same task as the UNet surrogate (params → 2D fields) but on the
native SOLPS mesh represented as a graph.

Node features: [psi_n, |B|]  (geometry/magnetic topology)
Global conditioning: 5 control parameters injected via FiLM
Output: per-node predictions [Te, Ti, ne, ua, Sp, Qe, Qi, Sm]
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class EdgeConv(MessagePassing):
    """Message passing with edge attributes (dR, dZ, distance)."""

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
        """h: (N, hidden), params: (N, P) — already broadcast to nodes."""
        out = self.net(params)
        gamma, beta = out.split(self.hidden_dim, dim=-1)
        return h * (1.0 + gamma) + beta


class ConditionalGNN(nn.Module):
    """
    GNN surrogate: control params + mesh geometry → plasma fields.

    Args:
        node_features: Number of per-node geometry features (default 2: psi_n, |B|).
        param_dim: Number of global control parameters (default 5).
        out_features: Number of output fields per node.
        hidden: Hidden dimension.
        n_layers: Number of message-passing layers.
        edge_dim: Edge attribute dimension (3: dR, dZ, dist).
        dropout: Dropout rate on residual connections.
        film_hidden: FiLM generator hidden size.
    """

    def __init__(self, node_features=2, param_dim=5, out_features=8,
                 hidden=128, n_layers=6, edge_dim=3, dropout=0.1,
                 film_hidden=128):
        super().__init__()
        self.param_dim = param_dim

        # Encode node geometry
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        # Message-passing layers with FiLM conditioning
        self.layers = nn.ModuleList([
            EdgeConv(hidden, hidden, edge_dim) for _ in range(n_layers)
        ])
        self.films = nn.ModuleList([
            NodeFiLM(param_dim, hidden, film_hidden) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        # Decode to output fields
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, out_features),
        )

    def forward(self, data):
        """
        data.x:          (N, node_features) — psi_n, |B|
        data.edge_index: (2, E)
        data.edge_attr:  (E, 3) — dR, dZ, dist
        data.params:     (N, param_dim) — control params broadcast to all nodes
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        params = data.params  # (N, param_dim) — pre-broadcast

        h = self.node_encoder(x)

        for layer, film in zip(self.layers, self.films):
            h_new = layer(h, edge_index, edge_attr)
            h_new = film(h_new, params)
            h = h + self.dropout(h_new)  # residual

        return self.decoder(h)
