# Authors: Abdourahmane (Abdou) Diaw - diawa@ornl.gov
# SPDX-License-Identifier: MIT
"""
Conditional GNN surrogate: control params + mesh geometry → plasma fields.

Paper comparison model — same task as UNet (params → 2D fields)
but on the native SOLPS mesh.

Node features: [psi_n, |B|]
Global conditioning: control parameters via FiLM at each layer
Output: per-node [Te, Ti, ne, ua, Sp, Qe, Qi, Sm]
"""

import torch.nn as nn
from .layers import EdgeConv, NodeFiLM


class ConditionalGNN(nn.Module):
    """
    GNN surrogate: control params + mesh geometry → plasma fields.

    Args:
        node_features: Per-node geometry features (default 2: psi_n, |B|).
        param_dim: Global control parameters (default 5).
        out_features: Output fields per node.
        hidden: Hidden dimension.
        n_layers: Message-passing layers.
        edge_dim: Edge attribute dimension (3: dR, dZ, dist).
        dropout: Dropout rate on residual connections.
        film_hidden: FiLM generator hidden size.
    """

    def __init__(self, node_features=2, param_dim=5, out_features=8,
                 hidden=128, n_layers=6, edge_dim=3, dropout=0.1,
                 film_hidden=128):
        super().__init__()
        self.param_dim = param_dim

        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        self.layers = nn.ModuleList([
            EdgeConv(hidden, hidden, edge_dim) for _ in range(n_layers)
        ])
        self.films = nn.ModuleList([
            NodeFiLM(param_dim, hidden, film_hidden) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

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
        data.params:     (N, param_dim) — control params broadcast to nodes
        """
        h = self.node_encoder(data.x)

        for layer, film in zip(self.layers, self.films):
            h_new = layer(h, data.edge_index, data.edge_attr)
            h_new = film(h_new, data.params)
            h = h + self.dropout(h_new)

        return self.decoder(h)
