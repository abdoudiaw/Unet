# Authors: Abdourahmane (Abdou) Diaw - diawa@ornl.gov
# SPDX-License-Identifier: MIT
"""
EIRENE-replacement GNN: plasma state → EIRENE source terms + neutrals.

Designed to replace EIRENE inside the B2.5-EIRENE coupling loop.

Node features: [Te, Ti, ne, ni, ua, vol, hx, hy, bb0-bb3, R, Z] (14)
Output: [Sp, Sne, Qe, Qi, Sm, dab2, dmb2, tab2, tmb2] (9)

No global parameter conditioning — the plasma state at each node
is the input, and the model predicts what EIRENE would return.
"""

import torch.nn as nn
from .layers import EdgeConv


class EireneGNN(nn.Module):
    """GNN that maps plasma node features to EIRENE source terms.

    Args:
        in_features: Number of input node features (14 for default config).
        out_features: Number of target fields (9).
        hidden: Hidden dimension.
        n_layers: Number of message passing layers.
        edge_dim: Dimension of edge attributes (3: dR, dZ, dist).
        dropout: Dropout rate.
    """

    def __init__(self, in_features=14, out_features=9, hidden=128,
                 n_layers=6, edge_dim=3, dropout=0.1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        self.layers = nn.ModuleList([
            EdgeConv(hidden, hidden, edge_dim) for _ in range(n_layers)
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
        data.x:          (N, in_features) — plasma state per node
        data.edge_index: (2, E)
        data.edge_attr:  (E, 3) — dR, dZ, dist
        """
        h = self.encoder(data.x)

        for layer in self.layers:
            h_new = layer(h, data.edge_index, data.edge_attr)
            h = h + self.dropout(h_new)

        return self.decoder(h)
