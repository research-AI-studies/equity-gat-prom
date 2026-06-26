"""Model architectures: multi-task GAT, GCN ablation, and MLP baseline.

All three share the same multi-task head structure (7 regression heads + 1
binary head) and homoscedastic-uncertainty loss weighting so that comparisons
isolate the effect of the relational/attention inductive bias.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv

from . import config as C


class MultiTaskHeads(nn.Module):
    """7 regression heads + 1 binary head on a shared node embedding."""

    def __init__(self, in_dim: int, n_reg: int):
        super().__init__()
        self.reg = nn.ModuleList([nn.Linear(in_dim, 1) for _ in range(n_reg)])
        self.bin = nn.Linear(in_dim, 1)

    def forward(self, h):
        reg = torch.cat([head(h) for head in self.reg], dim=1)   # (n, n_reg)
        logit = self.bin(h).squeeze(-1)                          # (n,)
        return reg, logit


class GATTrunk(nn.Module):
    def __init__(self, in_dim, hidden, heads, n_layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        d = in_dim
        for li in range(n_layers):
            last = li == n_layers - 1
            out = hidden
            self.convs.append(GATConv(d, out, heads=heads,
                                      concat=not last, dropout=dropout))
            d = out * heads if not last else out
        self.out_dim = d

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GCNTrunk(nn.Module):
    def __init__(self, in_dim, hidden, n_layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        d = in_dim
        for li in range(n_layers):
            out = hidden
            self.convs.append(GCNConv(d, out))
            d = out
        self.out_dim = d

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class MLPTrunk(nn.Module):
    def __init__(self, in_dim, hidden, n_layers, dropout):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(max(1, n_layers)):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.net = nn.Sequential(*layers)
        self.out_dim = d

    def forward(self, x, *args, **kwargs):
        return self.net(x)


class MultiTaskNet(nn.Module):
    """Trunk (GAT/GCN/MLP) + multi-task heads + homoscedastic log-variances."""

    def __init__(self, kind, in_dim, hidden=64, heads=4, n_layers=2,
                 dropout=0.3, n_reg=len(C.REG_TARGETS), skip=True):
        super().__init__()
        self.kind = kind
        self.skip = skip
        if kind == "gat":
            self.trunk = GATTrunk(in_dim, hidden, heads, n_layers, dropout)
        elif kind == "gcn":
            self.trunk = GCNTrunk(in_dim, hidden, n_layers, dropout)
        elif kind == "mlp":
            self.trunk = MLPTrunk(in_dim, hidden, n_layers, dropout)
        else:
            raise ValueError(kind)
        head_in = self.trunk.out_dim + (in_dim if skip else 0)
        self.heads = MultiTaskHeads(head_in, n_reg)
        # log-variance per task (n_reg regression + 1 binary)
        self.log_vars = nn.Parameter(torch.zeros(n_reg + 1))

    def forward(self, x, edge_index, edge_weight=None):
        if self.kind == "gcn":
            h = self.trunk(x, edge_index, edge_weight)
        elif self.kind == "gat":
            h = self.trunk(x, edge_index)
        else:
            h = self.trunk(x)
        if self.skip:
            h = torch.cat([h, x], dim=1)
        return self.heads(h)

    def loss(self, reg_pred, logit, reg_true, reg_mask, bin_true, bin_mask):
        """Homoscedastic-uncertainty-weighted masked multi-task loss."""
        n_reg = reg_pred.shape[1]
        total = 0.0
        for t in range(n_reg):
            m = reg_mask[:, t]
            if m.any():
                mse = F.mse_loss(reg_pred[m, t], reg_true[m, t])
                prec = torch.exp(-self.log_vars[t])
                total = total + 0.5 * prec * mse + 0.5 * self.log_vars[t]
        if bin_mask.any():
            bce = F.binary_cross_entropy_with_logits(logit[bin_mask], bin_true[bin_mask])
            prec = torch.exp(-self.log_vars[-1])
            total = total + prec * bce + 0.5 * self.log_vars[-1]
        return total
