#!/usr/bin/env python3
"""
Lite-STGNN Model Components
Modularized architecture components for cleaner code organization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dlinear_module import DLinear


class LowRankTopKAdjacency(nn.Module):
    """Low-rank factorization + TopK sparsification producing row-normalized adjacency.
    A = softplus(E1 @ E2^T); keep top-k per row; add self-loop; row-normalize.
    """
    def __init__(self, n_nodes: int, rank: int = 16, tau: float = 1.0, k: int = 8, self_loop_alpha: float = 0.2):
        super().__init__()
        self.n = n_nodes
        self.rank = rank
        self.tau = tau
        self.k = k
        self.self_loop_alpha = self_loop_alpha
        self.E1 = nn.Parameter(torch.randn(n_nodes, rank) * 0.02)
        self.E2 = nn.Parameter(torch.randn(n_nodes, rank) * 0.02)

    def forward(self) -> torch.Tensor:
        sim = (self.E1 @ self.E2.t()) / max(self.tau, 1e-6)
        A_soft = torch.softmax(sim, dim=-1)
        n = A_soft.size(0)
        device = A_soft.device
        A_soft = A_soft - torch.diag_embed(torch.diag(A_soft))
        
        if self.k < n:
            topk_vals, topk_idx = torch.topk(A_soft, k=self.k, dim=1)
            mask = torch.zeros_like(A_soft)
            mask.scatter_(1, topk_idx, 1.0)
            sparse = A_soft * mask
        else:
            sparse = A_soft
        
        I = torch.eye(n, device=device)
        A = sparse + self.self_loop_alpha * I
        row_sum = A.sum(dim=1, keepdim=True) + 1e-6
        A = A / row_sum
        return A


class ResidualGating(nn.Module):
    """Horizon-wise adaptive sigmoid gating: 'flat' or 'band' (4 equal bands).
    Returns gates in [0,1] with shape [1, L, 1] for horizon-wise gating (shared across features).
    """
    def __init__(self, pred_len: int, n_features: int, mode: str = 'band'):
        super().__init__()
        self.pred_len = pred_len
        self.n_features = n_features
        self.mode = mode
        
        if mode == 'flat':
            self.g_params = nn.Parameter(torch.full((1,), -4.0))
        else:
            self.g_params = nn.Parameter(torch.full((4,), -4.0))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        B, L, N = r.shape
        if self.mode == 'band':
            b1, b2, b3 = 96, 192, 336
            idx = torch.arange(L, device=r.device)
            band_idx = torch.zeros(L, dtype=torch.long, device=r.device)
            band_idx = band_idx + (idx >= b1).long()
            band_idx = band_idx + (idx >= b2).long()
            band_idx = band_idx + (idx >= b3).long()
            band_idx = torch.clamp(band_idx, 0, 3)
            gates = torch.sigmoid(self.g_params)
            gate_h = gates[band_idx]
            gate_h = gate_h.view(1, L, 1)
        else:
            gate_h = torch.sigmoid(self.g_params).view(1, 1, 1)
        return gate_h * r


class LiteSTGNN(nn.Module):
    """Lite-STGNN: DLinear backbone + LowRankTopK spatial residual with horizon-wise gating.
    Input:  x [B, L_in, N]
    Output: y [B, L_out, N]
    """
    def __init__(self, n_nodes: int, seq_len: int, pred_len: int,
                 adj_rank: int = 16, adj_topk: int = 10, adj_tau: float = 1.2,
                 self_loop_alpha: float = 0.2, gate_mode: str = 'band',
                 prop_orders: int = 1,
                 use_temporal_head: bool = False, temporal_head_ratio: float = 0.5, temporal_head_dropout: float = 0.1,
                 use_input_residual: bool = False, input_residual_dropout: float = 0.0,
                 use_feature_norm: bool = False,
                 residual_dropout: float = 0.0):
        super().__init__()
        self.n_nodes = n_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.prop_orders = max(1, int(prop_orders))
        self.use_temporal_head = bool(use_temporal_head)
        self.use_input_residual = bool(use_input_residual)
        self.use_feature_norm = bool(use_feature_norm)
        self.residual_dropout = nn.Dropout(float(residual_dropout)) if residual_dropout > 0 else nn.Identity()
        
        if self.use_feature_norm:
            self.pre_norm = nn.LayerNorm(n_nodes)
            self.post_norm = nn.LayerNorm(n_nodes)
        else:
            self.pre_norm = nn.Identity()
            self.post_norm = nn.Identity()
        
        self.backbone = DLinear(seq_len=seq_len, pred_len=pred_len, individual=False)
        self.adj = LowRankTopKAdjacency(n_nodes, rank=adj_rank, tau=adj_tau, k=adj_topk, self_loop_alpha=self_loop_alpha)
        self.gate = ResidualGating(pred_len, n_features=n_nodes, mode=gate_mode)
        
        if self.use_temporal_head:
            hidden = max(1, int(pred_len * temporal_head_ratio))
            self.temporal_head = nn.Sequential(
                nn.Linear(pred_len, hidden),
                nn.GELU(),
                nn.Dropout(p=float(temporal_head_dropout)),
                nn.Linear(hidden, pred_len),
            )
        else:
            self.temporal_head = nn.Identity()
        
        if self.use_input_residual:
            self.inp_proj = nn.Linear(seq_len, pred_len)
            self.inp_drop = nn.Dropout(p=float(input_residual_dropout)) if input_residual_dropout > 0 else nn.Identity()
        else:
            self.inp_proj = None
            self.inp_drop = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_lin = self.backbone(x)
        
        if self.use_temporal_head:
            y_lin = y_lin.transpose(1, 2)
            y_lin = self.temporal_head(y_lin)
            y_lin = y_lin.transpose(1, 2)
        
        y_lin = self.pre_norm(y_lin)
        A = self.adj()
        B, L, N = y_lin.shape
        y_bl = y_lin.reshape(B*L, N, 1)
        
        y_agg = torch.matmul(A, y_bl)
        delta = y_agg - y_bl
        
        if self.prop_orders > 1:
            A_pow = A
            for p in range(2, self.prop_orders + 1):
                A_pow = torch.matmul(A_pow, A)
                y_p = torch.matmul(A_pow, y_bl)
                delta = delta + (y_p - y_bl)
        
        residual = delta.view(B, L, N)
        residual = self.post_norm(residual)
        residual = self.residual_dropout(residual)
        gated_residual = self.gate(residual)
        y_hat = y_lin + gated_residual
        
        if self.inp_proj is not None:
            inp_res = self.inp_proj(x.transpose(1, 2)).transpose(1, 2)
            inp_res = self.inp_drop(inp_res)
            y_hat = y_hat + inp_res
        
        return y_hat
