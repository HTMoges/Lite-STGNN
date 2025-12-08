import torch
import torch.nn as nn

class DLinear(nn.Module):
    """
    Minimal DLinear implementation with moving average decomposition.
    [See https://github.com/vivva/DLinear for reference]
    Inputs:  x [B, L_in, N]
    Outputs: y [B, L_out, N]
    """
    def __init__(self, seq_len: int, pred_len: int, individual: bool = True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        # Note: Always use separate Trend and Seasonal heads (even for individual=False)
        self.Linear_Trend = nn.Linear(seq_len, pred_len)
        self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
        

    @staticmethod
    def moving_average(x: torch.Tensor, kernel_size: int = 25) -> torch.Tensor:
        # x: [B, L, N]
        if kernel_size <= 1:
            return x
        pad = (kernel_size - 1) // 2
        weight = torch.ones(1, 1, kernel_size, device=x.device) / kernel_size
        # reshape to [B*N, 1, L]
        b, l, n = x.shape
        x_ = x.permute(0, 2, 1).reshape(b * n, 1, l)
        trend = torch.conv1d(nn.functional.pad(x_, (pad, pad), mode='replicate'), weight)
        trend = trend.squeeze(1).reshape(b, n, l).permute(0, 2, 1)
        return trend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L_in, N]
        # Moving average decomposition (critical for DLinear performance)
        trend = self.moving_average(x)
        seasonal = x - trend
        # Apply separate linear projections to trend and seasonal components
        out_trend = self.Linear_Trend(trend.transpose(1, 2)).transpose(1, 2)
        out_seasonal = self.Linear_Seasonal(seasonal.transpose(1, 2)).transpose(1, 2)
        y = out_trend + out_seasonal
        return y
