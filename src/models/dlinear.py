"""
DLinear model implementation
Based on: Are Transformers Effective for Time Series Forecasting?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MovingAverage(nn.Module):
    """
    Moving Average with edge padding to keep the same length.
    Input/Output: (batch, seq_len, channels)
    """
    def __init__(self, kernel_size: int):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        self.kernel_size = kernel_size
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        b, l, c = x.shape

        # pad on both ends by repeating edge values to keep length
        pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        x_padded = torch.cat([front, x, end], dim=1) # (B, L+2*pad, C)

        x_padded = x_padded.permute(0, 2, 1)
        y = self.avg_pool(x_padded)
        y = y.permute(0, 2, 1)
        return y

class SeriesDecomposition(nn.Module):
    """
    Decompose a series into seasonal and trend components.
    """
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend

class DLinear(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, channels: int, kernel_size: int = 25, individual: bool = True):
        super().__init__()
        
        self.individual = individual
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels

        self.decomp = SeriesDecomposition(kernel_size=kernel_size)

        if self.individual:
            self.linear_seasonal = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(channels)])
            self.linear_trend = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(channels)])
        else:
            self.linear_seasonal = nn.Linear(seq_len, pred_len)
            self.linear_trend = nn.Linear(seq_len, pred_len)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, seq_len, Channel]
        # return: [Batch, pred_len, Channel]

        if x.ndim != 3:
            raise ValueError(f"Expected x.ndim==3 (B, L, C), got {x.ndim} with shape {tuple(x.shape)}")
        b, l, c = x.shape
        if l != self.seq_len or c != self.channels:
            raise ValueError(f"Expected x shape (B, {self.seq_len}, {self.channels}), got {tuple(x.shape)}")
        
        seasonal, trend = self.decomp(x)
        # seasonal, trend: (B, L, C)
        seasonal = seasonal.permute(0, 2, 1)  # (B, C, L)
        trend = trend.permute(0, 2, 1)        # (B, C, L)

        if self.individual:
            seasonal_out = []
            trend_out = []
            for ch in range(self.channels):
                seasonal_out.append(self.linear_seasonal[ch](seasonal[:, ch, :]))  # (B, P)
                trend_out.append(self.linear_trend[ch](trend[:, ch, :]))           # (B, P)

            seasonal_out = torch.stack(seasonal_out, dim=1)  # (B, C, P)
            trend_out = torch.stack(trend_out, dim=1)        # (B, C, P)
            y = seasonal_out + trend_out                     # (B, C, P)
            y = y.permute(0, 2, 1)                           # (B, P, C)
        else:
            b, l, c = x.shape
            seasonal = seasonal.reshape(b * c, l)            # (B*C, L)
            trend = trend.reshape(b * c, l)                  # (B*C, L)

            seasonal_out = self.linear_seasonal(seasonal)    # (B*C, P)
            trend_out = self.linear_trend(trend)             # (B*C, P)

            y = seasonal_out + trend_out                     # (B*C, P)
            y = y.reshape(b, c, self.pred_len).permute(0, 2, 1)  # (B, P, C)
        return y
