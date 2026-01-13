import torch
import torch.nn as nn


class NLinear(nn.Module):
    """
    Normalized Linear (NLinear).
    Uses last value normalization:
      x_last = x[:, -1:, :]
      x_norm = x - x_last
      y = Linear(x_norm) + x_last
    """

    def __init__(self, seq_len: int, pred_len: int, channels: int, individual: bool = True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.individual = individual

        if self.individual:
            self.Linear = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(channels)])
        else:
            self.Linear = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must be 3D (B, L, C), got shape {tuple(x.shape)}")
        b, l, c = x.shape
        if l != self.seq_len or c != self.channels:
            raise ValueError(f"Expected x shape (B, {self.seq_len}, {self.channels}), got (B, {l}, {c})")

        x_last = x[:, -1:, :]           # (B, 1, C)
        x_norm = x - x_last             # (B, L, C)

        x_norm = x_norm.permute(0, 2, 1)  # (B, C, L)

        if self.individual:
            outs = []
            for ch in range(self.channels):
                outs.append(self.Linear[ch](x_norm[:, ch, :]))  # (B, P)
            y = torch.stack(outs, dim=1)  # (B, C, P)
        else:
            y = self.Linear(x_norm.reshape(b * c, l)).reshape(b, c, self.pred_len)

        y = y.permute(0, 2, 1)  # (B, P, C)
        return y + x_last       # broadcast (B, 1, C) -> (B, P, C)
