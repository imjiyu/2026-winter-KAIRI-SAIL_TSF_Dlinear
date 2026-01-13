import torch
from src.models.dlinear import DLinear

model = DLinear(seq_len=96, pred_len=24, channels=7, kernel_size=25)
x = torch.randn(2, 96, 7)
y = model(x)

print(y.shape)