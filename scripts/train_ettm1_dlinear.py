import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.dlinear import DLinear
from src.data.ettm1 import ETTm1WindowDataset


def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 재현성 강화(속도 약간 손해)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, device):
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    n_batches = 0

    mse_fn = torch.nn.MSELoss()
    mae_fn = torch.nn.L1Loss()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)  # (B, L, C)
            y = y.to(device)  # (B, P, C)
            pred = model(x)

            mse_sum += mse_fn(pred, y).item()
            mae_sum += mae_fn(pred, y).item()
            n_batches += 1

    return mse_sum / n_batches, mae_sum / n_batches


def main():
    set_seed(2025)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 논문 대표 조합
    seq_len = 336
    pred_len = 96

    csv_path = "data/ETTm1.csv"

    train_ds = ETTm1WindowDataset(csv_path, seq_len, pred_len, split="train")
    val_ds   = ETTm1WindowDataset(csv_path, seq_len, pred_len, split="val")
    test_ds  = ETTm1WindowDataset(csv_path, seq_len, pred_len, split="test")

    # 채널 수
    x0, _ = train_ds[0]
    channels = x0.shape[1]

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False, drop_last=True)

    model = DLinear(
        seq_len=seq_len, 
        pred_len=pred_len, 
        channels=channels, 
        kernel_size=25, 
        individual=True
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=5e-4)

    mse_fn = torch.nn.MSELoss()

    # 30 epoch
    for epoch in range(30):
        model.train()
        total = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = mse_fn(pred, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total += loss.item()

        train_mse = total / len(train_loader)
        val_mse, val_mae = evaluate(model, val_loader, device)

        print(
            f"epoch={epoch} "
            f"train_mse={train_mse:.6f} "
            f"val_mse={val_mse:.6f} val_mae={val_mae:.6f} "
            f"device={device}"
        )

    # 마지막에 test 평가 (표 비교용)
    test_mse, test_mae = evaluate(model, test_loader, device)
    print(f"[TEST] mse={test_mse:.6f} mae={test_mae:.6f} device={device}")


if __name__ == "__main__":
    main()
