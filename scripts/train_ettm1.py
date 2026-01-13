import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import csv

from src.data.ettm1 import ETTm1WindowDataset
from src.models.dlinear import DLinear
from src.models.linear import Linear
from src.models.nlinear import NLinear


def set_seed(seed: int = 2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, device):
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    n = 0

    mse_fn = torch.nn.MSELoss()
    mae_fn = torch.nn.L1Loss()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            mse_sum += mse_fn(pred, y).item()
            mae_sum += mae_fn(pred, y).item()
            n += 1

    return mse_sum / n, mae_sum / n


def build_model(name: str, seq_len: int, pred_len: int, channels: int, individual: bool, kernel_size: int):
    name = name.lower()
    if name == "linear":
        return Linear(seq_len, pred_len, channels, individual=individual)
    if name == "nlinear":
        return NLinear(seq_len, pred_len, channels, individual=individual)
    if name == "dlinear":
        return DLinear(seq_len, pred_len, channels, kernel_size=kernel_size, individual=individual)
    raise ValueError(f"Unknown model: {name}")


def append_metrics_csv(
    metrics_path: str,
    row: list,
    header: list,
):
    write_header = not os.path.exists(metrics_path)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="dlinear", choices=["linear", "nlinear", "dlinear"])
    parser.add_argument("--seq_len", type=int, default=336)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--individual", action="store_true", help="Use per-channel Linear layers")
    parser.add_argument("--kernel_size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    csv_path = "data/ETTm1.csv"

    # 동일 split / 동일 scaler(train 기준 fit) 유지
    train_ds = ETTm1WindowDataset(csv_path, args.seq_len, args.pred_len, split="train", scale=True)
    val_ds   = ETTm1WindowDataset(csv_path, args.seq_len, args.pred_len, split="val", scale=True)
    test_ds  = ETTm1WindowDataset(csv_path, args.seq_len, args.pred_len, split="test", scale=True)

    x0, _ = train_ds[0]
    channels = x0.shape[1]

    # shuffle 순서까지 seed로 고정하기
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, generator=g)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    model = build_model(args.model, args.seq_len, args.pred_len, channels, args.individual, args.kernel_size).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(args.epochs):
        model.train()
        total = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total += loss.item()

        train_mse = total / len(train_loader)
        val_mse, val_mae = evaluate(model, val_loader, device)

        # best-val 체크포인트(공정한 비교 핵심)
        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"model={args.model} epoch={epoch} "
            f"train_mse={train_mse:.6f} "
            f"val_mse={val_mse:.6f} val_mae={val_mae:.6f} "
            f"device={device}"
        )

    # 1) best-val 상태로 복원한 뒤 test 평가
    if best_state is None:
        raise RuntimeError("best_state is None. (This should not happen unless val_loader is empty.)")
    model.load_state_dict(best_state)

    # 2) 체크포인트 저장
    os.makedirs("results/checkpoints", exist_ok=True)
    ckpt_path = (
        f"results/checkpoints/{args.model}_L{args.seq_len}_P{args.pred_len}"
        f"_seed{args.seed}_ind{int(args.individual)}.pt"
    )
    torch.save(model.state_dict(), ckpt_path)

    # 3) test 평가(=best-val 기반)
    test_mse, test_mae = evaluate(model, test_loader, device)
    print(
        f"[TEST-BEST] model={args.model} "
        f"best_epoch={best_epoch} best_val_mse={best_val:.6f} "
        f"test_mse={test_mse:.6f} test_mae={test_mae:.6f} "
        f"device={device}"
    )
    print(f"Saved checkpoint: {ckpt_path}")

    # 4) metrics.csv append 저장 (모델 3개 돌리면 총 3줄 쌓임 -> 이따 확인하기)
    metrics_path = "results/metrics.csv"
    header = [
        "model", "seed", "seq_len", "pred_len", "lr", "epochs", "individual", "kernel_size",
        "best_epoch", "best_val_mse", "test_mse", "test_mae", "ckpt_path"
    ]
    row = [
        args.model, args.seed, args.seq_len, args.pred_len, args.lr, args.epochs, int(args.individual), args.kernel_size,
        best_epoch, best_val, test_mse, test_mae, ckpt_path
    ]
    append_metrics_csv(metrics_path, row, header)
    print(f"Appended metrics: {metrics_path}")


if __name__ == "__main__":
    main()
