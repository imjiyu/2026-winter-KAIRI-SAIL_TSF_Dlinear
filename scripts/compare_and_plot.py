import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data.ettm1 import ETTm1WindowDataset
from src.models.dlinear import DLinear
from src.models.linear import Linear
from src.models.nlinear import NLinear


def set_seed(seed: int = 2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def inverse_transform_2d(scaler, arr_2d: np.ndarray) -> np.ndarray:
    """StandardScaler inverse_transform expects (N, C)."""
    return scaler.inverse_transform(arr_2d)


def main():
    print("COMPARE SCRIPT START (ckpt-only, GT unified)")

    # ===== 실험 설정 =====
    seq_len = 336
    pred_len = 96
    csv_path = "data/ETTm1.csv"

    # 시각화에서 보여줄 과거 길이(예측 직전, 3000 너무 긺)
    tail_len = 300

    # train_ettm1.py에서 학습 시 사용한 설정과 동일해야 함
    seed = 2026
    individual = True
    kernel_size = 25

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== 데이터셋 로드 =====
    # scale=True로 scaler 확보(훈련 구간으로 fit)
    train_ds = ETTm1WindowDataset(csv_path, seq_len, pred_len, split="train", scale=True)
    test_ds  = ETTm1WindowDataset(csv_path, seq_len, pred_len, split="test", scale=True)

    x0, _ = train_ds[0]
    channels = x0.shape[1]
    ot_idx = channels - 1  # 보통 마지막 컬럼이 OT

    # ===== 모델 생성 =====
    models = {
        "Linear":  Linear(seq_len, pred_len, channels, individual=individual).to(device),
        "NLinear": NLinear(seq_len, pred_len, channels, individual=individual).to(device),
        "DLinear": DLinear(seq_len, pred_len, channels, kernel_size=kernel_size, individual=individual).to(device),
    }

    # ===== 체크포인트 로드 =====
    ckpt_dir = "results/checkpoints"
    ckpt_map = {
        "Linear":  os.path.join(ckpt_dir, f"linear_L{seq_len}_P{pred_len}_seed{seed}_ind{int(individual)}.pt"),
        "NLinear": os.path.join(ckpt_dir, f"nlinear_L{seq_len}_P{pred_len}_seed{seed}_ind{int(individual)}.pt"),
        "DLinear": os.path.join(ckpt_dir, f"dlinear_L{seq_len}_P{pred_len}_seed{seed}_ind{int(individual)}.pt"),
    }

    for name, model in models.items():
        path = ckpt_map[name]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Checkpoint not found: {path}\n"
                f"먼저 아래처럼 학습해서 체크포인트를 만들어야 합니다:\n"
                f"python -m scripts.train_ettm1 --model {name.lower()} --individual --epochs 50 --lr 5e-4 --seed {seed}"
            )
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        print(f"Loaded ckpt: {name} <- {path}")

    # ------------------------------------------------------------
    # 1) test OT 실제 시계열(원스케일) 준비
    # ------------------------------------------------------------
    test_data_scaled = test_ds.data  # (T_test, C)
    scaler = getattr(train_ds, "scaler", None)

    if scaler is not None:
        test_data = inverse_transform_2d(scaler, test_data_scaled)  # (T_test, C) original scale
    else:
        test_data = test_data_scaled

    gt_ot_full = test_data[:, ot_idx]  # (T_test,)

    # 예측 시작점(진짜 미래 시작) = 마지막 pred_len 구간의 시작
    T = len(gt_ot_full)
    split = T - pred_len  # 예측 시작 인덱스 (future의 첫 index)

    # 시각화 past 길이는 split을 넘을 수 없음
    tail_len = min(tail_len, split)

    # 과거 구간(예측 직전까지) + 미래 구간(정답) 합쳐서 "연속 GT"로 그림
    gt_past   = gt_ot_full[split - tail_len : split]              # (tail_len,)
    gt_future = gt_ot_full[split : split + pred_len]              # (pred_len,)
    gt_concat = np.concatenate([gt_past, gt_future], axis=0)      # (tail_len + pred_len,)

    # ------------------------------------------------------------
    # 2) 예측 만들기 (test 마지막 샘플의 입력 윈도우로 예측)
    # ------------------------------------------------------------
    x_last, _ = test_ds[len(test_ds) - 1]  # x:(L,C) scaled
    x_b = x_last.unsqueeze(0).to(device)   # (1,L,C)

    preds_ot = {}
    with torch.no_grad():
        for name, model in models.items():
            pred_scaled = model(x_b).squeeze(0).cpu().numpy()  # (P,C)
            if scaler is not None:
                pred = inverse_transform_2d(scaler, pred_scaled)  # (P,C) original scale
            else:
                pred = pred_scaled
            preds_ot[name] = pred[:, ot_idx]  # (P,)

    # ------------------------------------------------------------
    # 3) Plot
    #   - GT는 같은색으로 쭉 연속 표시
    #   - 예측은 미래 구간(x_pred)에만 덧그리기
    # ------------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join(
        "results",
        f"ettm1_ot_gt_unified_L{seq_len}_P{pred_len}_tail{tail_len}_seed{seed}.png"
    )

    x_all  = np.arange(tail_len + pred_len)           # 0 ... tail_len+pred_len-1
    x_pred = np.arange(tail_len, tail_len + pred_len) # 예측 구간

    plt.figure(figsize=(12, 4))

    # GT를 쭉 같은 파란색으로 연속 표시
    plt.plot(x_all, gt_concat, label="GT(OT)", linewidth=2)

    # 예측 시작점 표시 (past 끝 다음 점선으로 뚜뚜)
    plt.axvline(tail_len - 1, linestyle="--", linewidth=1)

    # 모델 예측은 미래 구간에만 덧그림
    for name in ["Linear", "NLinear", "DLinear"]:
        plt.plot(x_pred, preds_ot[name], label=name)

    plt.title(f"ETTm1 OT GT + forecast | L={seq_len}, P={pred_len}, tail={tail_len}, seed={seed}")
    plt.xlabel("time index (relative)")
    plt.ylabel("OT (original scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
