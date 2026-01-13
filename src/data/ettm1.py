import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class ETTm1WindowDataset(Dataset):
    """
    ETTm1 sliding-window dataset.
    Returns:
      x: (seq_len, C)
      y: (pred_len, C)
    """

    def __init__(
        self,
        csv_path: str,
        seq_len: int,
        pred_len: int,
        split: str = "train",          # "train" | "val" | "test"
        scale: bool = True,
    ):
        assert split in ("train", "val", "test")
        self.seq_len = seq_len
        self.pred_len = pred_len

        df = pd.read_csv(csv_path)
        data = df.iloc[:, 1:].to_numpy(dtype=np.float32)  # (T, C)

        # ETTm1 fixed split borders (15-min -> 4 / hour)
        border1s = {
            "train": 0,
            "val": 12*30*24*4 - seq_len,
            "test": (12+4)*30*24*4 - seq_len,
        }

        border2s = {
            "train": 12*30*24*4,
            "val": (12+4)*30*24*4,
            "test": (12+4+4)*30*24*4,
        }

        b1 = border1s[split]
        b2 = border2s[split]

        if scale:
            self.scaler = StandardScaler()
            train_end = border2s["train"]
            self.scaler.fit(data[:train_end])
            data = self.scaler.transform(data).astype(np.float32)
        else:
            self.scaler = None

        self.data = data[b1:b2]

        # 가능한 시작 인덱스 수
        self.max_i = len(self.data) - (seq_len + pred_len) + 1
        if self.max_i <= 0:
            raise ValueError("Split is too small for the given seq_len/pred_len.")

    def __len__(self):
        return self.max_i

    def __getitem__(self, i: int):
        x = self.data[i : i + self.seq_len]                           # (L, C)
        y = self.data[i + self.seq_len : i + self.seq_len + self.pred_len]  # (P, C)
        return torch.from_numpy(x), torch.from_numpy(y)