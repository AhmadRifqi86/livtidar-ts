"""
Time Series Forecasting Dataloader for LIV-TiDAR experiments.

Supports: ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Traffic, Exchange, Weather, ILI
Standard split: 0.7 / 0.1 / 0.2 (train / val / test)
ETT uses: 12/4/4 months split (standard in literature)

Usage:
    from dataload import get_dataloader, get_dataset_info

    train_dl, val_dl, test_dl = get_dataloader(
        dataset="ETTh1", seq_len=96, pred_len=96, batch_size=32
    )

    for batch_x, batch_y in train_dl:
        # batch_x: (B, seq_len, C)  - lookback window
        # batch_y: (B, pred_len, C) - forecast horizon
        ...
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# ── Dataset registry ──────────────────────────────────────────────────────────

DATASET_CONFIG = {
    "ETTh1": {
        "path": "ETT-small/ETTh1.csv",
        "format": "csv_header",     # CSV with header row + date column
        "freq": "h",
        "split": "ett",             # ETT uses fixed month-based splits
    },
    "ETTh2": {
        "path": "ETT-small/ETTh2.csv",
        "format": "csv_header",
        "freq": "h",
        "split": "ett",
    },
    "ETTm1": {
        "path": "ETT-small/ETTm1.csv",
        "format": "csv_header",
        "freq": "15min",
        "split": "ett",
    },
    "ETTm2": {
        "path": "ETT-small/ETTm2.csv",
        "format": "csv_header",
        "freq": "15min",
        "split": "ett",
    },
    "Electricity": {
        "path": "electricity/electricity.txt",
        "format": "txt_numeric",     # plain numeric CSV, no header
        "freq": "h",
        "split": "standard",         # 0.7 / 0.1 / 0.2
    },
    "Traffic": {
        "path": "traffic/traffic.txt",
        "format": "txt_numeric",
        "freq": "h",
        "split": "standard",
    },
    "Exchange": {
        "path": "exchange_rate/exchange_rate.txt",
        "format": "txt_numeric",
        "freq": "d",
        "split": "standard",
    },
    "Weather": {
        "path": "weather/weather.csv",
        "format": "csv_header",
        "freq": "10min",
        "split": "standard",
    },
    "ILI": {
        "path": "illness/national_illness.csv",
        "format": "csv_header",
        "freq": "w",
        "split": "standard",
    },
}


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_raw(dataset_name: str) -> np.ndarray:
    """Load raw data as numpy array (T, C). Drops date columns if present."""
    cfg = DATASET_CONFIG[dataset_name]
    filepath = os.path.join(DATA_ROOT, cfg["path"])

    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found or empty at {filepath}. "
            f"Run ./download_data.sh or download manually."
        )

    if cfg["format"] == "csv_header":
        # CSV with header; first column is typically 'date'
        import csv
        with open(filepath, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        # skip date column (first col) if it contains non-numeric strings
        try:
            float(rows[0][0])
            start_col = 0
        except ValueError:
            start_col = 1
        data = np.array(
            [[float(v) for v in row[start_col:]] for row in rows],
            dtype=np.float32,
        )
    elif cfg["format"] == "txt_numeric":
        # plain comma-separated numeric, no header
        data = np.loadtxt(filepath, delimiter=",", dtype=np.float32)
    else:
        raise ValueError(f"Unknown format: {cfg['format']}")

    return data  # (T, C)


def _split_indices(dataset_name: str, total_len: int):
    """Return (train_end, val_end) indices for splitting."""
    cfg = DATASET_CONFIG[dataset_name]

    if cfg["split"] == "ett":
        # ETT standard: 12 months train, 4 months val, 4 months test
        # ETTh: 24*30*12=8640 train, 24*30*4=2880 val, 2880 test
        # ETTm: 24*4*30*12=34560 train, 24*4*30*4=11520 val, 11520 test
        if "h" in dataset_name.lower() and "m" not in dataset_name.lower():
            train_end = 12 * 30 * 24  # 8640
            val_end = train_end + 4 * 30 * 24  # 11520
        else:
            train_end = 12 * 30 * 24 * 4  # 34560
            val_end = train_end + 4 * 30 * 24 * 4  # 46080
    else:
        # Standard 0.7 / 0.1 / 0.2
        train_end = int(total_len * 0.7)
        val_end = int(total_len * 0.8)

    return train_end, val_end


# ── Normalization ─────────────────────────────────────────────────────────────

class StandardScaler:
    """Per-channel z-score normalization fitted on training data."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        self.mean = data.mean(axis=0, keepdims=True)
        self.std = data.std(axis=0, keepdims=True)
        self.std[self.std < 1e-8] = 1.0  # avoid div-by-zero
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean


# ── Dataset ───────────────────────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    """Sliding window time series dataset.

    Args:
        data:     (T, C) normalized numpy array
        seq_len:  lookback window length
        pred_len: forecast horizon length
        stride:   step between consecutive windows (default=1)
    """

    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int, stride: int = 1):
        self.data = torch.from_numpy(data).float()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        self.total_len = seq_len + pred_len
        # number of valid windows
        self.n_samples = max(0, (len(self.data) - self.total_len) // stride + 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.total_len
        window = self.data[start:end]
        x = window[:self.seq_len]        # (seq_len, C)
        y = window[self.seq_len:]         # (pred_len, C)
        return x, y


# ── Public API ────────────────────────────────────────────────────────────────

def get_dataset_info(dataset_name: str) -> dict:
    """Return metadata about a dataset without loading it."""
    cfg = DATASET_CONFIG[dataset_name].copy()
    filepath = os.path.join(DATA_ROOT, cfg["path"])
    cfg["exists"] = os.path.exists(filepath) and os.path.getsize(filepath) > 0
    if cfg["exists"]:
        data = _load_raw(dataset_name)
        cfg["length"] = data.shape[0]
        cfg["n_variates"] = data.shape[1]
    return cfg


def get_dataloader(
    dataset: str = "ETTh1",
    seq_len: int = 96,
    pred_len: int = 96,
    batch_size: int = 32,
    num_workers: int = 4,
    stride: int = 1,
) -> tuple:
    """Build train/val/test DataLoaders for a given dataset.

    Args:
        dataset:     Dataset name (see DATASET_CONFIG keys)
        seq_len:     Lookback window length
        pred_len:    Forecast horizon length
        batch_size:  Batch size
        num_workers: DataLoader workers
        stride:      Sliding window stride (1 = fully overlapping)

    Returns:
        (train_loader, val_loader, test_loader, scaler)
        scaler can be used for inverse_transform on predictions
    """
    if dataset not in DATASET_CONFIG:
        available = ", ".join(sorted(DATASET_CONFIG.keys()))
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {available}")

    raw = _load_raw(dataset)  # (T, C)
    T, C = raw.shape
    train_end, val_end = _split_indices(dataset, T)

    train_data = raw[:train_end]
    val_data = raw[train_end:val_end]
    test_data = raw[val_end:]

    # fit scaler on training data only
    scaler = StandardScaler()
    scaler.fit(train_data)

    train_norm = scaler.transform(train_data)
    val_norm = scaler.transform(val_data)
    test_norm = scaler.transform(test_data)

    train_ds = TimeSeriesDataset(train_norm, seq_len, pred_len, stride)
    val_ds = TimeSeriesDataset(val_norm, seq_len, pred_len, stride)
    test_ds = TimeSeriesDataset(test_norm, seq_len, pred_len, stride)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )

    return train_loader, val_loader, test_loader, scaler


# ── CLI quick test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(" Time Series Dataloader — Dataset Summary")
    print("=" * 60)

    for name in sorted(DATASET_CONFIG.keys()):
        info = get_dataset_info(name)
        status = "OK" if info["exists"] else "MISSING"
        if info["exists"]:
            print(
                f"  [{status}] {name:15s}  "
                f"T={info['length']:>6d}  C={info['n_variates']:>4d}  "
                f"freq={info['freq']:>5s}  split={info['split']}"
            )
        else:
            print(f"  [{status}] {name:15s}  (run ./download_data.sh)")

    # quick smoke test on first available dataset
    print()
    print("-" * 60)
    for name in ["ETTh1", "ETTm1", "Electricity", "Exchange", "Traffic"]:
        try:
            info = get_dataset_info(name)
            if not info["exists"]:
                continue
            print(f"\n  Smoke test: {name} (seq=96, pred=96, batch=32)")
            train_dl, val_dl, test_dl, scaler = get_dataloader(
                dataset=name, seq_len=96, pred_len=96, batch_size=32, num_workers=0,
            )
            bx, by = next(iter(train_dl))
            print(f"    train: {len(train_dl):>5d} batches | x={tuple(bx.shape)} y={tuple(by.shape)}")
            print(f"    val:   {len(val_dl):>5d} batches")
            print(f"    test:  {len(test_dl):>5d} batches")
            print(f"    scaler mean shape: {scaler.mean.shape}, std shape: {scaler.std.shape}")
            break
        except Exception as e:
            print(f"  [{name}] Error: {e}")

    print()
    print("Done.")