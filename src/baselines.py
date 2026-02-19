"""Baseline models for time series forecasting comparison.

Implements five baselines against which LIV-TS is compared:
  - DLinear       : Decomposition + per-variate Linear (Zeng et al. 2023)
  - PatchTST      : Patched temporal tokens + Transformer (Nie et al. 2023)
  - iTransformer  : LIV-based fixed genome (SA1 cross-variate + GMemless per-variate)
  - DMamba        : LIV-based fixed genome (Rec1 seasonal + GMemless trend)
  - S-Mamba       : LIV-based fixed genome (SA1 variate-corr + GMemless temporal)

The three LIV-based baselines use STARLIVTSModel with hand-crafted fixed
architectures (no search), reusing the same training infrastructure as the
GA candidates. This ensures a fair comparison.

Usage:
  # Single baseline, single horizon
  python -m src.baselines --model dlinear --dataset ETTh1 --pred_len 96

  # All baselines, all horizons on one dataset
  python -m src.baselines --model all --dataset ETTh1 --all_horizons

  # All baselines, all horizons, all datasets
  python -m src.baselines --model all --all_horizons --all_datasets
"""

import argparse
import json
import logging
import math
import time
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.liv import SA1, Rec1, GMemless, UnifiedLIVBlock
from core.modeldef import STARLIVTSModel, RevIN
from src.dataload import get_dataloader, get_dataset_info
from src.train import train_ts_model, evaluate_ts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HORIZONS = [96, 192, 336, 720]
ALL_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Electricity", "Traffic", "Exchange"]


# =============================================================================
# DLinear
# =============================================================================

class _MovingAvg(nn.Module):
    """Centred moving average decomposition (padding mirrors boundary)."""
    def __init__(self, kernel_size: int):
        super().__init__()
        self.k = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        """x: (B, L, C) → seasonal, trend each (B, L, C)"""
        pad = (self.k - 1) // 2
        x_t = x.transpose(1, 2)                                  # (B, C, L)
        front = x_t[:, :, :1].expand(-1, -1, pad)
        back  = x_t[:, :, -1:].expand(-1, -1, self.k - 1 - pad)
        x_pad = torch.cat([front, x_t, back], dim=2)             # (B, C, L+k-1)
        trend = self.avg(x_pad).transpose(1, 2)                  # (B, L, C)
        return x - trend, trend


class DLinear(nn.Module):
    """Decomposition + per-variate linear projection (Zeng et al. 2023).

    Each variate is treated independently: seasonal and trend components are
    each projected from seq_len → pred_len by a separate linear layer.
    No inter-variate mixing. RevIN normalization applied.
    """

    def __init__(self, seq_len: int, pred_len: int, n_variates: int,
                 kernel_size: int = 25):
        super().__init__()
        self.revin   = RevIN(n_variates)
        self.decomp  = _MovingAvg(kernel_size)
        self.lin_s   = nn.Linear(seq_len, pred_len)
        self.lin_t   = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, C) → (B, H, C)"""
        x = self.revin(x, mode='norm')
        seasonal, trend = self.decomp(x)              # (B, L, C) each
        out = (self.lin_s(seasonal.transpose(1, 2))
             + self.lin_t(trend.transpose(1, 2))).transpose(1, 2)  # (B, H, C)
        return self.revin(out, mode='denorm')


# =============================================================================
# PatchTST
# =============================================================================

class PatchTST(nn.Module):
    """Patched temporal tokens + Transformer encoder (Nie et al. 2023).

    Each variate is processed independently:
      - Slice lookback into overlapping patches of length patch_len
      - Project patches to d_model
      - N transformer encoder layers
      - Flatten + linear head → pred_len output

    RevIN normalization applied per-variate.
    """

    def __init__(self, seq_len: int, pred_len: int, n_variates: int,
                 d_model: int = 128, n_heads: int = 8, n_layers: int = 3,
                 patch_len: int = 16, stride: int = 8, dropout: float = 0.1):
        super().__init__()
        self.seq_len   = seq_len
        self.pred_len  = pred_len
        self.n_variates = n_variates
        self.patch_len = patch_len
        self.stride    = stride

        # Number of patches
        n_patches = (seq_len - patch_len) // stride + 1
        self.n_patches = n_patches

        self.revin      = RevIN(n_variates)
        self.patch_emb  = nn.Linear(patch_len, d_model)
        self.pos_emb    = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        self.dropout    = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(n_patches * d_model, pred_len)

        nn.init.xavier_uniform_(self.patch_emb.weight)
        nn.init.xavier_uniform_(self.head.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, C) → (B, H, C)"""
        B, L, C = x.shape
        x = self.revin(x, mode='norm')        # (B, L, C)

        # Patch: (B, L, C) → (B*C, n_patches, patch_len)
        x = x.transpose(1, 2).reshape(B * C, L)   # (B*C, L)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # patches: (B*C, n_patches, patch_len)

        z = self.dropout(self.patch_emb(patches) + self.pos_emb)  # (B*C, n_patches, d)
        z = self.encoder(z)                                         # (B*C, n_patches, d)
        z = z.reshape(B * C, -1)                                    # (B*C, n_patches*d)
        out = self.head(z).reshape(B, C, self.pred_len)            # (B, C, H)
        out = out.transpose(1, 2)                                   # (B, H, C)
        return self.revin(out, mode='denorm')


# =============================================================================
# LIV-based fixed baselines
# =============================================================================

def _seq_backbone(block_fn, dim: int, n_layers: int, **kw) -> nn.Sequential:
    """Build nn.Sequential of n_layers UnifiedLIVBlocks using block_fn."""
    return nn.Sequential(*[
        UnifiedLIVBlock(dim, block_fn(dim, **kw))
        for _ in range(n_layers)
    ])


def build_dmamba_baseline(n_variates: int, seq_len: int, pred_len: int,
                          dim: int = 256, num_layers: int = 4) -> STARLIVTSModel:
    """DMamba: Rec1 seasonal + GMemless trend (causal, fixed genome)."""
    half = max(1, num_layers // 2)
    seasonal_bb = _seq_backbone(Rec1,     dim, half)
    trend_bb    = _seq_backbone(GMemless, dim, half)
    return STARLIVTSModel(
        seasonal_bb, n_variates, seq_len, pred_len, dim,
        structure='dmamba', backbone_aux=trend_bb,
    )


def build_smamba_baseline(n_variates: int, seq_len: int, pred_len: int,
                          dim: int = 256, num_layers: int = 4) -> STARLIVTSModel:
    """S-Mamba: SA1 variate-correlation + GMemless temporal (bidirectional, fixed genome)."""
    half = max(1, num_layers // 2)
    variate_bb  = _seq_backbone(SA1,      dim, half)
    temporal_bb = _seq_backbone(GMemless, dim, half)
    return STARLIVTSModel(
        variate_bb, n_variates, seq_len, pred_len, dim,
        structure='smamba', backbone_aux=temporal_bb,
    )


def build_itransformer_baseline(n_variates: int, seq_len: int, pred_len: int,
                                dim: int = 256, num_layers: int = 4) -> STARLIVTSModel:
    """iTransformer: interleaved SA1 + GMemless per block (bidirectional, fixed genome)."""
    blocks = []
    for _ in range(num_layers // 2):
        blocks.append(UnifiedLIVBlock(dim, SA1(dim)))
        blocks.append(UnifiedLIVBlock(dim, GMemless(dim)))
    if num_layers % 2 == 1:
        blocks.append(UnifiedLIVBlock(dim, SA1(dim)))
    backbone = nn.Sequential(*blocks)
    return STARLIVTSModel(
        backbone, n_variates, seq_len, pred_len, dim,
        structure='itransformer',
    )


def build_baseline(name: str, n_variates: int, seq_len: int, pred_len: int,
                   dim: int = 256, num_layers: int = 4) -> nn.Module:
    """Factory: return a baseline model by name."""
    name = name.lower()
    if name == 'dlinear':
        return DLinear(seq_len, pred_len, n_variates)
    elif name == 'patchtst':
        return PatchTST(seq_len, pred_len, n_variates,
                        d_model=dim, n_heads=min(8, dim // 16), n_layers=num_layers)
    elif name == 'dmamba':
        return build_dmamba_baseline(n_variates, seq_len, pred_len, dim, num_layers)
    elif name == 'smamba':
        return build_smamba_baseline(n_variates, seq_len, pred_len, dim, num_layers)
    elif name == 'itransformer':
        return build_itransformer_baseline(n_variates, seq_len, pred_len, dim, num_layers)
    else:
        raise ValueError(f"Unknown baseline: {name}. "
                         f"Choose from: dlinear, patchtst, dmamba, smamba, itransformer")


# =============================================================================
# Training wrapper
# =============================================================================

def _make_train_args(args: Namespace, device: str) -> Namespace:
    """Build a minimal args namespace compatible with train_ts_model / evaluate_ts."""
    return Namespace(
        device=device,
        lr=args.lr,
        beta1=0.9,
        beta2=0.95,
        weight_decay=0.1,
        grad_clip=1.0,
        warmup_steps=min(500, args.train_steps // 10),
        train_steps=args.train_steps,
        amp=args.amp,
    )


def run_baseline(
    name: str,
    dataset: str,
    pred_len: int,
    seq_len: int = 96,
    dim: int = 256,
    num_layers: int = 4,
    train_steps: int = 10_000,
    batch_size: int = 32,
    lr: float = 1e-3,
    amp: bool = False,
    num_workers: int = 0,
    device: str = None,
) -> dict:
    """Train and evaluate one baseline. Returns result dict."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    info = get_dataset_info(dataset)
    if not info.get("exists"):
        raise FileNotFoundError(f"Dataset '{dataset}' not found. Run ./download_data.sh")
    n_variates = info["n_variates"]

    train_dl, val_dl, test_dl, _ = get_dataloader(
        dataset=dataset, seq_len=seq_len, pred_len=pred_len,
        batch_size=batch_size, num_workers=num_workers,
    )

    model = build_baseline(name, n_variates, seq_len, pred_len, dim, num_layers)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"[{name}] {dataset} H={pred_len}  params={n_params:,}")

    train_args = _make_train_args(
        Namespace(lr=lr, train_steps=train_steps, amp=amp), device
    )

    t0 = time.perf_counter()
    val_mse = train_ts_model(model, train_dl, val_dl, train_args,
                             prefix=f"[{name}/{dataset}/H{pred_len}] ")
    train_sec = time.perf_counter() - t0

    model = model.to(device)
    test_mse, test_mae = evaluate_ts(model, test_dl, train_args)
    log.info(f"[{name}] {dataset} H={pred_len}  "
             f"val_MSE={val_mse:.4f}  test_MSE={test_mse:.4f}  test_MAE={test_mae:.4f}")

    return {
        "model": name,
        "dataset": dataset,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "n_params": n_params,
        "val_mse": val_mse,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "train_sec": round(train_sec, 1),
    }


# =============================================================================
# CLI
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser(description="Train and evaluate time series baselines")
    p.add_argument("--model", type=str, default="all",
                   help="Baseline name or 'all'. "
                        "Choices: dlinear, patchtst, dmamba, smamba, itransformer, all")
    p.add_argument("--dataset", type=str, default="ETTh1")
    p.add_argument("--all_datasets", action="store_true",
                   help="Run on all 7 benchmark datasets")
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--all_horizons", action="store_true",
                   help="Run all 4 horizons: 96, 192, 336, 720")
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--train_steps", type=int, default=10_000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="baseline_results")
    return p


def main():
    args = build_parser().parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    models = (["dlinear", "patchtst", "dmamba", "smamba", "itransformer"]
              if args.model == "all" else [args.model])
    datasets = ALL_DATASETS if args.all_datasets else [args.dataset]
    horizons = HORIZONS if args.all_horizons else [args.pred_len]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for dataset in datasets:
        for pred_len in horizons:
            for model_name in models:
                try:
                    result = run_baseline(
                        name=model_name, dataset=dataset, pred_len=pred_len,
                        seq_len=args.seq_len, dim=args.dim, num_layers=args.num_layers,
                        train_steps=args.train_steps, batch_size=args.batch_size,
                        lr=args.lr, amp=args.amp, num_workers=args.num_workers,
                        device=device,
                    )
                    all_results.append(result)
                except Exception as e:
                    log.error(f"[{model_name}/{dataset}/H{pred_len}] FAILED: {e}")

    out_path = out_dir / "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Results saved to {out_path}")

    # Print summary table
    print(f"\n{'Model':<15} {'Dataset':<12} {'H':>4} {'test_MSE':>10} {'test_MAE':>10} {'params':>10}")
    print("-" * 65)
    for r in all_results:
        print(f"{r['model']:<15} {r['dataset']:<12} {r['pred_len']:>4} "
              f"{r['test_mse']:>10.4f} {r['test_mae']:>10.4f} {r['n_params']:>10,}")


if __name__ == "__main__":
    main()