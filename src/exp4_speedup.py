"""Experiment 4: TiDAR-TS Inference Speedup Benchmark.

Compares wall-clock inference time and forecast quality across TiDARTSModel
inference modes:
  draft (ar_steps=0) — single forward pass, all H steps in parallel
  partial-AR (k)     — draft + k AR-refined steps at the start
  full-AR (ar_steps=H) — H sequential passes (equivalent to standard AR)

The speedup claim: draft is ~H× faster than full-AR at equal or near-equal MSE
for short horizons, with quality degrading gracefully for longer horizons.

What gets measured per mode:
  - Wall-clock ms / sample  (median over test set)
  - MSE / MAE vs ground truth

The model is either:
  a) Loaded from a ts-tidar-train checkpoint  (--checkpoint path)
  b) Built fresh from a genome checkpoint     (--genome_checkpoint path)
  c) Built from scratch with a default genome (no checkpoint)

Usage:
  # Quick benchmark with fresh default model
  python -m src.exp4_speedup --dataset ETTh1 --pred_len 720

  # Use a trained TiDAR-TS checkpoint
  python -m src.exp4_speedup --dataset ETTh1 --pred_len 720 \\
      --checkpoint exp1_results/ETTh1_H720_dmamba/tidar_ts_candidate_1.pt

  # All horizons
  python -m src.exp4_speedup --dataset ETTh1 --all_horizons
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from core.nsga import build_class_pool, Genome, LayerGene
from core.tidar import TiDARTSConfig, TiDARTSModel
from src.dataload import get_dataloader, get_dataset_info
from src.exp2_ablation import make_uniform_genome
from src.train import _build_tidar_ts_model, train_tidar_ts_model, count_params

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HORIZONS = [96, 192, 336, 720]


# =============================================================================
# Model loading
# =============================================================================

def load_tidar_model(
    checkpoint: str,
    genome_checkpoint: str,
    n_variates: int,
    seq_len: int,
    pred_len: int,
    dim: int,
    num_layers: int,
    device: str,
) -> TiDARTSModel:
    """Return a TiDARTSModel, sourcing weights from checkpoint if provided."""
    from argparse import Namespace
    from src.exp1_search import make_args
    from src.exp2_ablation import load_genome_from_checkpoint

    if checkpoint:
        # Full TiDAR-TS checkpoint: load model state directly
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
        pool = build_class_pool(include_extended=True)
        genome = Genome.from_flat(ckpt['genome'], num_layers)
        args = make_args(
            dataset='ETTh1', pred_len=pred_len, structure='itransformer',
            out_dir='/tmp', dim=dim, num_layers=num_layers, seq_len=seq_len,
            device=device, include_extended=True,
        )
        args.n_variates = n_variates
        args.ts_seq_len = seq_len
        args.ts_pred_len = pred_len
        model = _build_tidar_ts_model(genome, pool, args)
        model.load_state_dict(ckpt['model_state_dict'])
        log.info(f"Loaded TiDAR-TS model from {checkpoint}")

    elif genome_checkpoint:
        # Load genome only, random weights (for architecture comparison)
        pool = build_class_pool(include_extended=True)
        genome = load_genome_from_checkpoint(genome_checkpoint, num_layers)
        args = make_args(
            dataset='ETTh1', pred_len=pred_len, structure='itransformer',
            out_dir='/tmp', dim=dim, num_layers=num_layers, seq_len=seq_len,
            device=device, include_extended=True,
        )
        args.n_variates = n_variates
        args.ts_seq_len = seq_len
        args.ts_pred_len = pred_len
        model = _build_tidar_ts_model(genome, pool, args)
        log.info(f"Built TiDAR-TS from genome checkpoint {genome_checkpoint} (random weights)")

    else:
        # Default: uniform Rec-1 genome, fresh weights
        pool = build_class_pool(include_extended=False)
        genome = make_uniform_genome(5, num_layers, pool)  # Rec-1
        args = make_args(
            dataset='ETTh1', pred_len=pred_len, structure='itransformer',
            out_dir='/tmp', dim=dim, num_layers=num_layers, seq_len=seq_len,
            device=device,
        )
        args.n_variates = n_variates
        args.ts_seq_len = seq_len
        args.ts_pred_len = pred_len
        model = _build_tidar_ts_model(genome, pool, args)
        log.info(f"Built default TiDAR-TS model (uniform Rec-1, fresh weights)")

    return model


# =============================================================================
# Timing benchmark
# =============================================================================

@torch.no_grad()
def benchmark_mode(
    model: TiDARTSModel,
    test_loader,
    ar_steps: int,
    device: str,
    n_warmup: int = 3,
) -> dict:
    """Benchmark one inference mode. Returns timing + quality metrics."""
    model.eval()
    model = model.to(device)

    latencies = []        # ms per sample
    total_mse = 0.0
    total_mae = 0.0
    n_elements = 0

    for i, (batch_x, batch_y) in enumerate(test_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        B = batch_x.size(0)

        if i < n_warmup:
            # Warmup: run but don't time
            _ = model.forecast(batch_x, ar_steps=ar_steps)
            continue

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = model.forecast(batch_x, ar_steps=ar_steps)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        latencies.append(elapsed_ms / B)   # per-sample ms
        total_mse += F.mse_loss(pred, batch_y, reduction='sum').item()
        total_mae += F.l1_loss(pred, batch_y, reduction='sum').item()
        n_elements += batch_y.numel()

    latencies.sort()
    n = len(latencies)
    median_ms = latencies[n // 2] if n > 0 else float('nan')
    p90_ms    = latencies[int(n * 0.9)] if n > 0 else float('nan')

    return {
        "ar_steps": ar_steps,
        "mode": "draft" if ar_steps == 0 else f"partial-AR(k={ar_steps})" if ar_steps < 999 else "full-AR",
        "median_ms_per_sample": round(median_ms, 3),
        "p90_ms_per_sample": round(p90_ms, 3),
        "test_mse": total_mse / max(n_elements, 1),
        "test_mae": total_mae / max(n_elements, 1),
        "n_batches_timed": n,
    }


# =============================================================================
# Per-horizon benchmark
# =============================================================================

def run_speedup_benchmark(
    dataset: str,
    pred_len: int,
    seq_len: int,
    dim: int,
    num_layers: int,
    checkpoint: str,
    genome_checkpoint: str,
    train_steps: int,
    batch_size: int,
    lr: float,
    amp: bool,
    num_workers: int,
    device: str,
    out_dir: str,
) -> list:
    """Full benchmark: train model if needed, then time all inference modes."""
    info = get_dataset_info(dataset)
    if not info.get("exists"):
        raise FileNotFoundError(f"Dataset '{dataset}' not found. Run ./download_data.sh")
    n_variates = info["n_variates"]

    train_dl, val_dl, test_dl, _ = get_dataloader(
        dataset=dataset, seq_len=seq_len, pred_len=pred_len,
        batch_size=batch_size, num_workers=num_workers,
    )

    model = load_tidar_model(
        checkpoint=checkpoint,
        genome_checkpoint=genome_checkpoint,
        n_variates=n_variates,
        seq_len=seq_len, pred_len=pred_len,
        dim=dim, num_layers=num_layers, device=device,
    )
    log.info(f"Model params: {count_params(model):,}")

    # Train if no pre-trained weights provided
    if not checkpoint and train_steps > 0:
        from argparse import Namespace
        train_args = Namespace(
            device=device, lr=lr, beta1=0.9, beta2=0.95,
            weight_decay=0.1, grad_clip=1.0,
            warmup_steps=min(500, train_steps // 10),
            train_steps=train_steps, amp=amp,
        )
        log.info(f"Training for {train_steps} steps before benchmarking...")
        train_tidar_ts_model(model, train_dl, val_dl, train_args,
                             steps=train_steps, prefix="[pretrain] ")

    # AR steps to benchmark: 0 (draft), partial steps, full H
    partial_steps = [1, 4, 16]
    partial_steps = [k for k in partial_steps if k < pred_len]
    ar_step_list = [0] + partial_steps + [pred_len]   # 0 = draft, pred_len = full-AR

    results = []
    for ar_steps in ar_step_list:
        log.info(f"Benchmarking ar_steps={ar_steps} ({'draft' if ar_steps == 0 else 'full-AR' if ar_steps == pred_len else f'partial-k={ar_steps}'})...")
        r = benchmark_mode(model, test_dl, ar_steps=ar_steps, device=device)
        r.update({"dataset": dataset, "pred_len": pred_len, "n_params": count_params(model)})
        results.append(r)
        log.info(f"  median={r['median_ms_per_sample']:.2f}ms  "
                 f"MSE={r['test_mse']:.4f}  MAE={r['test_mae']:.4f}")

    # Compute speedup relative to full-AR
    full_ar = next(r for r in results if r["ar_steps"] == pred_len)
    for r in results:
        if full_ar["median_ms_per_sample"] > 0:
            r["speedup_vs_full_ar"] = round(
                full_ar["median_ms_per_sample"] / r["median_ms_per_sample"], 2)
        else:
            r["speedup_vs_full_ar"] = None

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / "speedup_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Speedup results saved to {out_path}")

    return results


# =============================================================================
# CLI
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser(description="Exp 4: TiDAR-TS Speedup Benchmark")
    p.add_argument("--dataset", type=str, default="ETTh1")
    p.add_argument("--pred_len", type=int, default=720)
    p.add_argument("--all_horizons", action="store_true",
                   help="Run all 4 horizons: 96, 192, 336, 720")
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)
    # Model source: pick one
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to tidar_ts_candidate_*.pt (trained model)")
    p.add_argument("--genome_checkpoint", type=str, default=None,
                   help="Path to ts_candidate_*.pt (genome only, random weights)")
    # If no checkpoint: train first
    p.add_argument("--train_steps", type=int, default=5_000,
                   help="Steps to train before benchmarking (if no checkpoint)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="exp4_results")
    return p


def main():
    args = build_parser().parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    horizons = HORIZONS if args.all_horizons else [args.pred_len]
    all_results = []

    for pred_len in horizons:
        run_dir = str(Path(args.out_dir) / f"{args.dataset}_H{pred_len}")
        results = run_speedup_benchmark(
            dataset=args.dataset, pred_len=pred_len, seq_len=args.seq_len,
            dim=args.dim, num_layers=args.num_layers,
            checkpoint=args.checkpoint, genome_checkpoint=args.genome_checkpoint,
            train_steps=args.train_steps,
            batch_size=args.batch_size, lr=args.lr, amp=args.amp,
            num_workers=args.num_workers, device=device, out_dir=run_dir,
        )
        all_results.extend(results)

    # Global summary
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir) / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print table
    print(f"\n{'Mode':<22} {'H':>4} {'ms/sample':>10} {'Speedup':>8} "
          f"{'MSE':>8} {'MAE':>8}")
    print("-" * 65)
    for r in all_results:
        spd = f"{r['speedup_vs_full_ar']}×" if r['speedup_vs_full_ar'] else "—"
        print(f"{r['mode']:<22} {r['pred_len']:>4} "
              f"{r['median_ms_per_sample']:>10.3f} {spd:>8} "
              f"{r['test_mse']:>8.4f} {r['test_mae']:>8.4f}")


if __name__ == "__main__":
    main()