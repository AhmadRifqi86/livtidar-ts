"""Experiment 6: Alpha Regularization Ablation.

Studies the effect of the AR auxiliary loss weight (alpha) on the draft head's
forecast quality. All variants are evaluated with ar_steps=0 (pure draft mode),
so differences in MSE are caused entirely by the regularization strength during
training — not by AR refinement at inference.

Loss formula:
    L = 1 / (1 + alpha) * [alpha * L_AR + L_Diff]

    alpha = 0.0  → pure diffusion head, no AR regularization
    alpha = 1.0  → equal weight (default)
    alpha = inf  → backbone trained as a next-step predictor only

Hypothesis: the draft head quality peaks at some intermediate alpha because:
    - alpha=0 removes the causal regularizer → backbone takes shortcuts
    - alpha too high → backbone optimized for next-step task, not H-step forecast

Usage:
    # Dry run (pipeline check)
    python -m src.exp6_alpha_ablation --dataset ETTh1 --pred_len 96 --train_steps 2

    # Smoke test
    python -m src.exp6_alpha_ablation --dataset ETTh1 --pred_len 96 \\
        --base_genome exp1_results/ETTh1_H96_itransformer/ts_candidate_1.pt \\
        --train_steps 500 --batch_size 16 --out_dir exp6_results_smoke

    # Full run
    python -m src.exp6_alpha_ablation --dataset ETTh1 --all_horizons \\
        --base_genome exp1_results/ETTh1_H96_itransformer/ts_candidate_1.pt \\
        --train_steps 30000 --batch_size 16 --out_dir exp6_results
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from core.nsga import build_class_pool, repair
from src.dataload import get_dataloader, get_dataset_info
from src.exp1_search import make_args
from src.exp2_ablation import make_uniform_genome, load_genome_from_checkpoint
from src.train import _build_tidar_ts_model, train_tidar_ts_model, evaluate_tidar_ts, count_params

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HORIZONS     = [96, 192, 336, 720]
DEFAULT_ALPHAS = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]


# =============================================================================
# Per-(dataset, horizon) alpha sweep
# =============================================================================

def run_alpha_sweep(
    dataset: str,
    pred_len: int,
    structure: str,
    base_genome_path: str,
    alphas: list,
    train_steps: int,
    dim: int,
    num_layers: int,
    seq_len: int,
    batch_size: int,
    lr: float,
    amp: bool,
    num_workers: int,
    device: str,
    out_dir: str,
) -> list:
    """Train one TiDARTSModel per alpha value; evaluate all in draft mode."""
    pool = build_class_pool(include_extended=True)

    info = get_dataset_info(dataset)
    if not info.get("exists"):
        raise FileNotFoundError(f"Dataset '{dataset}' not found. Run ./download_data.sh")
    n_variates = info["n_variates"]

    train_dl, val_dl, test_dl, _ = get_dataloader(
        dataset=dataset, seq_len=seq_len, pred_len=pred_len,
        batch_size=batch_size, num_workers=num_workers,
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Load or build genome (same genome used for all alpha variants)
    if base_genome_path:
        genome = load_genome_from_checkpoint(base_genome_path, num_layers)
        repair(genome, pool)
        genome_name = "FROM_EXP1"
    else:
        genome = make_uniform_genome(5, num_layers, pool)   # uniform Rec-1
        genome_name = "UNIFORM_SSM"

    layer_classes = [g.liv_class for g in genome.layers]
    log.info(f"Genome: {layer_classes}  ({genome_name})")

    results = []

    for alpha in alphas:
        variant = f"alpha={alpha:.2f}"
        log.info(f"\n{'─'*50}")
        log.info(f"[{variant}] dataset={dataset}  H={pred_len}  structure={structure}")

        # Build args with tidar_alpha set for this variant
        args = make_args(
            dataset=dataset, pred_len=pred_len, structure=structure,
            out_dir=out_dir, dim=dim, num_layers=num_layers, seq_len=seq_len,
            lr=lr, batch_size=batch_size, num_workers=num_workers, amp=amp,
            device=device, full_train_steps=train_steps, include_extended=True,
        )
        args.train_steps = train_steps
        args.tidar_alpha = alpha

        model = _build_tidar_ts_model(genome, pool, args)
        n_params = count_params(model)
        log.info(f"[{variant}] params={n_params:,}")

        val_mse = train_tidar_ts_model(
            model, train_dl, val_dl, args,
            steps=train_steps,
            prefix=f"[{variant}] ",
        )
        model = model.to(device)
        test_mse, test_mae = evaluate_tidar_ts(model, test_dl, args)
        log.info(f"[{variant}] val={val_mse:.4f}  test_MSE={test_mse:.4f}  MAE={test_mae:.4f}")

        results.append({
            "alpha": alpha,
            "variant": variant,
            "dataset": dataset,
            "pred_len": pred_len,
            "structure": structure,
            "genome": layer_classes,
            "genome_source": genome_name,
            "n_params": n_params,
            "val_mse": val_mse,
            "test_mse": test_mse,
            "test_mae": test_mae,
        })

    # Save per-(dataset, horizon) results
    out_path = Path(out_dir) / "alpha_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Alpha results saved to {out_path}")

    # Print comparison table
    baseline = next((r for r in results if r["alpha"] == 0.0), None)
    print(f"\n{'Alpha':>8}  {'val_MSE':>10}  {'test_MSE':>10}  {'test_MAE':>10}  {'ΔMSE vs α=0':>12}")
    print("─" * 60)
    for r in results:
        delta = f"{r['test_mse'] - baseline['test_mse']:+.4f}" if baseline else "—"
        print(f"{r['alpha']:>8.2f}  {r['val_mse']:>10.4f}  {r['test_mse']:>10.4f}  "
              f"{r['test_mae']:>10.4f}  {delta:>12}")

    best = min(results, key=lambda r: r["test_mse"])
    log.info(f"\nBest alpha: {best['alpha']}  test_MSE={best['test_mse']:.4f}")

    return results


# =============================================================================
# CLI
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser(description="Exp 6: Alpha Regularization Ablation")
    p.add_argument("--dataset", type=str, default="ETTh1")
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--all_horizons", action="store_true",
                   help="Run all 4 horizons: 96, 192, 336, 720")
    p.add_argument("--structure", type=str, default="itransformer",
                   choices=["dmamba", "smamba", "itransformer"])
    p.add_argument("--base_genome", type=str, default=None,
                   help="Path to ts_candidate_*.pt from Exp 1. "
                        "If not given, uses uniform Rec-1 genome.")
    p.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS,
                   help="Alpha values to sweep (default: 0.0 0.1 0.5 1.0 2.0 5.0)")
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--train_steps", type=int, default=10_000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="exp6_results")
    return p


def main():
    args = build_parser().parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    horizons = HORIZONS if args.all_horizons else [args.pred_len]
    all_results = []

    for pred_len in horizons:
        run_dir = str(Path(args.out_dir) / f"{args.dataset}_H{pred_len}_{args.structure}")
        try:
            results = run_alpha_sweep(
                dataset=args.dataset,
                pred_len=pred_len,
                structure=args.structure,
                base_genome_path=args.base_genome,
                alphas=args.alphas,
                train_steps=args.train_steps,
                dim=args.dim,
                num_layers=args.num_layers,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                lr=args.lr,
                amp=args.amp,
                num_workers=args.num_workers,
                device=device,
                out_dir=run_dir,
            )
            all_results.extend(results)
        except Exception as e:
            log.error(f"Failed for {args.dataset}/H{pred_len}: {e}")
            raise

    # Global summary
    summary_path = Path(args.out_dir) / "summary.json"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Global summary saved to {summary_path}")

    # Global best alpha per horizon
    print(f"\n{'H':>4}  {'Best Alpha':>10}  {'Best MSE':>10}")
    print("─" * 30)
    for pred_len in horizons:
        horizon_results = [r for r in all_results if r["pred_len"] == pred_len]
        if horizon_results:
            best = min(horizon_results, key=lambda r: r["test_mse"])
            print(f"{pred_len:>4}  {best['alpha']:>10.2f}  {best['test_mse']:>10.4f}")


if __name__ == "__main__":
    main()