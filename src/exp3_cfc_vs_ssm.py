"""Experiment 3: CfC vs SSM Head-to-Head.

For the same structure and slot assignment, swap all SSM layers for CfC (Rec-4)
or vice versa, keeping everything else identical. This isolates the operator
choice from architecture/training differences.

Two swap modes:
  ssm_only  — start from a uniform Rec-1 genome; also run Rec-4 genome
  from_base — start from a base genome (Exp 1 checkpoint or uniform);
              produce an SSM variant and a CfC variant by targeted swapping

Expected result: CfC (Rec-4) wins on datasets with irregular intervals or
continuous physical dynamics (Weather, ILI); SSM is comparable on regular
benchmark datasets (ETTh1, ETTm1).

SSM class IDs  : 5  (Rec-1), 6  (Rec-2), 17 (Diff-Rec-1),
                 18 (Rec-3), 20 (Diff-Rec-3)
CfC class IDs  : 19 (Rec-4), 21 (Diff-Rec-4)

Usage:
  # Quick comparison, ETTh1
  python -m src.exp3_cfc_vs_ssm --dataset ETTh1 --pred_len 96

  # Focus datasets: Weather + ILI (irregular intervals)
  python -m src.exp3_cfc_vs_ssm --irregular_focus --all_horizons

  # Full comparison: all datasets, all horizons
  python -m src.exp3_cfc_vs_ssm --all_datasets --all_horizons

  # Start from a specific Exp-1 genome
  python -m src.exp3_cfc_vs_ssm --dataset ETTh1 --pred_len 96 \\
      --base_genome exp1_results/ETTh1_H96_dmamba/ts_candidate_1.pt
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from core.nsga import (
    build_class_pool, Genome, LayerGene, repair, VALID_FG_MASKS,
)
from src.dataload import get_dataloader, get_dataset_info
from src.exp1_search import make_args
from src.exp2_ablation import make_uniform_genome, load_genome_from_checkpoint
from src.train import _build_ts_model, train_ts_model, evaluate_ts, count_params

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HORIZONS     = [96, 192, 336, 720]
ALL_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Electricity", "Traffic", "Exchange"]
IRREGULAR_DATASETS = ["Weather", "ILI"]  # irregular / continuous-dynamics focus

# Class ID sets for swap operations
SSM_CLASSES = {5, 6, 17, 18, 20}   # Rec-1/2/Diff and Rec-3/Diff
CFC_CLASSES = {19, 21}              # Rec-4 (CfC), Diff-Rec-4


# =============================================================================
# Swap operations
# =============================================================================

def swap_to_ssm(genome: Genome, class_pool: dict,
                target_class: int = 5) -> Genome:
    """Replace all CfC layers with Rec-1 (or target_class).

    Non-CfC layers (attention, conv, FFN) are left unchanged.
    """
    g = genome.copy()
    for gene in g.layers:
        if gene.liv_class in CFC_CLASSES:
            gene.liv_class = target_class
    repair(g, class_pool)
    return g


def swap_to_cfc(genome: Genome, class_pool: dict,
                target_class: int = 19) -> Genome:
    """Replace all SSM layers with Rec-4 (CfC), or target_class.

    Non-SSM layers (attention, conv, FFN) are left unchanged.
    """
    g = genome.copy()
    for gene in g.layers:
        if gene.liv_class in SSM_CLASSES:
            gene.liv_class = target_class
    repair(g, class_pool)
    return g


def force_uniform(genome: Genome, class_pool: dict,
                  class_id: int) -> Genome:
    """Force every layer to the given class_id."""
    g = genome.copy()
    for gene in g.layers:
        gene.liv_class = class_id
    repair(g, class_pool)
    return g


# =============================================================================
# Single-variant runner (shared with exp2)
# =============================================================================

def run_variant(
    variant_name: str,
    genome: Genome,
    class_pool: dict,
    train_dl, val_dl, test_dl,
    args,
) -> dict:
    model = _build_ts_model(genome, class_pool, args)
    n_params = count_params(model)
    classes = [g.liv_class for g in genome.layers]
    log.info(f"[{variant_name}] params={n_params:,}  classes={classes}")

    val_mse = train_ts_model(
        model, train_dl, val_dl, args,
        steps=args.train_steps,
        prefix=f"[{variant_name}] ",
    )
    model = model.to(args.device)
    test_mse, test_mae = evaluate_ts(model, test_dl, args)
    log.info(f"[{variant_name}] val={val_mse:.4f}  "
             f"test_MSE={test_mse:.4f}  MAE={test_mae:.4f}")

    return {
        "variant": variant_name,
        "layer_classes": classes,
        "n_params": n_params,
        "val_mse": val_mse,
        "test_mse": test_mse,
        "test_mae": test_mae,
    }


# =============================================================================
# Per-(dataset, horizon) comparison
# =============================================================================

def run_comparison(
    dataset: str,
    pred_len: int,
    structure: str,
    base_genome_path: str,      # None → use uniform SSM as base
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
    """Train SSM and CfC variants; return list of result dicts."""
    # CfC always needs include_extended
    pool = build_class_pool(include_extended=True)

    args = make_args(
        dataset=dataset, pred_len=pred_len, structure=structure,
        out_dir=out_dir, dim=dim, num_layers=num_layers, seq_len=seq_len,
        lr=lr, batch_size=batch_size, num_workers=num_workers, amp=amp,
        device=device, full_train_steps=train_steps,
        include_extended=True,
    )
    args.train_steps = train_steps

    info = get_dataset_info(dataset)
    if not info.get("exists"):
        raise FileNotFoundError(f"Dataset '{dataset}' not found. Run ./download_data.sh")
    args.n_variates = info["n_variates"]

    train_dl, val_dl, test_dl, _ = get_dataloader(
        dataset=dataset, seq_len=seq_len, pred_len=pred_len,
        batch_size=batch_size, num_workers=num_workers,
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ── Determine base genome ─────────────────────────────────────────────────
    if base_genome_path:
        base = load_genome_from_checkpoint(base_genome_path, num_layers)
        repair(base, pool)
        base_name = "FROM_EXP1"
    else:
        # Default: uniform Rec-1 genome as the SSM anchor
        base = make_uniform_genome(5, num_layers, pool)
        base_name = "UNIFORM_SSM"

    # ── Produce paired variants ───────────────────────────────────────────────
    # SSM variant: replace any CfC layers in base with Rec-1
    ssm_genome = swap_to_ssm(base, pool, target_class=5)
    # CfC variant: replace all SSM layers in base with Rec-4
    cfc_genome = swap_to_cfc(base, pool, target_class=19)

    results = []
    for name, genome in [("SSM_Rec1", ssm_genome), ("CfC_Rec4", cfc_genome)]:
        r = run_variant(name, genome, pool, train_dl, val_dl, test_dl, args)
        r.update({
            "dataset": dataset,
            "pred_len": pred_len,
            "structure": structure,
            "base": base_name,
        })
        results.append(r)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = Path(out_dir) / "cfc_vs_ssm.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Log delta
    ssm_r = next(r for r in results if r["variant"] == "SSM_Rec1")
    cfc_r = next(r for r in results if r["variant"] == "CfC_Rec4")
    delta = cfc_r["test_mse"] - ssm_r["test_mse"]
    winner = "CfC" if delta < 0 else "SSM"
    log.info(f"[{dataset} H={pred_len}] ΔMSE(CfC-SSM)={delta:+.4f}  winner={winner}")

    return results


# =============================================================================
# CLI
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser(description="Exp 3: CfC vs SSM Head-to-Head")
    p.add_argument("--dataset", type=str, default="ETTh1")
    p.add_argument("--all_datasets", action="store_true",
                   help="Run on all 7 benchmark datasets")
    p.add_argument("--irregular_focus", action="store_true",
                   help="Run only on Weather + ILI (irregular interval datasets)")
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--all_horizons", action="store_true",
                   help="Run all 4 horizons: 96, 192, 336, 720")
    p.add_argument("--structure", type=str, default="dmamba",
                   choices=["dmamba", "smamba", "itransformer"])
    p.add_argument("--base_genome", type=str, default=None,
                   help="Path to ts_candidate_*.pt from Exp 1. "
                        "If not given, uses uniform Rec-1 as SSM base.")
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--train_steps", type=int, default=10_000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="exp3_results")
    return p


def main():
    args = build_parser().parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.irregular_focus:
        datasets = IRREGULAR_DATASETS
    elif args.all_datasets:
        datasets = ALL_DATASETS
    else:
        datasets = [args.dataset]

    horizons = HORIZONS if args.all_horizons else [args.pred_len]
    all_results = []

    for dataset in datasets:
        for pred_len in horizons:
            run_dir = str(
                Path(args.out_dir) / f"{dataset}_H{pred_len}_{args.structure}"
            )
            try:
                results = run_comparison(
                    dataset=dataset,
                    pred_len=pred_len,
                    structure=args.structure,
                    base_genome_path=args.base_genome,
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
                log.error(f"Failed for {dataset}/H{pred_len}: {e}")
                raise

    # Global summary
    summary_path = Path(args.out_dir) / "summary.json"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print comparison table: SSM vs CfC delta per dataset/horizon
    print(f"\n{'Dataset':<12} {'H':>4} {'SSM_MSE':>10} {'CfC_MSE':>10} "
          f"{'ΔMSE':>8} {'Winner':>7}")
    print("-" * 57)

    rows = {}
    for r in all_results:
        key = (r["dataset"], r["pred_len"])
        rows.setdefault(key, {})[r["variant"]] = r

    for (dataset, pred_len), variants in sorted(rows.items()):
        ssm = variants.get("SSM_Rec1", {})
        cfc = variants.get("CfC_Rec4", {})
        if not ssm or not cfc:
            continue
        delta = cfc["test_mse"] - ssm["test_mse"]
        winner = "CfC" if delta < 0 else "SSM"
        print(f"{dataset:<12} {pred_len:>4} "
              f"{ssm['test_mse']:>10.4f} {cfc['test_mse']:>10.4f} "
              f"{delta:>+8.4f} {winner:>7}")

    log.info(f"\nGlobal summary saved to {summary_path}")


if __name__ == "__main__":
    main()