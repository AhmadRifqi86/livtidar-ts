"""Experiment 2: Operator Ablation.

For a fixed structural template (default: dmamba), compare training each
operator family uniformly against the GA-discovered mixed genome from Exp 1.

Ablation variants:
  ALL_SSM   — every layer is Rec-1 (Mamba-like SSM)
  ALL_CFC   — every layer is Rec-4 (CfC/LNN)  [requires --include_extended]
  ALL_CONV  — every layer is GConv-1 (short depthwise conv)
  ALL_ATTN  — every layer is SA-1 (standard multi-head attention)
  ALL_FFN   — every layer is GMemless (SwiGLU FFN, no token mixing)
  MIXED     — GA-found genome loaded from an Exp-1 checkpoint

Hypothesis: the GA-found MIXED combination outperforms any single-family
uniform architecture, justifying the search cost.

Usage:
  # All variants on one dataset/horizon, no GA genome
  python -m src.exp2_ablation --dataset ETTh1 --pred_len 96 --structure dmamba

  # Include CfC variant + load GA genome from Exp 1
  python -m src.exp2_ablation --dataset ETTh1 --pred_len 96 \\
      --include_extended \\
      --ga_genome exp1_results/ETTh1_H96_dmamba/ts_candidate_1.pt

  # All horizons
  python -m src.exp2_ablation --dataset ETTh1 --all_horizons --structure dmamba
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
from src.train import _build_ts_model, train_ts_model, evaluate_ts, count_params

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HORIZONS = [96, 192, 336, 720]

# Operator family → representative class ID
ABLATION_VARIANTS = {
    "ALL_SSM":  5,   # Rec-1
    "ALL_CONV": 7,   # GConv-1
    "ALL_ATTN": 1,   # SA-1
    "ALL_FFN":  9,   # GMemless
}
# CfC added only when include_extended=True
CFC_VARIANT = {"ALL_CFC": 19}  # Rec-4


# =============================================================================
# Genome helpers
# =============================================================================

def make_uniform_genome(class_id: int, num_layers: int,
                        class_pool: dict) -> Genome:
    """Genome where every layer uses the same LIV class (no parameter sharing)."""
    cat = class_pool[class_id].category
    fg_mask = VALID_FG_MASKS[cat][0]   # first valid mask (usually 0 = no sharing)
    layers = [
        LayerGene(
            liv_class=class_id,
            feat_share_group=i,    # each layer in its own group
            feat_share_strategy=1, # no featurizer sharing
            fg_share_group=i,
            fg_share_strategy=fg_mask,
        )
        for i in range(num_layers)
    ]
    genome = Genome(layers=layers)
    repair(genome, class_pool)
    return genome


def load_genome_from_checkpoint(path: str, num_layers: int) -> Genome:
    """Load genome from a ts_candidate_*.pt checkpoint saved by post_ts_train."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if 'genome' in ckpt:
        return Genome.from_flat(ckpt['genome'], num_layers)
    if 'layer_classes' in ckpt:
        # Fallback: reconstruct from class list (sharing info lost)
        classes = ckpt['layer_classes']
        layers = [LayerGene(c, i, 1, i, 0) for i, c in enumerate(classes)]
        return Genome(layers=layers)
    raise ValueError(f"Cannot extract genome from {path}")


# =============================================================================
# Single-variant runner
# =============================================================================

def run_variant(
    variant_name: str,
    genome: Genome,
    class_pool: dict,
    train_dl, val_dl, test_dl,
    args,
) -> dict:
    """Train and evaluate one ablation variant. Returns result dict."""
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
    log.info(f"[{variant_name}] val={val_mse:.4f}  test_MSE={test_mse:.4f}  MAE={test_mae:.4f}")

    return {
        "variant": variant_name,
        "layer_classes": classes,
        "n_params": n_params,
        "val_mse": val_mse,
        "test_mse": test_mse,
        "test_mae": test_mae,
    }


# =============================================================================
# Per-horizon ablation
# =============================================================================

def run_ablation(
    dataset: str,
    pred_len: int,
    structure: str,
    include_extended: bool,
    ga_genome_path: str,
    h96_genome_path: str,
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
    """Run all ablation variants for one (dataset, horizon, structure)."""
    pool = build_class_pool(include_extended)

    # Build args namespace reused by _build_ts_model + train_ts_model
    args = make_args(
        dataset=dataset, pred_len=pred_len, structure=structure,
        out_dir=out_dir, dim=dim, num_layers=num_layers, seq_len=seq_len,
        lr=lr, batch_size=batch_size, num_workers=num_workers, amp=amp,
        device=device, full_train_steps=train_steps,
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

    # Build variant list
    variants = dict(ABLATION_VARIANTS)
    if include_extended:
        variants.update(CFC_VARIANT)

    results = []
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for name, class_id in variants.items():
        if class_id not in pool:
            log.warning(f"Class {class_id} not in pool, skipping {name}")
            continue
        genome = make_uniform_genome(class_id, num_layers, pool)
        r = run_variant(name, genome, pool, train_dl, val_dl, test_dl, args)
        r.update({"dataset": dataset, "pred_len": pred_len, "structure": structure})
        results.append(r)

    # GA-found MIXED variant (best genome for this specific horizon from Exp 1)
    if ga_genome_path:
        try:
            genome = load_genome_from_checkpoint(ga_genome_path, num_layers)
            repair(genome, pool)
            r = run_variant("MIXED_GA", genome, pool, train_dl, val_dl, test_dl, args)
            r.update({"dataset": dataset, "pred_len": pred_len, "structure": structure})
            results.append(r)
        except Exception as e:
            log.warning(f"Could not load GA genome from {ga_genome_path}: {e}")

    # H=96 best genome retrained at current horizon — tests "does the short-horizon
    # architecture generalise to longer horizons, or does SSM win at H=336/720?"
    if h96_genome_path:
        try:
            genome = load_genome_from_checkpoint(h96_genome_path, num_layers)
            repair(genome, pool)
            r = run_variant("H96_FIXED", genome, pool, train_dl, val_dl, test_dl, args)
            r.update({"dataset": dataset, "pred_len": pred_len, "structure": structure})
            results.append(r)
        except Exception as e:
            log.warning(f"Could not load H96 genome from {h96_genome_path}: {e}")

    # Save
    out_path = Path(out_dir) / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Ablation results saved to {out_path}")

    return results


# =============================================================================
# CLI
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser(description="Exp 2: Operator Ablation")
    p.add_argument("--dataset", type=str, default="ETTh1")
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--all_horizons", action="store_true",
                   help="Run all 4 horizons: 96, 192, 336, 720")
    p.add_argument("--structure", type=str, default="dmamba",
                   choices=["dmamba", "smamba", "itransformer"])
    p.add_argument("--include_extended", action="store_true",
                   help="Add ALL_CFC variant (requires Rec-4 = class 19)")
    p.add_argument("--ga_genome", type=str, default=None,
                   help="Path to ts_candidate_*.pt from Exp 1 (adds MIXED_GA variant)")
    p.add_argument("--h96_genome", type=str, default=None,
                   help="Path to H=96 best checkpoint — retrained at all horizons to test "
                        "whether short-horizon architecture transfers (adds H96_FIXED variant)")
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--train_steps", type=int, default=10_000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="exp2_results")
    return p


def main():
    args = build_parser().parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    horizons = HORIZONS if args.all_horizons else [args.pred_len]
    all_results = []

    for pred_len in horizons:
        run_dir = str(Path(args.out_dir) / f"{args.dataset}_H{pred_len}_{args.structure}")
        results = run_ablation(
            dataset=args.dataset,
            pred_len=pred_len,
            structure=args.structure,
            include_extended=args.include_extended,
            ga_genome_path=args.ga_genome,
            h96_genome_path=args.h96_genome,
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

    # Global summary
    summary_path = Path(args.out_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print table
    print(f"\n{'Variant':<14} {'H':>4} {'test_MSE':>10} {'test_MAE':>10} {'params':>10}")
    print("-" * 55)
    for r in sorted(all_results, key=lambda x: (x["pred_len"], x["test_mse"])):
        print(f"{r['variant']:<14} {r['pred_len']:>4} "
              f"{r['test_mse']:>10.4f} {r['test_mae']:>10.4f} {r['n_params']:>10,}")


if __name__ == "__main__":
    main()