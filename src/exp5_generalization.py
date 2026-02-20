"""Experiment 5: Cross-Dataset Generalization.

Search on a source dataset (default ETTh1), then evaluate how well the
discovered architecture generalises to other datasets — two transfer modes:

  zero_shot           — load the checkpoint's trained weights directly and
                        evaluate on a target dataset with the *same number
                        of variates* (e.g. ETTh1 → ETTh2, ETTm1, ETTm2).
                        No retraining; tests pure weight transfer.

  architecture_transfer — keep only the discovered genome (architecture), train
                          fresh weights on the target dataset, and compare the
                          test MSE against a default uniform Rec-1 baseline
                          trained on the same target dataset.

Expected outcome: architecture transfer should outperform the default Rec-1
baseline on most target datasets, demonstrating that the searched architecture
captures dataset-agnostic inductive biases.

Usage:
  # Zero-shot evaluation on all same-variate ETT datasets
  python -m src.exp5_generalization \\
      --source_checkpoint exp1_results/ETTh1_H96_itransformer/tidar_ts_candidate_1.pt \\
      --pred_len 96 --zero_shot

  # Architecture transfer: train from scratch on all datasets
  python -m src.exp5_generalization \\
      --source_checkpoint exp1_results/ETTh1_H96_itransformer/ts_candidate_1.pt \\
      --pred_len 96 --all_targets --transfer_steps 10000

  # Both modes, all horizons
  python -m src.exp5_generalization \\
      --source_checkpoint exp1_results/ETTh1_H96_itransformer/ts_candidate_1.pt \\
      --all_horizons --all_targets --zero_shot
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from core.nsga import build_class_pool, Genome, LayerGene, repair
from src.dataload import get_dataloader, get_dataset_info
from src.exp1_search import make_args
from src.exp2_ablation import make_uniform_genome, load_genome_from_checkpoint
from src.train import (
    _build_ts_model,
    train_ts_model,
    evaluate_ts,
    count_params,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HORIZONS     = [96, 192, 336, 720]
ALL_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Electricity", "Traffic", "Exchange"]

# Datasets that share the same variate count as ETTh1/ETTh2/ETTm1/ETTm2 (7 variates)
ETT_FAMILY = {"ETTh1", "ETTh2", "ETTm1", "ETTm2"}


# =============================================================================
# Genome / checkpoint helpers
# =============================================================================

def load_checkpoint(path: str, num_layers: int) -> tuple:
    """Load a ts_candidate_*.pt checkpoint.

    Returns (genome, model_state_dict_or_None, structure_str_or_None).
    model_state_dict is None for genome-only checkpoints.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    genome = load_genome_from_checkpoint(path, num_layers)

    model_state = ckpt.get("model_state_dict", None)
    structure   = ckpt.get("structure", None)   # saved by post_ts_train if present

    return genome, model_state, structure


# =============================================================================
# Zero-shot evaluation
# =============================================================================

def run_zero_shot(
    source_checkpoint: str,
    source_dataset: str,
    target_dataset: str,
    pred_len: int,
    seq_len: int,
    dim: int,
    num_layers: int,
    structure: str,
    batch_size: int,
    num_workers: int,
    device: str,
    out_dir: str,
) -> dict:
    """Evaluate source checkpoint on target dataset without retraining.

    Only valid when source and target share the same number of variates
    (e.g. ETTh1 → ETTh2, ETTm1, ETTm2).
    """
    src_info = get_dataset_info(source_dataset)
    tgt_info = get_dataset_info(target_dataset)

    if not src_info.get("exists"):
        raise FileNotFoundError(f"Source dataset '{source_dataset}' not found.")
    if not tgt_info.get("exists"):
        raise FileNotFoundError(f"Target dataset '{target_dataset}' not found.")

    if src_info["n_variates"] != tgt_info["n_variates"]:
        raise ValueError(
            f"Zero-shot requires matching variates: "
            f"{source_dataset}={src_info['n_variates']} vs "
            f"{target_dataset}={tgt_info['n_variates']}"
        )

    pool = build_class_pool(include_extended=True)
    genome, model_state, ckpt_structure = load_checkpoint(source_checkpoint, num_layers)
    repair(genome, pool)

    if model_state is None:
        raise ValueError(
            f"Checkpoint {source_checkpoint} has no model_state_dict — "
            "cannot do zero-shot (no trained weights). Use architecture_transfer instead."
        )

    effective_structure = ckpt_structure or structure
    args = make_args(
        dataset=target_dataset, pred_len=pred_len, structure=effective_structure,
        out_dir=out_dir, dim=dim, num_layers=num_layers, seq_len=seq_len,
        device=device, include_extended=True,
    )
    args.n_variates = tgt_info["n_variates"]

    model = _build_ts_model(genome, pool, args)
    try:
        model.load_state_dict(model_state)
        log.info("Zero-shot: loaded weights from checkpoint (strict=True)")
    except RuntimeError:
        model.load_state_dict(model_state, strict=False)
        log.warning("Zero-shot: loaded weights with strict=False (partial match)")

    _, _, test_dl, _ = get_dataloader(
        dataset=target_dataset, seq_len=seq_len, pred_len=pred_len,
        batch_size=batch_size, num_workers=num_workers,
    )

    model = model.to(device)
    test_mse, test_mae = evaluate_ts(model, test_dl, args)
    n_params = count_params(model)

    log.info(
        f"[zero_shot] {source_dataset}→{target_dataset} H={pred_len}  "
        f"MSE={test_mse:.4f}  MAE={test_mae:.4f}"
    )

    return {
        "transfer_type":  "zero_shot",
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "pred_len":       pred_len,
        "structure":      effective_structure,
        "layer_classes":  [g.liv_class for g in genome.layers],
        "n_params":       n_params,
        "test_mse":       test_mse,
        "test_mae":       test_mae,
    }


# =============================================================================
# Architecture transfer
# =============================================================================

def run_architecture_transfer(
    source_checkpoint: str,
    source_dataset: str,
    target_dataset: str,
    pred_len: int,
    seq_len: int,
    dim: int,
    num_layers: int,
    structure: str,
    transfer_steps: int,
    batch_size: int,
    lr: float,
    amp: bool,
    num_workers: int,
    device: str,
    out_dir: str,
) -> list:
    """Train from scratch using searched architecture on target dataset.

    Returns two result dicts: the searched genome variant and a Rec-1 baseline.
    """
    tgt_info = get_dataset_info(target_dataset)
    if not tgt_info.get("exists"):
        raise FileNotFoundError(f"Target dataset '{target_dataset}' not found.")

    pool = build_class_pool(include_extended=True)
    genome, _, ckpt_structure = load_checkpoint(source_checkpoint, num_layers)
    repair(genome, pool)

    effective_structure = ckpt_structure or structure
    args = make_args(
        dataset=target_dataset, pred_len=pred_len, structure=effective_structure,
        out_dir=out_dir, dim=dim, num_layers=num_layers, seq_len=seq_len,
        lr=lr, batch_size=batch_size, num_workers=num_workers, amp=amp,
        device=device, full_train_steps=transfer_steps, include_extended=True,
    )
    args.train_steps  = transfer_steps
    args.n_variates   = tgt_info["n_variates"]

    train_dl, val_dl, test_dl, _ = get_dataloader(
        dataset=target_dataset, seq_len=seq_len, pred_len=pred_len,
        batch_size=batch_size, num_workers=num_workers,
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    results = []

    # ── Searched architecture ─────────────────────────────────────────────────
    log.info(f"[arch_transfer] Training searched genome on {target_dataset} H={pred_len} ...")
    model = _build_ts_model(genome, pool, args)
    log.info(f"  params={count_params(model):,}  classes={[g.liv_class for g in genome.layers]}")
    val_mse = train_ts_model(model, train_dl, val_dl, args,
                             steps=transfer_steps, prefix="[searched] ")
    model   = model.to(device)
    test_mse, test_mae = evaluate_ts(model, test_dl, args)
    log.info(f"  val={val_mse:.4f}  test_MSE={test_mse:.4f}  MAE={test_mae:.4f}")

    results.append({
        "transfer_type":  "architecture_transfer",
        "variant":        "searched",
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "pred_len":       pred_len,
        "structure":      effective_structure,
        "layer_classes":  [g.liv_class for g in genome.layers],
        "n_params":       count_params(model),
        "val_mse":        val_mse,
        "test_mse":       test_mse,
        "test_mae":       test_mae,
    })

    # ── Rec-1 baseline (same structure, uniform genome) ───────────────────────
    log.info(f"[arch_transfer] Training Rec-1 baseline on {target_dataset} H={pred_len} ...")
    base_pool   = build_class_pool(include_extended=False)
    base_genome = make_uniform_genome(5, num_layers, base_pool)

    base_args = make_args(
        dataset=target_dataset, pred_len=pred_len, structure=effective_structure,
        out_dir=out_dir, dim=dim, num_layers=num_layers, seq_len=seq_len,
        lr=lr, batch_size=batch_size, num_workers=num_workers, amp=amp,
        device=device, full_train_steps=transfer_steps,
    )
    base_args.train_steps = transfer_steps
    base_args.n_variates  = tgt_info["n_variates"]

    base_model = _build_ts_model(base_genome, base_pool, base_args)
    log.info(f"  params={count_params(base_model):,}  classes=[5,5,5,5] (Rec-1 uniform)")
    base_val = train_ts_model(base_model, train_dl, val_dl, base_args,
                              steps=transfer_steps, prefix="[rec1_base] ")
    base_model  = base_model.to(device)
    b_test_mse, b_test_mae = evaluate_ts(base_model, test_dl, base_args)
    log.info(f"  val={base_val:.4f}  test_MSE={b_test_mse:.4f}  MAE={b_test_mae:.4f}")

    results.append({
        "transfer_type":  "architecture_transfer",
        "variant":        "rec1_baseline",
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "pred_len":       pred_len,
        "structure":      effective_structure,
        "layer_classes":  [5] * num_layers,
        "n_params":       count_params(base_model),
        "val_mse":        base_val,
        "test_mse":       b_test_mse,
        "test_mae":       b_test_mae,
    })

    # Log delta
    delta  = test_mse - b_test_mse
    winner = "searched" if delta < 0 else "rec1_baseline"
    log.info(
        f"[arch_transfer] {source_dataset}→{target_dataset} H={pred_len}  "
        f"ΔMSE(searched-baseline)={delta:+.4f}  winner={winner}"
    )

    return results


# =============================================================================
# Per-horizon runner
# =============================================================================

def run_generalization(
    source_checkpoint: str,
    source_dataset: str,
    target_datasets: list,
    pred_len: int,
    seq_len: int,
    dim: int,
    num_layers: int,
    structure: str,
    zero_shot: bool,
    transfer_steps: int,
    batch_size: int,
    lr: float,
    amp: bool,
    num_workers: int,
    device: str,
    out_dir: str,
) -> list:
    """Run all transfer modes for one pred_len over all target datasets."""
    all_results = []
    src_info = get_dataset_info(source_dataset)
    src_variates = src_info.get("n_variates", 0)

    for target in target_datasets:
        if target == source_dataset:
            log.info(f"Skipping source dataset {source_dataset} as target.")
            continue

        run_dir = str(Path(out_dir) / f"{source_dataset}_to_{target}_H{pred_len}")
        Path(run_dir).mkdir(parents=True, exist_ok=True)

        tgt_info = get_dataset_info(target)
        if not tgt_info.get("exists"):
            log.warning(f"Dataset '{target}' not found, skipping.")
            continue

        # Zero-shot: only when variate counts match
        if zero_shot:
            tgt_variates = tgt_info.get("n_variates", -1)
            if tgt_variates == src_variates:
                try:
                    r = run_zero_shot(
                        source_checkpoint=source_checkpoint,
                        source_dataset=source_dataset,
                        target_dataset=target,
                        pred_len=pred_len,
                        seq_len=seq_len,
                        dim=dim,
                        num_layers=num_layers,
                        structure=structure,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        device=device,
                        out_dir=run_dir,
                    )
                    all_results.append(r)
                    with open(Path(run_dir) / "zero_shot.json", "w") as f:
                        json.dump(r, f, indent=2)
                except Exception as e:
                    log.error(f"Zero-shot failed for {target}: {e}")
            else:
                log.info(
                    f"Skipping zero-shot {source_dataset}→{target}: "
                    f"variate mismatch ({src_variates} vs {tgt_variates})."
                )

        # Architecture transfer: always run
        if transfer_steps > 0:
            try:
                rs = run_architecture_transfer(
                    source_checkpoint=source_checkpoint,
                    source_dataset=source_dataset,
                    target_dataset=target,
                    pred_len=pred_len,
                    seq_len=seq_len,
                    dim=dim,
                    num_layers=num_layers,
                    structure=structure,
                    transfer_steps=transfer_steps,
                    batch_size=batch_size,
                    lr=lr,
                    amp=amp,
                    num_workers=num_workers,
                    device=device,
                    out_dir=run_dir,
                )
                all_results.extend(rs)
                with open(Path(run_dir) / "arch_transfer.json", "w") as f:
                    json.dump(rs, f, indent=2)
            except Exception as e:
                log.error(f"Architecture transfer failed for {target}: {e}")

    return all_results


# =============================================================================
# CLI
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser(description="Exp 5: Cross-Dataset Generalization")
    p.add_argument("--source_checkpoint", type=str, required=True,
                   help="Path to ts_candidate_*.pt from Exp 1 (source genome + weights)")
    p.add_argument("--source_dataset", type=str, default="ETTh1",
                   help="Dataset used to train the source checkpoint")
    # Target selection
    p.add_argument("--target_dataset", type=str, default=None,
                   help="Single target dataset (default: all non-source ETT datasets)")
    p.add_argument("--all_targets", action="store_true",
                   help="Transfer to all 7 benchmark datasets")
    # Horizon
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--all_horizons", action="store_true",
                   help="Run all 4 horizons: 96, 192, 336, 720")
    p.add_argument("--seq_len", type=int, default=96)
    # Model
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--structure", type=str, default="itransformer",
                   choices=["dmamba", "smamba", "itransformer"],
                   help="Fallback structure if not saved in checkpoint")
    # Transfer modes
    p.add_argument("--zero_shot", action="store_true",
                   help="Evaluate checkpoint weights directly (same-variate targets only)")
    p.add_argument("--transfer_steps", type=int, default=10_000,
                   help="Steps to train from scratch for architecture_transfer (0 = skip)")
    # Training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="exp5_results")
    return p


def main():
    args   = build_parser().parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Determine target datasets
    if args.all_targets:
        target_datasets = [d for d in ALL_DATASETS if d != args.source_dataset]
    elif args.target_dataset:
        target_datasets = [args.target_dataset]
    else:
        # Default: other ETT datasets (same variate family as ETTh1)
        target_datasets = [d for d in ETT_FAMILY if d != args.source_dataset]

    horizons    = HORIZONS if args.all_horizons else [args.pred_len]
    all_results = []

    for pred_len in horizons:
        results = run_generalization(
            source_checkpoint=args.source_checkpoint,
            source_dataset=args.source_dataset,
            target_datasets=target_datasets,
            pred_len=pred_len,
            seq_len=args.seq_len,
            dim=args.dim,
            num_layers=args.num_layers,
            structure=args.structure,
            zero_shot=args.zero_shot,
            transfer_steps=args.transfer_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            amp=args.amp,
            num_workers=args.num_workers,
            device=device,
            out_dir=str(Path(args.out_dir) / f"source_{args.source_dataset}_H{pred_len}"),
        )
        all_results.extend(results)

    # Global summary
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir) / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Print tables ──────────────────────────────────────────────────────────
    # Zero-shot table
    zs = [r for r in all_results if r["transfer_type"] == "zero_shot"]
    if zs:
        print(f"\n{'=== Zero-Shot Evaluation':}")
        print(f"{'Source→Target':<25} {'H':>4} {'MSE':>10} {'MAE':>10}")
        print("-" * 53)
        for r in sorted(zs, key=lambda x: (x["pred_len"], x["target_dataset"])):
            pair = f"{r['source_dataset']}→{r['target_dataset']}"
            print(f"{pair:<25} {r['pred_len']:>4} {r['test_mse']:>10.4f} {r['test_mae']:>10.4f}")

    # Architecture transfer table
    at = [r for r in all_results if r["transfer_type"] == "architecture_transfer"]
    if at:
        print(f"\n{'=== Architecture Transfer vs Rec-1 Baseline':}")
        print(f"{'Target':<12} {'H':>4} {'Searched_MSE':>13} {'Rec1_MSE':>10} {'ΔMSE':>8} {'Winner':>8}")
        print("-" * 60)
        rows: dict = {}
        for r in at:
            key = (r["target_dataset"], r["pred_len"])
            rows.setdefault(key, {})[r["variant"]] = r

        for (tgt, pred_len), variants in sorted(rows.items()):
            s = variants.get("searched", {})
            b = variants.get("rec1_baseline", {})
            if not s or not b:
                continue
            delta  = s["test_mse"] - b["test_mse"]
            winner = "searched" if delta < 0 else "rec1"
            print(f"{tgt:<12} {pred_len:>4} {s['test_mse']:>13.4f} {b['test_mse']:>10.4f} "
                  f"{delta:>+8.4f} {winner:>8}")

    log.info(f"Summary saved to {args.out_dir}/summary.json")


if __name__ == "__main__":
    main()