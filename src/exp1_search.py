"""Experiment 1: Architecture Search (main result).

Runs NSGA-II over all structural approaches (dmamba, smamba, itransformer)
on a given dataset and horizon, then full-trains the Pareto-optimal candidates.

This script is a thin orchestrator — it delegates to the infrastructure
already in src/train.py (evolve_with_seeds, post_ts_train, _make_ts_quality_fn,
_build_ts_model) and core/nsga.py.

Usage:
  # Single search: one structure, one horizon
  python -m src.exp1_search --dataset ETTh1 --pred_len 96 --structure dmamba

  # Full Exp 1: all 3 structures, all horizons
  python -m src.exp1_search --dataset ETTh1 --all_horizons --all_structures

  # Dry run (uses param count as quality proxy, no training)
  python -m src.exp1_search --dataset ETTh1 --pred_len 96 --all_structures --dry_run

Output layout (--out_dir exp1_results/):
  exp1_results/
    ETTh1_H96_dmamba/
      pareto_front.json        # Pareto-optimal genomes + fitness
      ts_candidate_1.pt        # trained model checkpoint
      ...
    ETTh1_H96_smamba/
    ETTh1_H96_itransformer/
    summary.json               # MSE table across all structures/horizons
"""

import argparse
import json
import logging
import random
from argparse import Namespace
from pathlib import Path

import torch

from core.nsga import (
    build_class_pool,
    DEFAULT_POP_SIZE, DEFAULT_GENERATIONS, DEFAULT_MUTATION_PROB,
    DEFAULT_ELITISM, DEFAULT_CROSSOVER_POINTS, DEFAULT_TOURNAMENT_K,
)
from src.dataload import get_dataloader, get_dataset_info
from src.train import (
    evolve_with_seeds,
    post_ts_train,
    _make_ts_quality_fn,
    _get_seeds,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

STRUCTURES   = ["dmamba", "smamba", "itransformer"]
HORIZONS     = [96, 192, 336, 720]
ALL_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Electricity", "Traffic", "Exchange"]


# =============================================================================
# Args namespace builder
# =============================================================================

def make_args(
    dataset: str,
    pred_len: int,
    structure: str,
    out_dir: str,
    # model
    dim: int = 256,
    num_layers: int = 4,
    seq_len: int = 96,
    # evolution
    pop_size: int = DEFAULT_POP_SIZE,
    generations: int = DEFAULT_GENERATIONS,
    evolution_steps: int = 500,
    mutation_prob: float = DEFAULT_MUTATION_PROB,
    elitism: int = DEFAULT_ELITISM,
    crossover_points: int = DEFAULT_CROSSOVER_POINTS,
    tournament_k: int = DEFAULT_TOURNAMENT_K,
    include_extended: bool = False,
    measure_latency: bool = False,
    # full training
    full_train_steps: int = 10_000,
    top_k: int = 8,
    # training hyper
    lr: float = 1e-3,
    batch_size: int = 32,
    num_workers: int = 0,
    amp: bool = False,
    # misc
    seed: int = 42,
    device: str = None,
    dry_run: bool = False,
    no_seeds: bool = False,
) -> Namespace:
    """Construct an args Namespace compatible with src/train.py helpers."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return Namespace(
        # dataset / structure
        ts_dataset=dataset,
        ts_pred_len=pred_len,
        ts_seq_len=seq_len,
        ts_structure=structure,
        ts_batch_size=batch_size,
        ts_num_workers=num_workers,
        # model
        dim=dim,
        num_layers=num_layers,
        seq_len=seq_len,          # used by FitnessEvaluator (KV-cache estimate)
        num_heads=min(8, dim // 32) or 1,
        # evolution
        pop_size=pop_size,
        generations=generations,
        evolution_steps=evolution_steps,
        mutation_prob=mutation_prob,
        elitism=elitism,
        crossover_points=crossover_points,
        tournament_k=tournament_k,
        include_extended=include_extended,
        measure_latency=measure_latency,
        # full training
        full_train_steps=full_train_steps,
        top_k=top_k,
        # training hyper
        lr=lr,
        beta1=0.9,
        beta2=0.95,
        weight_decay=0.1,
        grad_clip=1.0,
        warmup_steps=min(500, evolution_steps // 10),
        train_steps=full_train_steps,
        amp=amp,
        # misc
        seed=seed,
        device=device,
        dry_run=dry_run,
        no_seeds=no_seeds,
        out_dir=out_dir,
    )


# =============================================================================
# Per-structure search
# =============================================================================

def run_structure_search(args: Namespace) -> list:
    """Run NSGA-II + full training for one (structure, dataset, horizon).

    Returns list of result dicts for the trained candidates.
    """
    dataset  = args.ts_dataset
    pred_len = args.ts_pred_len
    struct   = args.ts_structure

    log.info(f"\n{'='*60}")
    log.info(f"Exp 1 Search: {struct}  {dataset}  H={pred_len}")
    log.info(f"{'='*60}")

    # ── Data ──────────────────────────────────────────────────────────────────
    info = get_dataset_info(dataset)
    if not info.get("exists"):
        raise FileNotFoundError(
            f"Dataset '{dataset}' not found. Run ./download_data.sh")
    args.n_variates = info["n_variates"]

    # seq_len for dmamba processes temporal tokens (L); smamba/itransformer
    # process variate tokens (C). FitnessEvaluator uses args.seq_len for
    # KV-cache estimation, so set it to match the actual token sequence.
    if struct == "dmamba":
        args.seq_len = args.ts_seq_len
    else:
        args.seq_len = args.n_variates

    train_dl, val_dl, test_dl, _ = get_dataloader(
        dataset=dataset, seq_len=args.ts_seq_len, pred_len=pred_len,
        batch_size=args.ts_batch_size, num_workers=args.ts_num_workers,
    )

    # ── Evolution ─────────────────────────────────────────────────────────────
    pool = build_class_pool(args.include_extended)
    causal = (struct == "dmamba")

    qfn = _make_ts_quality_fn(args, train_dl, val_dl, pool)
    seeds = None if args.no_seeds else _get_seeds(args)

    population, _ = evolve_with_seeds(
        args, qfn, seeds, causal=causal,
    )

    # ── Save Pareto front ─────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pareto = [
        {
            "rank": ind.rank,
            "quality": ind.fitness.quality,
            "param_count": ind.fitness.param_count,
            "kv_cache_size": ind.fitness.kv_cache_size,
            "latency_ms": ind.fitness.latency_ms,
            "layer_classes": [g.liv_class for g in ind.genome.layers],
        }
        for ind in population if ind.rank == 0
    ]
    with open(out_dir / "pareto_front.json", "w") as f:
        json.dump(pareto, f, indent=2)
    log.info(f"Pareto front ({len(pareto)} individuals) saved to {out_dir}/pareto_front.json")

    # ── Full training of top-K ────────────────────────────────────────────────
    results = post_ts_train(population, pool, train_dl, val_dl, test_dl, args)

    # ── Summarise ─────────────────────────────────────────────────────────────
    summary = []
    for rank, (ind, val_mse, test_mse, test_mae) in enumerate(results):
        summary.append({
            "rank_in_candidates": rank + 1,
            "structure": struct,
            "dataset": dataset,
            "pred_len": pred_len,
            "layer_classes": [g.liv_class for g in ind.genome.layers],
            "param_count": ind.fitness.param_count,
            "val_mse": val_mse,
            "test_mse": test_mse,
            "test_mae": test_mae,
        })

    with open(out_dir / "candidates_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# =============================================================================
# CLI
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser(description="Exp 1: Architecture Search")

    # Target
    p.add_argument("--dataset", type=str, default="ETTh1")
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--structure", type=str, default="itransformer",
                   choices=STRUCTURES)
    p.add_argument("--all_structures", action="store_true",
                   help="Search over dmamba, smamba, itransformer")
    p.add_argument("--all_horizons", action="store_true",
                   help="Run horizons 96, 192, 336, 720")

    # Model
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)

    # Evolution
    p.add_argument("--pop_size", type=int, default=DEFAULT_POP_SIZE)
    p.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    p.add_argument("--evolution_steps", type=int, default=500,
                   help="Training steps per candidate during search")
    p.add_argument("--include_extended", action="store_true",
                   help="Include Rec-3/Rec-4 (CfC) in search pool")
    p.add_argument("--measure_latency", action="store_true")
    p.add_argument("--no_seeds", action="store_true",
                   help="Start from random genomes (no hand-crafted seeds)")

    # Full training
    p.add_argument("--full_train_steps", type=int, default=10_000)
    p.add_argument("--top_k", type=int, default=8)

    # Training hyper
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--amp", action="store_true")

    # System
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="exp1_results")
    p.add_argument("--dry_run", action="store_true",
                   help="Skip training; use param count as quality proxy")

    return p


def main():
    cli = build_parser().parse_args()

    torch.manual_seed(cli.seed)
    random.seed(cli.seed)

    structures = STRUCTURES if cli.all_structures else [cli.structure]
    horizons   = HORIZONS   if cli.all_horizons   else [cli.pred_len]

    all_summaries = []

    for pred_len in horizons:
        for structure in structures:
            run_out = str(Path(cli.out_dir) / f"{cli.dataset}_H{pred_len}_{structure}")
            args = make_args(
                dataset=cli.dataset,
                pred_len=pred_len,
                structure=structure,
                out_dir=run_out,
                dim=cli.dim,
                num_layers=cli.num_layers,
                seq_len=cli.seq_len,
                pop_size=cli.pop_size,
                generations=cli.generations,
                evolution_steps=cli.evolution_steps,
                include_extended=cli.include_extended,
                measure_latency=cli.measure_latency,
                full_train_steps=cli.full_train_steps,
                top_k=cli.top_k,
                lr=cli.lr,
                batch_size=cli.batch_size,
                num_workers=cli.num_workers,
                amp=cli.amp,
                seed=cli.seed,
                device=cli.device,
                dry_run=cli.dry_run,
                no_seeds=cli.no_seeds,
            )

            try:
                summary = run_structure_search(args)
                all_summaries.extend(summary)
            except Exception as e:
                log.error(f"Search failed for {structure}/{cli.dataset}/H{pred_len}: {e}")
                raise

    # ── Global summary table ──────────────────────────────────────────────────
    summary_path = Path(cli.out_dir) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    log.info(f"\nGlobal summary saved to {summary_path}")

    # Print table
    print(f"\n{'Structure':<15} {'H':>4} {'Rank':>4} "
          f"{'test_MSE':>10} {'test_MAE':>10} {'params':>10}  Classes")
    print("-" * 80)
    for r in sorted(all_summaries, key=lambda x: (x["pred_len"], x["structure"])):
        cls_str = str(r["layer_classes"])
        print(f"{r['structure']:<15} {r['pred_len']:>4} {r['rank_in_candidates']:>4} "
              f"{r['test_mse']:>10.4f} {r['test_mae']:>10.4f} "
              f"{r['param_count']:>10,}  {cls_str}")


if __name__ == "__main__":
    main()