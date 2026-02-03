"""STAR Training: Evolution-guided architecture search with LIV operators.

Paper: STAR — Synthesis of Tailored Architectures (arXiv:2411.17800v1)

Training recipe from paper (Table A.1 / A.2):
  - AdamW: lr=8e-4, β1=0.9, β2=0.95, weight_decay=0.1
  - Cosine LR decay, 500-step warmup, grad_clip=1.0
  - Batch: 0.25M tokens/step, seq_len=4096
  - Evolution: 5000 steps/candidate, pop=16, gen=18
  - Post-evolution: top-8, 20000 steps (~5B tokens)

Usage:
    python train.py evolve --synthetic                  # quick test
    python train.py evolve --data_path tokens.pt        # real data
    python train.py train  --genome_path best.json ...  # train one genome
    python train.py both   --synthetic                  # evolve + train top-K
"""

import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from nsga import (
    build_class_pool, random_genome, repair,
    Genome, LayerGene, Individual, FitnessResult,
    GenomeModelBuilder, FitnessEvaluator,
    count_params, estimate_kv_cache,
    non_dominated_sort, crowding_distance,
    tournament_select, crossover, mutate,
    DEFAULT_POP_SIZE, DEFAULT_GENERATIONS, DEFAULT_MUTATION_PROB,
    DEFAULT_ELITISM, DEFAULT_CROSSOVER_POINTS, DEFAULT_TOURNAMENT_K,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================================
# Language Model Wrapper
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class STARLanguageModel(nn.Module):
    """Causal LM: embedding -> LIV backbone -> RMSNorm -> LM head (tied)."""

    def __init__(self, backbone, vocab_size, dim, tie_weights=True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.backbone = backbone
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        if tie_weights:
            self.head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, idx, targets=None):
        x = self.backbone(self.embed(idx))
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class STARVisionModel(nn.Module):
    """Vision model: patch_embed -> LIV backbone (causal=False) -> pool -> head.

    Build the backbone with causal=False:
        backbone = GenomeModelBuilder(pool, dim, causal=False).build(genome)
        model = STARVisionModel(backbone, num_classes=10, dim=64)
    """

    def __init__(self, backbone, num_classes, dim,
                 img_size=32, patch_size=4, in_channels=3):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size,
        )
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, dim) * 0.02
        )
        self.backbone = backbone
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        nn.init.normal_(self.patch_embed.weight, std=0.02)
        nn.init.zeros_(self.patch_embed.bias)

    def forward(self, imgs, targets=None):
        # [B, C, H, W] -> [B, num_patches, dim]
        x = self.patch_embed(imgs).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.backbone(x)
        # Global average pool -> classify
        x = x.mean(dim=1)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss


# ============================================================================
# Cosine LR with Linear Warmup (Paper Table A.1)
# ============================================================================

def get_lr(step, warmup, total, max_lr, min_lr_ratio=0.1):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total:
        return max_lr * min_lr_ratio
    t = (step - warmup) / (total - warmup)
    return max_lr * (min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * t)))


# ============================================================================
# Data Loading
# ============================================================================

class TokenDataset:
    """Simple random-access token dataset for causal LM training."""

    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    @classmethod
    def from_file(cls, path, seq_len):
        data = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(data, dict):
            data = data.get("input_ids", data.get("tokens", next(iter(data.values()))))
        return cls(torch.as_tensor(data, dtype=torch.long).flatten(), seq_len)

    @classmethod
    def synthetic(cls, num_tokens, vocab_size, seq_len):
        return cls(torch.randint(0, vocab_size, (num_tokens,)), seq_len)

    def get_batch(self, batch_size, device="cpu"):
        hi = len(self.tokens) - self.seq_len - 1
        ix = torch.randint(0, hi, (batch_size,))
        x = torch.stack([self.tokens[i : i + self.seq_len] for i in ix])
        y = torch.stack([self.tokens[i + 1 : i + self.seq_len + 1] for i in ix])
        return x.to(device), y.to(device)


def load_data(args):
    """Load train/val token data from file, HuggingFace, or synthetic."""
    if args.synthetic:
        n = args.synthetic_tokens or 10_000_000
        train = TokenDataset.synthetic(n, args.vocab_size, args.seq_len)
        val = TokenDataset.synthetic(n // 10, args.vocab_size, args.seq_len)
        log.info(f"Synthetic data: {n:,} train / {n // 10:,} val tokens")
        return train, val

    if args.data_path:
        train = TokenDataset.from_file(args.data_path, args.seq_len)
        if args.val_path:
            val = TokenDataset.from_file(args.val_path, args.seq_len)
        else:
            sp = int(len(train.tokens) * 0.9)
            val = TokenDataset(train.tokens[sp:], args.seq_len)
            train = TokenDataset(train.tokens[:sp], args.seq_len)
        log.info(f"File data: {len(train.tokens):,} train / {len(val.tokens):,} val")
        return train, val

    if args.hf_dataset:
        try:
            from datasets import load_dataset
            from transformers import AutoTokenizer
        except ImportError:
            raise RuntimeError("pip install datasets transformers")

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        args.vocab_size = len(tokenizer)
        log.info(f"Loading {args.hf_dataset} ({args.tokenizer}, vocab={args.vocab_size})")

        ds = load_dataset(args.hf_dataset, split=args.hf_split, streaming=True)
        target = args.hf_tokens or 50_000_000
        toks = []
        for ex in ds:
            text = ex.get("text", ex.get("content", ""))
            if text:
                toks.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
            if len(toks) >= target:
                break

        all_t = torch.tensor(toks[:target], dtype=torch.long)
        sp = int(len(all_t) * 0.9)
        train = TokenDataset(all_t[:sp], args.seq_len)
        val = TokenDataset(all_t[sp:], args.seq_len)
        log.info(f"HF data: {sp:,} train / {len(all_t) - sp:,} val tokens")
        return train, val

    raise ValueError("Specify --data_path, --hf_dataset, or --synthetic")


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_model(model, train_data, val_data, args, steps, prefix=""):
    """Train for `steps` steps. Returns validation perplexity."""
    device = args.device
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    # Gradient accumulation: total batch = batch_tokens / seq_len sequences
    total_bs = max(1, args.batch_tokens // args.seq_len)
    micro_bs = min(total_bs, args.micro_batch)
    grad_accum = max(1, total_bs // micro_bs)

    use_amp = args.amp and device != "cpu"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    log_every = max(1, steps // 20)

    model.train()
    running = 0.0

    for step in range(steps):
        lr = get_lr(step, args.warmup_steps, steps, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        step_loss = 0.0

        for _ in range(grad_accum):
            x, y = train_data.get_batch(micro_bs, device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                _, loss = model(x, y)
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            step_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running += step_loss
        if (step + 1) % log_every == 0:
            log.info(f"{prefix}step {step+1}/{steps}  lr={lr:.2e}  loss={running/log_every:.4f}")
            running = 0.0

    ppl = evaluate_ppl(model, val_data, args)
    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ppl


@torch.no_grad()
def evaluate_ppl(model, val_data, args, n_batches=None):
    """Validation perplexity."""
    n_batches = n_batches or args.eval_batches
    device = next(model.parameters()).device
    micro_bs = min(max(1, args.batch_tokens // args.seq_len), args.micro_batch)

    model.eval()
    total_loss, total_tok = 0.0, 0
    for _ in range(n_batches):
        x, y = val_data.get_batch(micro_bs, device)
        with torch.amp.autocast("cuda", enabled=args.amp and device.type != "cpu"
                                 if hasattr(device, "type") else args.amp and str(device) != "cpu"):
            _, loss = model(x, y)
        total_loss += loss.item() * y.numel()
        total_tok += y.numel()

    model.train()
    avg = total_loss / max(total_tok, 1)
    return math.exp(min(avg, 20.0))


# ============================================================================
# Seed Genomes — Known-Good Architectures (Paper Section 4.1)
# ============================================================================

def seed_transformer_pp(n):
    """Transformer++: alternating SA-1 (attention) and GMemless (SwiGLU FFN)."""
    return Genome([LayerGene(1 if i % 2 == 0 else 9, i, 1, i, 0) for i in range(n)])


def seed_striped_mamba(n):
    """StripedMamba hybrid: [SA-1, Rec-1, GMemless] repeating."""
    pat = [1, 5, 9]
    return Genome([LayerGene(pat[i % 3], i, 1, i, 0) for i in range(n)])


def seed_hybrid_conv(n):
    """All-families hybrid: [SA-1, GConv-1, Rec-1, GMemless] repeating."""
    pat = [1, 7, 5, 9]
    return Genome([LayerGene(pat[i % 4], i, 1, i, 0) for i in range(n)])


# ============================================================================
# NSGA-II Evolution with Seed Support & Checkpointing
# ============================================================================

def evolve_with_seeds(args, quality_fn, seed_genomes=None):
    """NSGA-II evolution loop with seed genomes, logging, and checkpoints.

    Uses all genetic operators from nsga.py; adds seed initialization and
    per-generation checkpointing.
    """
    rng = random.Random(args.seed)
    class_pool = build_class_pool(getattr(args, "include_extended", False))
    evaluator = FitnessEvaluator(
        class_pool, args.dim, quality_fn, seq_len=args.seq_len,
    )
    n_layers = args.num_layers
    pop_size = args.pop_size

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Initial population: seeds + random ---
    population = []
    if seed_genomes:
        for g in seed_genomes:
            repair(g, class_pool)
            log.info(f"Evaluating seed: {[l.liv_class for l in g.layers[:6]]}...")
            f = evaluator.evaluate(g)
            population.append(Individual(genome=g, fitness=f))
            log.info(f"  quality={f.quality:.2f}  params={f.param_count:,}")

    while len(population) < pop_size:
        g = random_genome(n_layers, class_pool, rng)
        repair(g, class_pool)
        log.info(f"Evaluating random genome {len(population)+1}/{pop_size}...")
        f = evaluator.evaluate(g)
        population.append(Individual(genome=g, fitness=f))
        log.info(f"  quality={f.quality:.2f}  params={f.param_count:,}")

    # --- Evolution loop (Paper: 18 generations, pop=16) ---
    for gen in range(args.generations):
        t0 = time.time()
        log.info(f"\n{'='*60}\nGeneration {gen+1}/{args.generations}\n{'='*60}")

        # 1. Non-dominated sort + crowding distance
        fronts = non_dominated_sort(population)
        for front in fronts:
            crowding_distance(population, front)
        population.sort(key=lambda ind: (ind.rank, -ind.crowding_distance))

        # Log Pareto front
        pf = [p for p in population if p.rank == 0]
        log.info(f"Pareto front: {len(pf)} individuals")
        for i, ind in enumerate(pf):
            log.info(f"  [{i}] q={ind.fitness.quality:.2f} "
                     f"p={ind.fitness.param_count:,} c={ind.fitness.kv_cache_size:,}")

        # Checkpoint
        _save_checkpoint(out_dir / f"gen_{gen+1:02d}.json", population, gen + 1)

        # 2. Create offspring
        offspring = []

        # Elitism (Paper: 2 elite individuals)
        for i in range(min(args.elitism, len(population))):
            offspring.append(Individual(
                genome=population[i].genome.copy(), fitness=population[i].fitness,
            ))

        # Crossover + mutation (Paper: 2-point crossover, 10% mutation)
        child_idx = len(offspring)
        while len(offspring) < pop_size:
            p1 = tournament_select(population, args.tournament_k, rng)
            p2 = tournament_select(population, args.tournament_k, rng)
            c1, c2 = crossover(p1.genome, p2.genome, args.crossover_points, rng)

            mutate(c1, args.mutation_prob, class_pool, rng)
            mutate(c2, args.mutation_prob, class_pool, rng)
            repair(c1, class_pool)
            repair(c2, class_pool)

            log.info(f"  Offspring {len(offspring)+1}/{pop_size}...")
            f1 = evaluator.evaluate(c1)
            offspring.append(Individual(genome=c1, fitness=f1))
            log.info(f"    q={f1.quality:.2f} p={f1.param_count:,}")

            if len(offspring) < pop_size:
                f2 = evaluator.evaluate(c2)
                offspring.append(Individual(genome=c2, fitness=f2))
                log.info(f"    q={f2.quality:.2f} p={f2.param_count:,}")

        population = offspring[:pop_size]
        log.info(f"Generation {gen+1} done ({time.time()-t0:.0f}s)")

    # Final sort
    fronts = non_dominated_sort(population)
    for front in fronts:
        crowding_distance(population, front)
    population.sort(key=lambda ind: (ind.rank, -ind.crowding_distance))

    _save_checkpoint(out_dir / "final.json", population, args.generations)
    return population, class_pool


def _save_checkpoint(path, population, generation):
    data = {
        "generation": generation,
        "individuals": [
            {
                "genome": ind.genome.flatten(),
                "fitness": {
                    "quality": ind.fitness.quality,
                    "param_count": ind.fitness.param_count,
                    "kv_cache_size": ind.fitness.kv_cache_size,
                } if ind.fitness else None,
                "rank": ind.rank,
            }
            for ind in population
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    log.info(f"Saved: {path}")


def load_genome(path, num_layers):
    """Load genome from JSON checkpoint (takes best-quality individual)."""
    with open(path) as f:
        data = json.load(f)
    if "individuals" in data:
        best = min(data["individuals"], key=lambda x: x["fitness"]["quality"])
        flat = best["genome"]
    elif "genome" in data:
        flat = data["genome"]
    else:
        raise ValueError(f"Cannot parse genome from {path}")
    return Genome.from_flat(flat, num_layers)


# ============================================================================
# Post-Evolution: Train Top-K (Paper Section 4.2 — 5B tokens)
# ============================================================================

def post_evolution_train(population, class_pool, train_data, val_data, args):
    """Select top-K Pareto-optimal genomes and train to convergence."""
    candidates = sorted(
        [ind for ind in population if ind.rank == 0],
        key=lambda ind: ind.fitness.quality,
    )[:args.top_k]
    if not candidates:
        candidates = sorted(population, key=lambda ind: ind.fitness.quality)[:args.top_k]

    log.info(f"\n{'='*60}\nPost-evolution: training {len(candidates)} candidates\n{'='*60}")
    builder = GenomeModelBuilder(class_pool, args.dim)
    out_dir = Path(args.out_dir)
    results = []

    for i, ind in enumerate(candidates):
        log.info(f"\n--- Candidate {i+1}/{len(candidates)} ---")
        log.info(f"Evolution quality={ind.fitness.quality:.2f}  params={ind.fitness.param_count:,}")
        classes = [g.liv_class for g in ind.genome.layers]
        log.info(f"Layer classes: {classes}")

        backbone = builder.build(ind.genome)
        lm = STARLanguageModel(backbone, args.vocab_size, args.dim)
        log.info(f"Total params (with embed): {count_params(lm):,}")

        ppl = train_model(lm, train_data, val_data, args,
                          steps=args.full_train_steps, prefix=f"[C{i+1}] ")
        log.info(f"Candidate {i+1} final PPL: {ppl:.2f}")
        results.append((ind, ppl))

        torch.save({
            "genome": ind.genome.flatten(),
            "layer_classes": classes,
            "model_state_dict": lm.state_dict(),
            "ppl": ppl,
            "params": ind.fitness.param_count,
        }, out_dir / f"candidate_{i+1}.pt")

    results.sort(key=lambda x: x[1])
    log.info(f"\n{'='*60}\nFinal results (sorted by PPL):")
    for rank, (ind, ppl) in enumerate(results):
        log.info(f"  #{rank+1}  PPL={ppl:.2f}  params={ind.fitness.param_count:,}")
    log.info("=" * 60)
    return results


# ============================================================================
# CLI
# ============================================================================

def build_parser():
    p = argparse.ArgumentParser(description="STAR Architecture Search + Training")
    sub = p.add_subparsers(dest="mode", required=True)

    # Shared arguments
    shared = argparse.ArgumentParser(add_help=False)
    g_model = shared.add_argument_group("Model (Paper: 125M scale)")
    g_model.add_argument("--num_layers", type=int, default=24, help="LIV layers")
    g_model.add_argument("--dim", type=int, default=768, help="Model width")
    g_model.add_argument("--vocab_size", type=int, default=50304)
    g_model.add_argument("--seq_len", type=int, default=4096)

    g_train = shared.add_argument_group("Training (Paper Table A.1)")
    g_train.add_argument("--lr", type=float, default=8e-4)
    g_train.add_argument("--beta1", type=float, default=0.9)
    g_train.add_argument("--beta2", type=float, default=0.95)
    g_train.add_argument("--weight_decay", type=float, default=0.1)
    g_train.add_argument("--grad_clip", type=float, default=1.0)
    g_train.add_argument("--warmup_steps", type=int, default=500)
    g_train.add_argument("--batch_tokens", type=int, default=250_000,
                         help="Tokens per optimizer step (paper: 0.25M)")
    g_train.add_argument("--micro_batch", type=int, default=4,
                         help="Sequences per gradient accumulation micro-step")
    g_train.add_argument("--amp", action="store_true", help="FP16 mixed precision")
    g_train.add_argument("--eval_batches", type=int, default=50)

    g_data = shared.add_argument_group("Data")
    g_data.add_argument("--data_path", type=str, help="Pre-tokenized .pt file")
    g_data.add_argument("--val_path", type=str, help="Validation .pt file")
    g_data.add_argument("--hf_dataset", type=str, help="HuggingFace dataset name")
    g_data.add_argument("--hf_split", type=str, default="train")
    g_data.add_argument("--hf_tokens", type=int, default=50_000_000)
    g_data.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    g_data.add_argument("--synthetic", action="store_true", help="Random tokens")
    g_data.add_argument("--synthetic_tokens", type=int, default=None)

    g_sys = shared.add_argument_group("System")
    g_sys.add_argument("--device", type=str, default=None)
    g_sys.add_argument("--seed", type=int, default=42)
    g_sys.add_argument("--out_dir", type=str, default="star_output")

    # --- evolve ---
    ep = sub.add_parser("evolve", parents=[shared],
                        help="Run NSGA-II architecture search")
    ep.add_argument("--evolution_steps", type=int, default=5000,
                    help="Training steps per candidate (paper: 5000)")
    ep.add_argument("--pop_size", type=int, default=DEFAULT_POP_SIZE)
    ep.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    ep.add_argument("--mutation_prob", type=float, default=DEFAULT_MUTATION_PROB)
    ep.add_argument("--elitism", type=int, default=DEFAULT_ELITISM)
    ep.add_argument("--crossover_points", type=int, default=DEFAULT_CROSSOVER_POINTS)
    ep.add_argument("--tournament_k", type=int, default=DEFAULT_TOURNAMENT_K)
    ep.add_argument("--include_extended", action="store_true")
    ep.add_argument("--no_seeds", action="store_true")
    ep.add_argument("--dry_run", action="store_true",
                    help="Skip training; use param count as quality proxy")

    # --- train ---
    tp = sub.add_parser("train", parents=[shared],
                        help="Train a specific genome")
    tp.add_argument("--genome_path", type=str, required=True,
                    help="JSON file from evolution checkpoint")
    tp.add_argument("--train_steps", type=int, default=20_000)

    # --- both ---
    bp = sub.add_parser("both", parents=[shared],
                        help="Evolve then train top-K")
    bp.add_argument("--evolution_steps", type=int, default=5000)
    bp.add_argument("--full_train_steps", type=int, default=20_000)
    bp.add_argument("--top_k", type=int, default=8,
                    help="Post-evolution: train top-K candidates (paper: 8)")
    bp.add_argument("--pop_size", type=int, default=DEFAULT_POP_SIZE)
    bp.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    bp.add_argument("--mutation_prob", type=float, default=DEFAULT_MUTATION_PROB)
    bp.add_argument("--elitism", type=int, default=DEFAULT_ELITISM)
    bp.add_argument("--crossover_points", type=int, default=DEFAULT_CROSSOVER_POINTS)
    bp.add_argument("--tournament_k", type=int, default=DEFAULT_TOURNAMENT_K)
    bp.add_argument("--include_extended", action="store_true")
    bp.add_argument("--no_seeds", action="store_true")
    bp.add_argument("--dry_run", action="store_true")

    return p


# ============================================================================
# Main Entry Points
# ============================================================================

def _get_seeds(args):
    if getattr(args, "no_seeds", False):
        return None
    return [
        seed_transformer_pp(args.num_layers),
        seed_striped_mamba(args.num_layers),
        seed_hybrid_conv(args.num_layers),
    ]


def _make_quality_fn(args, train_data, val_data):
    if getattr(args, "dry_run", False):
        return None
    evo_steps = getattr(args, "evolution_steps", 5000)
    counter = [0]

    def quality_fn(backbone, genome):
        counter[0] += 1
        lm = STARLanguageModel(backbone, args.vocab_size, args.dim)
        n = count_params(lm)
        log.info(f"[Eval #{counter[0]}] {n:,} params, {evo_steps} steps")
        ppl = train_model(lm, train_data, val_data, args,
                          steps=evo_steps, prefix=f"[E{counter[0]}] ")
        log.info(f"[Eval #{counter[0]}] PPL={ppl:.2f}")
        return ppl

    return quality_fn


def main():
    args = build_parser().parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {args.device}  Seed: {args.seed}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_data, val_data = load_data(args)

    if args.mode == "evolve":
        qfn = _make_quality_fn(args, train_data, val_data)
        population, pool = evolve_with_seeds(args, qfn, _get_seeds(args))

        pf = [p for p in population if p.rank == 0]
        log.info(f"\nEvolution complete. Pareto front: {len(pf)}")
        for i, ind in enumerate(pf):
            cls = [g.liv_class for g in ind.genome.layers]
            log.info(f"  [{i}] q={ind.fitness.quality:.2f} "
                     f"p={ind.fitness.param_count:,}  classes={cls}")

    elif args.mode == "train":
        genome = load_genome(args.genome_path, args.num_layers)
        pool = build_class_pool()
        backbone = GenomeModelBuilder(pool, args.dim).build(genome)
        lm = STARLanguageModel(backbone, args.vocab_size, args.dim)
        log.info(f"Params: {count_params(lm):,}")

        ppl = train_model(lm, train_data, val_data, args, steps=args.train_steps)
        log.info(f"Final PPL: {ppl:.2f}")

        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save({
            "genome": genome.flatten(),
            "model_state_dict": lm.state_dict(),
            "ppl": ppl,
        }, out / "trained_model.pt")

    elif args.mode == "both":
        qfn = _make_quality_fn(args, train_data, val_data)
        population, pool = evolve_with_seeds(args, qfn, _get_seeds(args))
        post_evolution_train(population, pool, train_data, val_data, args)


if __name__ == "__main__":
    main()