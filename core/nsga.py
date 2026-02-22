"""
STAR Genetic Algorithm: NSGA-II Architecture Search over the LIV Framework

Implements multi-objective evolutionary search (NSGA-II) to discover optimal
LIV layer configurations. Three objectives:
  1. Quality (PPL/loss) — requires training callback
  2. Parameter count — static, handles shared-parameter deduplication
  3. KV-cache / inference state size — static estimate

Genome encodes per-layer: LIV class, featurizer sharing group/strategy,
and feature-group sharing group/strategy (B, C, S, V bitmask).
"""

import copy
import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from core.liv import (
    SA1, SA2, SA3, SA4,
    Rec1, Rec2, Rec3, Rec4,
    GConv1, GConv2,
    GMemless,
    UnifiedLIV, UnifiedLIVBlock, DifferentialLIV,
    TokenMixType, ChannelMixType,
    FeaturizerBase, FEATURIZER_REGISTRY,
)

# =============================================================================
# Section 1: Constants & LIV Class Pool
# =============================================================================

DEFAULT_POP_SIZE = 16
DEFAULT_GENERATIONS = 18
DEFAULT_MUTATION_PROB = 0.10
DEFAULT_ELITISM = 2
DEFAULT_CROSSOVER_POINTS = 2
DEFAULT_TOURNAMENT_K = 3

# Category labels
CAT_ATTENTION = "attention"
CAT_RECURRENCE = "recurrence"
CAT_CONVOLUTION = "convolution"
CAT_MEMORYLESS = "memoryless"


@dataclass
class LIVClassSpec:
    """Specification for one LIV class in the search pool."""
    class_id: int
    name: str
    builder_fn: Callable  # (dim, **kw) -> UnifiedLIV or DifferentialLIV
    is_differential: bool
    base_class_id: Optional[int]  # None for base classes, ID for diff variants
    category: str  # attention / recurrence / convolution / memoryless
    default_kwargs: dict = field(default_factory=dict)


def _diff_builder(base_fn):
    """Return a builder that wraps base_fn in DifferentialLIV."""
    def builder(dim, **kw):
        return DifferentialLIV(base_fn, dim, **kw)
    return builder


def build_class_pool(include_extended: bool = False) -> Dict[int, LIVClassSpec]:
    """Build the LIV class pool mapping class IDs to specs."""
    pool = {}

    # --- Base classes 1-9 ---
    base_specs = [
        (1, "SA-1", SA1, CAT_ATTENTION, {}),
        (2, "SA-2", SA2, CAT_ATTENTION, {}),
        (3, "SA-3", SA3, CAT_ATTENTION, {}),
        (4, "SA-4", SA4, CAT_ATTENTION, {}),
        (5, "Rec-1", Rec1, CAT_RECURRENCE, {}),
        (6, "Rec-2", Rec2, CAT_RECURRENCE, {}),
        (7, "GConv-1", GConv1, CAT_CONVOLUTION, {}),
        (8, "GConv-2", GConv2, CAT_CONVOLUTION, {}),
        (9, "GMemless", GMemless, CAT_MEMORYLESS, {}),
    ]

    for cid, name, fn, cat, kw in base_specs:
        pool[cid] = LIVClassSpec(
            class_id=cid, name=name, builder_fn=fn,
            is_differential=False, base_class_id=None,
            category=cat, default_kwargs=kw,
        )

    # --- Differential variants 10-17 (of base 1-8) ---
    for i, base_id in enumerate(range(1, 9), start=10):
        base = pool[base_id]
        pool[i] = LIVClassSpec(
            class_id=i,
            name=f"Diff-{base.name}",
            builder_fn=_diff_builder(base.builder_fn),
            is_differential=True,
            base_class_id=base_id,
            category=base.category,
            default_kwargs=dict(base.default_kwargs),
        )

    # --- Extended: Rec3, Rec4, and their differential variants ---
    if include_extended:
        pool[18] = LIVClassSpec(
            class_id=18, name="Rec-3", builder_fn=Rec3,
            is_differential=False, base_class_id=None,
            category=CAT_RECURRENCE, default_kwargs={},
        )
        pool[19] = LIVClassSpec(
            class_id=19, name="Rec-4", builder_fn=Rec4,
            is_differential=False, base_class_id=None,
            category=CAT_RECURRENCE, default_kwargs={},
        )
        pool[20] = LIVClassSpec(
            class_id=20, name="Diff-Rec-3",
            builder_fn=_diff_builder(Rec3),
            is_differential=True, base_class_id=18,
            category=CAT_RECURRENCE, default_kwargs={},
        )
        pool[21] = LIVClassSpec(
            class_id=21, name="Diff-Rec-4",
            builder_fn=_diff_builder(Rec4),
            is_differential=True, base_class_id=19,
            category=CAT_RECURRENCE, default_kwargs={},
        )

    return pool


# =============================================================================
# Section 2: Genome Representation
# =============================================================================

# Feature-group sharing bitmask bits
BIT_B = 1   # bit 0
BIT_C = 2   # bit 1
BIT_S = 4   # bit 2
BIT_V = 8   # bit 3

# Valid bitmask values per category (which B/C/S/V can be shared)
VALID_FG_MASKS = {
    CAT_ATTENTION:   [0, BIT_B, BIT_C, BIT_V, BIT_B | BIT_C,
                      BIT_B | BIT_V, BIT_C | BIT_V,
                      BIT_B | BIT_C | BIT_V],
    CAT_RECURRENCE:  [0, BIT_B, BIT_C, BIT_S, BIT_V,
                      BIT_B | BIT_C, BIT_B | BIT_S, BIT_B | BIT_V,
                      BIT_C | BIT_S, BIT_C | BIT_V, BIT_S | BIT_V,
                      BIT_B | BIT_C | BIT_S, BIT_B | BIT_C | BIT_V,
                      BIT_B | BIT_S | BIT_V, BIT_C | BIT_S | BIT_V,
                      BIT_B | BIT_C | BIT_S | BIT_V],
    CAT_CONVOLUTION: [0, BIT_B, BIT_C, BIT_S,
                      BIT_B | BIT_C, BIT_B | BIT_S, BIT_C | BIT_S,
                      BIT_B | BIT_C | BIT_S],
    CAT_MEMORYLESS:  [0, BIT_B, BIT_V, BIT_B | BIT_V],
}

# Gene field count per layer
GENES_PER_LAYER = 5


@dataclass
class LayerGene:
    """Genome for a single layer."""
    liv_class: int           # Gene 1: LIV class ID (1-17+)
    feat_share_group: int    # Gene 2: featurizer sharing group ID
    feat_share_strategy: int # Gene 3: 1=none, 2=all_shared
    fg_share_group: int      # Gene 4: feature group sharing group ID
    fg_share_strategy: int   # Gene 5: bitmask of which B,C,S,V shared


@dataclass
class Genome:
    """Full genome: one LayerGene per model layer."""
    layers: List[LayerGene]

    def flatten(self) -> List[int]:
        """Flatten to 1D integer list for crossover operations."""
        flat = []
        for g in self.layers:
            flat.extend([
                g.liv_class,
                g.feat_share_group,
                g.feat_share_strategy,
                g.fg_share_group,
                g.fg_share_strategy,
            ])
        return flat

    @staticmethod
    def from_flat(flat: List[int], num_layers: int) -> "Genome":
        """Reconstruct Genome from flattened integer list."""
        assert len(flat) == num_layers * GENES_PER_LAYER
        layers = []
        for i in range(num_layers):
            off = i * GENES_PER_LAYER
            layers.append(LayerGene(
                liv_class=flat[off],
                feat_share_group=flat[off + 1],
                feat_share_strategy=flat[off + 2],
                fg_share_group=flat[off + 3],
                fg_share_strategy=flat[off + 4],
            ))
        return Genome(layers=layers)

    def copy(self) -> "Genome":
        """Deep copy."""
        return Genome(layers=[LayerGene(
            g.liv_class, g.feat_share_group, g.feat_share_strategy,
            g.fg_share_group, g.fg_share_strategy,
        ) for g in self.layers])


def random_genome(num_layers: int, class_pool: Dict[int, LIVClassSpec],
                  rng: random.Random = None) -> Genome:
    """Generate a random valid genome."""
    rng = rng or random.Random()
    valid_ids = list(class_pool.keys())
    layers = []
    for i in range(num_layers):
        cid = rng.choice(valid_ids)
        cat = class_pool[cid].category
        valid_masks = VALID_FG_MASKS[cat]
        layers.append(LayerGene(
            liv_class=cid,
            feat_share_group=rng.randint(0, num_layers - 1),
            feat_share_strategy=rng.choice([1, 2]),
            fg_share_group=rng.randint(0, num_layers - 1),
            fg_share_strategy=rng.choice(valid_masks),
        ))
    return Genome(layers=layers)


# =============================================================================
# Section 3: Genome-to-Model Builder
# =============================================================================

# Parameter name mapping per category for feature-group sharing
# Maps bitmask bit -> attribute name(s) on the featurizer
PARAM_MAP = {
    CAT_ATTENTION: {
        BIT_B: ["W_K"],
        BIT_C: ["W_Q"],
        BIT_V: ["W_V"],
        BIT_S: [],  # attention has no S parameter to share
    },
    CAT_RECURRENCE: {
        BIT_B: ["W_B"],
        BIT_C: ["W_C"],
        BIT_S: ["W_A", "A_log"],
        BIT_V: ["W_V"],
    },
    CAT_CONVOLUTION: {
        BIT_B: ["W_B"],
        BIT_C: ["W_C"],
        BIT_S: ["kernel", "kernel_net"],
        BIT_V: [],  # conv uses raw x, no V parameter
    },
    CAT_MEMORYLESS: {
        BIT_B: ["W_gate"],
        BIT_C: [],  # memoryless C is always ones
        BIT_S: [],
        BIT_V: ["W_value"],
    },
}


def _get_featurizer(liv_module):
    """Extract the featurizer from a UnifiedLIV or DifferentialLIV."""
    if isinstance(liv_module, DifferentialLIV):
        return liv_module.liv1.featurizer
    elif isinstance(liv_module, UnifiedLIV):
        return liv_module.featurizer
    return None


def _set_featurizer(liv_module, featurizer):
    """Replace the featurizer on a UnifiedLIV or DifferentialLIV."""
    if isinstance(liv_module, DifferentialLIV):
        liv_module.liv1.featurizer = featurizer
        liv_module.liv2.featurizer = featurizer
    elif isinstance(liv_module, UnifiedLIV):
        liv_module.featurizer = featurizer


class GenomeModelBuilder:
    """Builds a nn.Module from a Genome using the LIV class pool."""

    def __init__(self, class_pool: Dict[int, LIVClassSpec], dim: int, **kwargs):
        self.class_pool = class_pool
        self.dim = dim
        self.kwargs = kwargs

    def build(self, genome: Genome) -> nn.Module:
        """Build a full model (nn.Sequential of UnifiedLIVBlocks) from genome."""
        num_layers = len(genome.layers)
        dim = self.dim

        # Step 1: Build each layer's LIV module
        liv_modules = []
        for gene in genome.layers:
            spec = self.class_pool[gene.liv_class]
            kw = dict(spec.default_kwargs)
            kw.update(self.kwargs)
            module = spec.builder_fn(dim, **kw)
            liv_modules.append(module)

        # Step 2: Apply featurizer sharing (genes 2-3)
        self._apply_featurizer_sharing(genome, liv_modules)

        # Step 3: Apply feature-group sharing (genes 4-5)
        self._apply_fg_sharing(genome, liv_modules)

        # Step 4: Wrap in blocks
        blocks = nn.ModuleList()
        for module in liv_modules:
            blocks.append(UnifiedLIVBlock(dim, module))

        return nn.Sequential(*blocks)

    def _apply_featurizer_sharing(self, genome: Genome, liv_modules: list):
        """Apply featurizer sharing: layers in the same group with strategy=2
        share one featurizer instance."""
        # Group layers by (feat_share_group, feat_share_strategy=2)
        groups: Dict[int, List[int]] = {}
        for i, gene in enumerate(genome.layers):
            if gene.feat_share_strategy == 2:
                groups.setdefault(gene.feat_share_group, []).append(i)

        for group_id, indices in groups.items():
            if len(indices) < 2:
                continue

            # All layers in a sharing group must have compatible featurizers
            # (same category, same differential status). Use the first as source.
            source_idx = indices[0]
            source_feat = _get_featurizer(liv_modules[source_idx])
            if source_feat is None:
                continue

            for idx in indices[1:]:
                src_cat = self.class_pool[genome.layers[source_idx].liv_class].category
                dst_cat = self.class_pool[genome.layers[idx].liv_class].category
                src_diff = self.class_pool[genome.layers[source_idx].liv_class].is_differential
                dst_diff = self.class_pool[genome.layers[idx].liv_class].is_differential
                if src_cat == dst_cat and src_diff == dst_diff:
                    _set_featurizer(liv_modules[idx], source_feat)

    def _apply_fg_sharing(self, genome: Genome, liv_modules: list):
        """Apply feature-group sharing: share specific B/C/S/V parameters
        between featurizers in the same fg_share_group."""
        groups: Dict[int, List[int]] = {}
        for i, gene in enumerate(genome.layers):
            if gene.fg_share_strategy > 0:
                groups.setdefault(gene.fg_share_group, []).append(i)

        for group_id, indices in groups.items():
            if len(indices) < 2:
                continue

            source_idx = indices[0]
            source_gene = genome.layers[source_idx]
            source_feat = _get_featurizer(liv_modules[source_idx])
            if source_feat is None:
                continue
            source_cat = self.class_pool[source_gene.liv_class].category
            mask = source_gene.fg_share_strategy

            for idx in indices[1:]:
                target_gene = genome.layers[idx]
                target_feat = _get_featurizer(liv_modules[idx])
                target_cat = self.class_pool[target_gene.liv_class].category
                if target_feat is None or target_cat != source_cat:
                    continue

                target_mask = target_gene.fg_share_strategy
                shared_mask = mask & target_mask

                for bit in [BIT_B, BIT_C, BIT_S, BIT_V]:
                    if shared_mask & bit:
                        attr_names = PARAM_MAP.get(source_cat, {}).get(bit, [])
                        for attr in attr_names:
                            if hasattr(source_feat, attr) and hasattr(target_feat, attr):
                                src_param = getattr(source_feat, attr)
                                setattr(target_feat, attr, src_param)


# =============================================================================
# Section 4: Fitness Evaluation
# =============================================================================

@dataclass
class FitnessResult:
    """Multi-objective fitness values."""
    quality: float       # Obj 1: PPL/loss (lower is better)
    param_count: int     # Obj 2: trainable parameter count
    kv_cache_size: int   # Obj 3: inference cache/state estimate (bytes-like)
    latency_ms: float = 0.0  # Obj 4 (optional): backbone forward latency in ms
                              # 0.0 means not measured — excluded from dominance


def count_params(model: nn.Module) -> int:
    """Count trainable parameters, deduplicating shared ones via id()."""
    seen = set()
    total = 0
    for p in model.parameters():
        pid = id(p)
        if pid not in seen:
            seen.add(pid)
            total += p.numel()
    return total


def estimate_kv_cache(genome: Genome, class_pool: Dict[int, LIVClassSpec],
                      dim: int, seq_len: int = 1024) -> int:
    """Estimate inference-time cache/state size in number of elements.

    - Attention: 2 * seq_len * dim per layer (K and V caches)
    - Recurrence: state_dim (internal_dim) per layer
    - Convolution: kernel_size * dim per layer
    - Memoryless: 0
    - Differential: 2x the base estimate
    """
    total = 0
    for gene in genome.layers:
        spec = class_pool[gene.liv_class]
        cat = spec.category
        is_diff = spec.is_differential
        multiplier = 2 if is_diff else 1

        if cat == CAT_ATTENTION:
            layer_cache = 2 * seq_len * dim
        elif cat == CAT_RECURRENCE:
            # Recurrence state depends on expansion
            if spec.base_class_id in (5, 18) or gene.liv_class in (5, 18):
                expansion = 16
            elif spec.base_class_id == 19 or gene.liv_class == 19:
                expansion = 16
            else:
                expansion = 2
            layer_cache = dim * expansion
        elif cat == CAT_CONVOLUTION:
            # Kernel size for buffered conv
            if spec.base_class_id == 8 or gene.liv_class == 8:
                kernel_size = 64
            else:
                kernel_size = 3
            layer_cache = kernel_size * dim
        elif cat == CAT_MEMORYLESS:
            layer_cache = 0
        else:
            layer_cache = 0

        total += layer_cache * multiplier
    return total


class FitnessEvaluator:
    """Evaluates fitness for a genome."""

    def __init__(self, class_pool: Dict[int, LIVClassSpec], dim: int,
                 quality_fn: Optional[Callable] = None,
                 seq_len: int = 1024,
                 measure_latency: bool = False,
                 latency_warmup: int = 3,
                 latency_runs: int = 10,
                 max_params: int = 0,
                 **build_kwargs):
        self.class_pool = class_pool
        self.dim = dim
        self.quality_fn = quality_fn
        self.seq_len = seq_len
        self.measure_latency = measure_latency
        self.latency_warmup = latency_warmup
        self.latency_runs = latency_runs
        self.max_params = max_params  # 0 = no cap
        self.builder = GenomeModelBuilder(class_pool, dim, **build_kwargs)

    def _measure_latency(self, model: nn.Module) -> float:
        """Time a single-sample backbone forward pass (ms). CPU-only for fairness."""
        model.eval()
        x = torch.randn(1, self.seq_len, self.dim)
        with torch.no_grad():
            for _ in range(self.latency_warmup):
                model(x)
            t0 = time.perf_counter()
            for _ in range(self.latency_runs):
                model(x)
        model.train()
        return (time.perf_counter() - t0) / self.latency_runs * 1000.0

    def evaluate(self, genome: Genome) -> FitnessResult:
        """Compute all fitness objectives for a genome."""
        model = self.builder.build(genome)

        param_c = count_params(model)
        kv_cache = estimate_kv_cache(genome, self.class_pool, self.dim,
                                     self.seq_len)

        # Skip training for models that exceed the param cap — assign worst
        # possible quality so NSGA-II naturally rejects them without OOM.
        if self.max_params > 0 and param_c > self.max_params:
            return FitnessResult(
                quality=float("inf"),
                param_count=param_c,
                kv_cache_size=kv_cache,
                latency_ms=0.0,
            )

        if self.quality_fn is not None:
            quality = self.quality_fn(model, genome)
        else:
            # Without training, use param count as a proxy
            quality = float(param_c)

        latency = self._measure_latency(model) if self.measure_latency else 0.0

        return FitnessResult(
            quality=quality,
            param_count=param_c,
            kv_cache_size=kv_cache,
            latency_ms=latency,
        )


# =============================================================================
# Section 5: NSGA-II Core
# =============================================================================

@dataclass
class Individual:
    """An individual in the population."""
    genome: Genome
    fitness: Optional[FitnessResult] = None
    rank: int = 0
    crowding_distance: float = 0.0


def _dominates(a: FitnessResult, b: FitnessResult) -> bool:
    """True if a dominates b (no worse in all objectives, strictly better in >=1).

    latency_ms is included as a 4th objective only when measured (> 0 in either).
    """
    objs_a = [a.quality, a.param_count, a.kv_cache_size]
    objs_b = [b.quality, b.param_count, b.kv_cache_size]
    if a.latency_ms > 0 or b.latency_ms > 0:
        objs_a.append(a.latency_ms)
        objs_b.append(b.latency_ms)
    no_worse = all(oa <= ob for oa, ob in zip(objs_a, objs_b))
    strictly_better = any(oa < ob for oa, ob in zip(objs_a, objs_b))
    return no_worse and strictly_better


def non_dominated_sort(population: List[Individual]) -> List[List[int]]:
    """Fast non-dominated sort (Deb 2002).

    Returns list of fronts, each front is a list of indices into population.
    Front 0 = Pareto-optimal, Front 1 = next best, etc.
    """
    n = len(population)
    domination_count = [0] * n      # how many dominate me
    dominated_set: List[List[int]] = [[] for _ in range(n)]  # who I dominate
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            fp = population[p].fitness
            fq = population[q].fitness
            if fp is None or fq is None:
                continue
            if _dominates(fp, fq):
                dominated_set[p].append(q)
            elif _dominates(fq, fp):
                domination_count[p] += 1

        if domination_count[p] == 0:
            population[p].rank = 0
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_set[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    population[q].rank = i + 1
                    next_front.append(q)
        i += 1
        if next_front:
            fronts.append(next_front)

    # Remove empty trailing front
    return [f for f in fronts if f]


def crowding_distance(population: List[Individual],
                      front_indices: List[int]) -> None:
    """Compute crowding distance for individuals in a front (in-place)."""
    if len(front_indices) <= 2:
        for idx in front_indices:
            population[idx].crowding_distance = float('inf')
        return

    for idx in front_indices:
        population[idx].crowding_distance = 0.0

    # For each objective (latency added when measured in any individual in front)
    objectives = [
        lambda ind: ind.fitness.quality,
        lambda ind: float(ind.fitness.param_count),
        lambda ind: float(ind.fitness.kv_cache_size),
    ]
    if any(population[i].fitness.latency_ms > 0 for i in front_indices):
        objectives.append(lambda ind: ind.fitness.latency_ms)

    for obj_fn in objectives:
        sorted_idx = sorted(front_indices, key=lambda i: obj_fn(population[i]))

        # Boundary individuals get infinity
        population[sorted_idx[0]].crowding_distance = float('inf')
        population[sorted_idx[-1]].crowding_distance = float('inf')

        obj_min = obj_fn(population[sorted_idx[0]])
        obj_max = obj_fn(population[sorted_idx[-1]])
        obj_range = obj_max - obj_min
        if obj_range < 1e-12:
            continue

        for k in range(1, len(sorted_idx) - 1):
            dist = (obj_fn(population[sorted_idx[k + 1]])
                    - obj_fn(population[sorted_idx[k - 1]]))
            population[sorted_idx[k]].crowding_distance += dist / obj_range


# =============================================================================
# Section 6: Genetic Operators
# =============================================================================

def tournament_select(population: List[Individual], k: int,
                      rng: random.Random) -> Individual:
    """Tournament selection: pick k random, return best by (rank, -crowding)."""
    candidates = rng.sample(range(len(population)), min(k, len(population)))
    best = candidates[0]
    for c in candidates[1:]:
        p_best = population[best]
        p_c = population[c]
        # Prefer lower rank; if tie, prefer higher crowding distance
        if (p_c.rank < p_best.rank or
                (p_c.rank == p_best.rank and
                 p_c.crowding_distance > p_best.crowding_distance)):
            best = c
    return population[best]


def crossover(parent1: Genome, parent2: Genome,
              num_points: int = DEFAULT_CROSSOVER_POINTS,
              rng: random.Random = None) -> Tuple[Genome, Genome]:
    """Two-point (or n-point) crossover on flattened genomes."""
    rng = rng or random.Random()
    flat1 = parent1.flatten()
    flat2 = parent2.flatten()
    length = len(flat1)

    # Pick crossover points
    points = sorted(rng.sample(range(1, length), min(num_points, length - 1)))

    child1 = list(flat1)
    child2 = list(flat2)
    swap = False
    prev = 0
    for pt in points:
        if swap:
            child1[prev:pt], child2[prev:pt] = child2[prev:pt], child1[prev:pt]
        swap = not swap
        prev = pt
    if swap:
        child1[prev:], child2[prev:] = child2[prev:], child1[prev:]

    num_layers = len(parent1.layers)
    return (Genome.from_flat(child1, num_layers),
            Genome.from_flat(child2, num_layers))


def mutate(genome: Genome, mutation_prob: float,
           class_pool: Dict[int, LIVClassSpec],
           rng: random.Random = None) -> None:
    """Mutate genome in-place. Hierarchical: if liv_class mutates,
    dependent genes are re-validated."""
    rng = rng or random.Random()
    valid_ids = list(class_pool.keys())
    num_layers = len(genome.layers)

    for gene in genome.layers:
        # Gene 1: LIV class
        if rng.random() < mutation_prob:
            old_class = gene.liv_class
            gene.liv_class = rng.choice(valid_ids)
            # If class changed, re-validate dependent genes
            if gene.liv_class != old_class:
                cat = class_pool[gene.liv_class].category
                valid_masks = VALID_FG_MASKS[cat]
                if gene.fg_share_strategy not in valid_masks:
                    gene.fg_share_strategy = rng.choice(valid_masks)

        # Gene 2: featurizer sharing group
        if rng.random() < mutation_prob:
            gene.feat_share_group = rng.randint(0, num_layers - 1)

        # Gene 3: featurizer sharing strategy
        if rng.random() < mutation_prob:
            gene.feat_share_strategy = rng.choice([1, 2])

        # Gene 4: feature group sharing group
        if rng.random() < mutation_prob:
            gene.fg_share_group = rng.randint(0, num_layers - 1)

        # Gene 5: feature group sharing strategy (bitmask)
        if rng.random() < mutation_prob:
            cat = class_pool[gene.liv_class].category
            valid_masks = VALID_FG_MASKS[cat]
            gene.fg_share_strategy = rng.choice(valid_masks)


def repair(genome: Genome, class_pool: Dict[int, LIVClassSpec]) -> None:
    """Repair genome to enforce all constraints.

    1. liv_class must be in pool
    2. Featurizer sharing groups must have compatible classes
    3. fg_share_strategy bitmask valid for category
    4. Differential classes can't share featurizers with non-differential
    5. Solo sharing groups get strategy reset to 1
    """
    valid_ids = set(class_pool.keys())

    # 1. Validate class IDs
    for gene in genome.layers:
        if gene.liv_class not in valid_ids:
            gene.liv_class = min(valid_ids)

    # 3. Validate fg_share_strategy bitmask
    for gene in genome.layers:
        cat = class_pool[gene.liv_class].category
        valid_masks = VALID_FG_MASKS[cat]
        if gene.fg_share_strategy not in valid_masks:
            gene.fg_share_strategy = 0  # no sharing

    # 2 & 4. Validate featurizer sharing groups
    feat_groups: Dict[int, List[int]] = {}
    for i, gene in enumerate(genome.layers):
        if gene.feat_share_strategy == 2:
            feat_groups.setdefault(gene.feat_share_group, []).append(i)

    for group_id, indices in feat_groups.items():
        if len(indices) < 2:
            # 5. Solo group — reset to no sharing
            for idx in indices:
                genome.layers[idx].feat_share_strategy = 1
            continue

        # Check compatibility: same category and same differential status
        cats = set()
        diffs = set()
        for idx in indices:
            spec = class_pool[genome.layers[idx].liv_class]
            cats.add(spec.category)
            diffs.add(spec.is_differential)

        if len(cats) > 1 or len(diffs) > 1:
            # Incompatible — break the group
            for idx in indices:
                genome.layers[idx].feat_share_strategy = 1

    # Validate fg sharing groups similarly
    fg_groups: Dict[int, List[int]] = {}
    for i, gene in enumerate(genome.layers):
        if gene.fg_share_strategy > 0:
            fg_groups.setdefault(gene.fg_share_group, []).append(i)

    for group_id, indices in fg_groups.items():
        if len(indices) < 2:
            for idx in indices:
                genome.layers[idx].fg_share_strategy = 0
            continue

        # Must share same category
        cats = set()
        for idx in indices:
            cats.add(class_pool[genome.layers[idx].liv_class].category)
        if len(cats) > 1:
            for idx in indices:
                genome.layers[idx].fg_share_strategy = 0


# =============================================================================
# Section 7: Evolution Loop
# =============================================================================

def evolve(
    num_layers: int,
    dim: int,
    quality_fn: Optional[Callable] = None,
    pop_size: int = DEFAULT_POP_SIZE,
    generations: int = DEFAULT_GENERATIONS,
    mutation_prob: float = DEFAULT_MUTATION_PROB,
    elitism: int = DEFAULT_ELITISM,
    tournament_k: int = DEFAULT_TOURNAMENT_K,
    crossover_points: int = DEFAULT_CROSSOVER_POINTS,
    include_extended: bool = False,
    seed: Optional[int] = None,
    callback: Optional[Callable] = None,
    class_pool: Optional[Dict[int, LIVClassSpec]] = None,
    **kwargs,
) -> List[Individual]:
    """Run NSGA-II evolution loop.

    Args:
        num_layers: Number of LIV layers in each architecture.
        dim: Model dimension.
        quality_fn: Optional callable (model, genome) -> float for quality
            objective. If None, only static objectives are used.
        pop_size: Population size.
        generations: Number of generations.
        mutation_prob: Per-gene mutation probability.
        elitism: Number of elite individuals carried forward.
        tournament_k: Tournament selection size.
        crossover_points: Number of crossover points.
        include_extended: Include Rec3/Rec4 in the class pool.
        seed: Random seed for reproducibility.
        callback: Optional callable (generation, population) called each gen.
        class_pool: Optional custom class pool. If None, built from
            include_extended. Use to restrict search to specific LIV families.
        **kwargs: Extra keyword arguments passed to LIV builders.

    Returns:
        Final population sorted by (rank, -crowding_distance).
    """
    rng = random.Random(seed)
    if class_pool is None:
        class_pool = build_class_pool(include_extended)
    evaluator = FitnessEvaluator(class_pool, dim, quality_fn, **kwargs)

    # Initialize population
    population: List[Individual] = []
    for _ in range(pop_size):
        genome = random_genome(num_layers, class_pool, rng)
        repair(genome, class_pool)
        fitness = evaluator.evaluate(genome)
        population.append(Individual(genome=genome, fitness=fitness))

    for gen in range(generations):
        # 1. Non-dominated sort + crowding distance
        fronts = non_dominated_sort(population)
        for front in fronts:
            crowding_distance(population, front)

        # 2. Sort by (rank, -crowding_distance)
        population.sort(key=lambda ind: (ind.rank, -ind.crowding_distance))

        # Optional callback
        if callback is not None:
            callback(gen, population)

        # 3. Create offspring
        offspring: List[Individual] = []

        # Elitism: carry top individuals
        for i in range(min(elitism, len(population))):
            elite = Individual(genome=population[i].genome.copy(),
                               fitness=population[i].fitness)
            offspring.append(elite)

        # Fill rest with crossover + mutation
        while len(offspring) < pop_size:
            parent1 = tournament_select(population, tournament_k, rng)
            parent2 = tournament_select(population, tournament_k, rng)

            child1_genome, child2_genome = crossover(
                parent1.genome, parent2.genome, crossover_points, rng)

            mutate(child1_genome, mutation_prob, class_pool, rng)
            mutate(child2_genome, mutation_prob, class_pool, rng)
            repair(child1_genome, class_pool)
            repair(child2_genome, class_pool)

            fitness1 = evaluator.evaluate(child1_genome)
            offspring.append(Individual(genome=child1_genome, fitness=fitness1))

            if len(offspring) < pop_size:
                fitness2 = evaluator.evaluate(child2_genome)
                offspring.append(Individual(
                    genome=child2_genome, fitness=fitness2))

        population = offspring[:pop_size]

    # Final sort
    fronts = non_dominated_sort(population)
    for front in fronts:
        crowding_distance(population, front)
    population.sort(key=lambda ind: (ind.rank, -ind.crowding_distance))

    return population


# =============================================================================
# Section 8: User-Facing API
# =============================================================================

@dataclass
class SearchResult:
    """Results from an architecture search run."""
    population: List[Individual]
    class_pool: Dict[int, LIVClassSpec]
    dim: int
    build_kwargs: dict

    @property
    def pareto_front(self) -> List[Individual]:
        """Return rank-0 (Pareto-optimal) individuals."""
        return [ind for ind in self.population if ind.rank == 0]

    @property
    def best_quality(self) -> Individual:
        """Individual with the best (lowest) quality metric."""
        valid = [ind for ind in self.population if ind.fitness is not None]
        return min(valid, key=lambda ind: ind.fitness.quality)

    @property
    def smallest(self) -> Individual:
        """Individual with fewest parameters."""
        valid = [ind for ind in self.population if ind.fitness is not None]
        return min(valid, key=lambda ind: ind.fitness.param_count)

    def build_model(self, individual: Individual) -> nn.Module:
        """Build a PyTorch model from an individual's genome."""
        builder = GenomeModelBuilder(self.class_pool, self.dim,
                                     **self.build_kwargs)
        return builder.build(individual.genome)

    def summary(self) -> str:
        """Human-readable summary of search results."""
        lines = []
        lines.append("=" * 60)
        lines.append("STAR Architecture Search Results")
        lines.append("=" * 60)
        lines.append(f"Population size: {len(self.population)}")
        pf = self.pareto_front
        lines.append(f"Pareto front size: {len(pf)}")

        if pf:
            lines.append("")
            lines.append("Pareto-optimal architectures:")
            lines.append("-" * 60)
            for i, ind in enumerate(pf):
                f = ind.fitness
                classes = [g.liv_class for g in ind.genome.layers]
                class_names = []
                for cid in classes:
                    if cid in self.class_pool:
                        class_names.append(self.class_pool[cid].name)
                    else:
                        class_names.append(f"?{cid}")
                lines.append(
                    f"  [{i}] quality={f.quality:.4f}  "
                    f"params={f.param_count:,}  "
                    f"cache={f.kv_cache_size:,}")
                lines.append(f"      layers: {class_names}")

        bq = self.best_quality
        sm = self.smallest
        lines.append("")
        lines.append(f"Best quality: {bq.fitness.quality:.4f} "
                     f"(params={bq.fitness.param_count:,})")
        lines.append(f"Smallest:     {sm.fitness.param_count:,} params "
                     f"(quality={sm.fitness.quality:.4f})")
        lines.append("=" * 60)
        return "\n".join(lines)


def search(
    num_layers: int,
    dim: int,
    quality_fn: Optional[Callable] = None,
    class_pool: Optional[Dict[int, LIVClassSpec]] = None,
    **kwargs,
) -> SearchResult:
    """Run STAR architecture search.

    Args:
        num_layers: Number of LIV layers in each architecture.
        dim: Model dimension.
        quality_fn: Optional callable (model, genome) -> float for training-
            based quality evaluation. If None, parameter count is used as proxy.
        class_pool: Optional custom class pool to restrict search space.
        **kwargs: All other arguments passed to evolve().

    Returns:
        SearchResult with the final population and helper methods.
    """
    include_extended = kwargs.pop("include_extended", False)
    build_kwargs = {}
    # Extract build-specific kwargs that shouldn't go to evolve
    for key in list(kwargs.keys()):
        if key in ("num_heads", "kernel_size", "expansion",
                    "causal", "use_softmax"):
            build_kwargs[key] = kwargs.pop(key)

    if class_pool is None:
        class_pool = build_class_pool(include_extended)

    population = evolve(
        num_layers=num_layers,
        dim=dim,
        quality_fn=quality_fn,
        include_extended=include_extended,
        class_pool=class_pool,
        **kwargs,
    )

    return SearchResult(
        population=population,
        class_pool=class_pool,
        dim=dim,
        build_kwargs=build_kwargs,
    )


def genome_to_config(genome: Genome,
                     class_pool: Dict[int, LIVClassSpec]) -> List[tuple]:
    """Convert a genome to STARBackbone-compatible config list.

    Returns list of (featurizer_cls, token_mix, channel_mix) tuples.
    Note: sharing information is lost in this conversion.
    """
    # Mapping from category to (token_mix_type, channel_mix_type)
    cat_to_mix = {
        CAT_ATTENTION:   (TokenMixType.LOW_RANK.value, ChannelMixType.GROUPED.value),
        CAT_RECURRENCE:  (TokenMixType.SEMI_SEPARABLE.value, ChannelMixType.DIAGONAL.value),
        CAT_CONVOLUTION: (TokenMixType.TOEPLITZ.value, ChannelMixType.DIAGONAL.value),
        CAT_MEMORYLESS:  (TokenMixType.DIAGONAL.value, ChannelMixType.DENSE.value),
    }

    # Mapping from class_id to featurizer_cls ID in FEATURIZER_REGISTRY
    class_to_feat = {
        1: 1, 2: 2, 3: 3, 4: 4,
        5: 5, 6: 6, 7: 7, 8: 8, 9: 9,
        # Differential variants map to their base featurizer
        10: 1, 11: 2, 12: 3, 13: 4,
        14: 5, 15: 6, 16: 7, 17: 8,
        # Extended
        18: 10, 19: 11, 20: 10, 21: 11,
    }

    configs = []
    for gene in genome.layers:
        spec = class_pool[gene.liv_class]
        t, c = cat_to_mix[spec.category]
        feat_cls = class_to_feat.get(gene.liv_class, 1)
        configs.append((feat_cls, t, c))
    return configs


# =============================================================================
# Quick self-test (python star_evolution.py)
# =============================================================================

if __name__ == "__main__":
    print("Running STAR evolution self-test...")

    pool = build_class_pool(include_extended=False)
    print(f"Class pool: {len(pool)} classes")
    for cid, spec in sorted(pool.items()):
        print(f"  {cid:2d}: {spec.name:<16s} cat={spec.category:<12s} "
              f"diff={spec.is_differential}")

    # 1. Random genome + repair
    g = random_genome(24, pool)
    assert len(g.layers) == 24
    assert all(gene.liv_class in pool for gene in g.layers)
    repair(g, pool)
    print("  [OK] Random genome generation + repair")

    # 2. Flatten/unflatten roundtrip
    flat = g.flatten()
    g2 = Genome.from_flat(flat, 24)
    assert g2.flatten() == flat
    print("  [OK] Genome flatten/unflatten roundtrip")

    # 3. Build model with sharing
    g_share = random_genome(4, pool, random.Random(42))
    # Force featurizer sharing: layers 0 and 1 share the same class + group
    g_share.layers[0].feat_share_group = 0
    g_share.layers[0].feat_share_strategy = 2
    g_share.layers[1].feat_share_group = 0
    g_share.layers[1].feat_share_strategy = 2
    g_share.layers[1].liv_class = g_share.layers[0].liv_class
    repair(g_share, pool)
    builder = GenomeModelBuilder(pool, 64)
    model = builder.build(g_share)
    # Verify sharing by checking featurizer object identity
    feat0 = _get_featurizer(model[0].liv)
    feat1 = _get_featurizer(model[1].liv)
    if g_share.layers[0].feat_share_strategy == 2:
        assert feat0 is feat1, "Expected shared featurizer objects"
        # Shared featurizer means count_params < sum of individual params
        params_shared = count_params(model)
        g_no_share = g_share.copy()
        g_no_share.layers[0].feat_share_strategy = 1
        g_no_share.layers[1].feat_share_strategy = 1
        model_no_share = builder.build(g_no_share)
        params_unshared = count_params(model_no_share)
        assert params_shared < params_unshared, \
            f"Shared ({params_shared}) should be < unshared ({params_unshared})"
        print(f"  [OK] Model build with parameter sharing "
              f"({params_shared:,} < {params_unshared:,})")
    else:
        print("  [OK] Model build (sharing was repaired away)")

    # 4. Static fitness
    evaluator = FitnessEvaluator(pool, 64)
    result = evaluator.evaluate(g_share)
    assert result.param_count > 0
    assert result.kv_cache_size >= 0
    print(f"  [OK] Static fitness: params={result.param_count:,}, "
          f"cache={result.kv_cache_size:,}")

    # 5. NSGA-II mini run
    pop = evolve(num_layers=4, dim=64, quality_fn=None,
                 pop_size=8, generations=3, seed=123)
    assert len(pop) == 8
    assert all(ind.fitness is not None for ind in pop)
    assert pop[0].rank == 0
    print(f"  [OK] NSGA-II: {len(pop)} individuals, "
          f"best rank={pop[0].rank}")

    # 6. Crossover + mutation + repair
    g_a = random_genome(24, pool, random.Random(1))
    g_b = random_genome(24, pool, random.Random(2))
    c1, c2 = crossover(g_a, g_b)
    mutate(c1, 0.5, pool)
    repair(c1, pool)
    assert all(gene.liv_class in pool for gene in c1.layers)
    print("  [OK] Crossover + mutation + repair")

    # 7. Forward + backward
    x = torch.randn(2, 16, 64)
    small_g = random_genome(2, pool, random.Random(99))
    repair(small_g, pool)
    small_model = builder.build(small_g)
    y = small_model(x)
    y.sum().backward()
    print(f"  [OK] Forward+backward: input={x.shape} -> output={y.shape}")

    # 8. Search API
    sr = search(num_layers=4, dim=64, pop_size=8, generations=2, seed=42)
    print(sr.summary())

    # 9. genome_to_config
    cfg = genome_to_config(sr.best_quality.genome, sr.class_pool)
    print(f"  [OK] genome_to_config: {cfg}")

    print("\nAll self-tests passed.")
