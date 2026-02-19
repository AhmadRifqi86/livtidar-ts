# LIV-TiDAR for Time Series Forecasting: Research Plan

## Title (Working)
**"LIV-TS: Evolutionary Architecture Search for Time Series Forecasting via Linear Input-Varying Systems"**

---

## 1. Problem Statement

Current time series forecasting models use fixed architectural choices:
- DMamba hardcodes Mamba (SSM) for seasonal + MLP for trend
- S-Mamba hardcodes bidirectional Mamba for variate correlation + FFN for temporal
- iTransformer hardcodes attention for variate correlation + FFN for temporal

**Question**: What if we don't assume which operator is best for each component? Can evolutionary search over LIV operators automatically discover the optimal architecture for time series?

---

## 2. Technical Background

### 2.1 LIV — Linear Input-Varying Systems

A LIV operator computes output Y from input x via a **mixing matrix T whose entries depend on the input itself**:

```
Y_i = Σ_j T_ij(x) · x_j          (token mixing)
Y    = W · mixed                    (channel mixing, W is a static learned parameter)
```

The token mixing matrix T factorises as:

```
T_ij = C_i · M_ij · B_j
```

where B, C are input-dependent projections (the *featurizer*), M_ij is a **structural matrix** that determines the operator family, and the output before channel mixing is `mixed_i = Σ_j T_ij · V_j`.

**Token mixing types** — the four choices of M_ij and their implications:

**DIAGONAL** — `M_ij = δ_ij`
```
T_ij = C_i · B_i · δ_ij   →   T = diag(C ⊙ B)
mixed_i = (C_i · B_i) · V_i          (each token mixes only with itself)
```
- No cross-token interaction — equivalent to element-wise gating
- Used by FFN / SwiGLU; C is typically set to all-ones, so `T[i,i] = B_i`
- Cost: O(L·d) — no quadratic dependency on sequence length

**LOW_RANK** — `M_ij = 1` (outer product, scaled)
```
T_ij = C_i · B_j / √d     →   T = C Bᵀ / √d    (L×L matrix)
mixed_i = Σ_j T_ij · V_j              (every token attends to every other)
```
- Softmax-normalised → standard attention; without softmax → linear attention
- Used by attention variants SA-1 through SA-4
- Cost: O(L²·d) — quadratic in sequence length; dominant for long sequences

**TOEPLITZ** — `M_ij = K_{i−j}` (convolution kernel)
```
T_ij = C_i · K_{i-j} · B_j           (non-zero only when |i−j| < kernel_size)
mixed_i = Σ_{j: |i-j|<k} T_ij · V_j  (each token mixes within a window)
```
- Shift-equivariant: same kernel K applied at every position
- K is either an explicit learnable parameter (GConv-1) or generated from global context by an MLP (GConv-2)
- Cost: O(L·kernel_size·d) — linear for short kernels, grows with kernel_size

**SEMI_SEPARABLE** — `M_ij = ∏_{k=j}^{i−1} A_k` (state-space recurrence)
```
T_ij = C_i · (∏_{k=j}^{i-1} A_k) · B_j   for j ≤ i   (lower-triangular by construction)
     = 0                                    for j > i
mixed_i = Σ_{j≤i} T_ij · V_j              (state-space: all past tokens, decaying weight)
```
- A_k ∈ (0,1) per step — controls how fast past information decays
- Causally lower-triangular by construction (no masking needed)
- Used by SSM / Mamba variants; equivalent to a recurrent hidden state
- Cost: O(L·d) in sequential form; O(L²·d) in the explicit matrix form used here

The four types trade off **context range** against **sequence-length cost**:

```
DIAGONAL:        no context (each position independent)         — O(L·d)
TOEPLITZ:        local context (within kernel_size window)      — O(L·k·d)
SEMI_SEPARABLE:  full causal context (all past, decaying)       — O(L·d) recurrent
LOW_RANK:        full bidirectional context (all positions)     — O(L²·d)
```

---

**Featurizer classes** define *how* the input x is projected into B, C, S, V. Each class pairs a specific projection strategy with a compatible token mix type. The `internal_dim` may differ from `dim` due to expansion, and is always projected back to `dim` by channel mixing.

**Attention family (token mix: LOW_RANK, internal_dim = dim)**

| Class | Operator | B (key) | C (query) | V (value) | Special |
|---|---|---|---|---|---|
| 1 — SA-1 | Standard MHA | W_K(x) | W_Q(x) | W_V(x) | Full dim for all heads |
| 2 — SA-2 | Conv-pre MHA | W_K(x̃) | W_Q(x̃) | W_V(x̃) | x̃ = depthwise-conv(x) before projection |
| 3 — SA-3 | MQA | W_K(x) → repeated h× | W_Q(x) | W_V(x) → repeated h× | 1 KV head shared across all Q heads |
| 4 — SA-4 | GQA | W_K(x) → repeated 2× | W_Q(x) | W_V(x) → repeated 2× | h/2 KV heads, each serving 2 Q heads |

SA-2 applies a causal depthwise Conv1d (kernel=3) to x before all projections, giving local context awareness inside the attention featurizer itself. SA-3/SA-4 reduce KV parameter count by sharing heads; SA-3 is the extreme (1 KV head), SA-4 is moderate (h/2 KV heads).

**Recurrence family (token mix: SEMI_SEPARABLE)**

| Class | Operator | internal_dim | B | C | S (transition A) | V |
|---|---|---|---|---|---|---|
| 5 — Rec-1 | Mamba-like SSM | dim × 16 | W_B(x) | W_C(x) | sigmoid(W_A(x) + A_log) | W_V(x) |
| 6 — Rec-2 | Compact SSM | dim × 2 | W_B(x) | W_C(x) | sigmoid(W_A(x) + A_log) | W_V(x) |
| 10 — Disc-SSM | S4/Mamba (discretized) | dim × 16 | Δ · W_B(x) | W_C(x) | exp(Δ · A_log) | W_V(x) |
| 11 — CfC | Liquid / CfC | dim × 16 | (1−gate) · W_in(x) | W_C(x) | gate = σ(W_gate(x)) | W_V(x) |

`A_log` is a learnable parameter (negative for Disc-SSM for stability). The key difference:
- Rec-1/2: A is purely input-dependent via `sigmoid(W_A(x) + A_log)`
- Disc-SSM: uses Euler discretization `Δ = softplus(W_dt(x))`, then `A_bar = exp(Δ · A_log)`, `B_bar = Δ · W_B(x)` — matches the S4/Mamba continuous-time parameterization
- CfC: `A = gate`, `B = (1−gate) · g(x)` — convex interpolation between past state and new input; `A + B_coeff = 1` per element

**Convolution family (token mix: TOEPLITZ, internal_dim = dim)**

| Class | Operator | B | C | S (kernel) | V |
|---|---|---|---|---|---|
| 7 — GConv-1 | Short gated conv | sigmoid(W_B(x)) | sigmoid(W_C(x)) | Explicit `nn.Parameter` [dim, 3] | x (raw) |
| 8 — GConv-2 | Long gated conv | sigmoid(W_B(x)) | sigmoid(W_C(x)) | MLP(mean(x)) → [dim, 64] | x (raw) |

GConv-1's kernel S is a fixed learned parameter — the same 3 taps for every input. GConv-2 generates a 64-tap kernel from the **global average** of x via a 2-layer MLP, making it input-dependent (Hyena / implicit-kernel style). V = x directly (no projection) for both, keeping parameter count low.

**Memoryless family (token mix: DIAGONAL, internal_dim = dim × 4)**

| Class | Operator | B | C | V |
|---|---|---|---|---|
| 9 — GMemless | SwiGLU FFN | SiLU(W_gate(x)) | ones (unused) | W_value(x) |

The diagonal T means `mixed_i = B_i · V_i` — a gated element-wise product (SwiGLU). No cross-token interaction. C = 1 so the featurizer effectively contributes only a gate (B) and a value (V). Channel mixing (DENSE) provides the output projection back to dim.

Each LIV class also has a **differential variant** (classes 10–17 in the base pool) that computes `LIV_1(x) − LIV_2(x)` using two independent instances of the same featurizer, increasing expressiveness at roughly 2× the parameter cost.

**Channel mixing** projects the token-mixed output across the feature dimension using a static learned matrix W (paper definition — does **not** depend on input):

```
Y_i^α = Σ_β W_αβ · mixed_i^β          (applied per token position i)
```

W_αβ is a fixed `nn.Parameter` whose structure is determined by the channel mix type:

| W_αβ structure | Type | Effect |
|---|---|---|
| w_α · δ_αβ | DIAGONAL | Per-channel scalar scaling only; no cross-channel interaction; equivalent to a learned per-dimension gain |
| W[α, β] (full matrix) | DENSE | All channels mix freely; most expressive; equivalent to a standard `nn.Linear` without bias |
| block_diag(W_1, …, W_h) | GROUPED | dim split into h equal head-sized groups; each group mixes independently; equivalent to h parallel small dense layers |

The three types trade off expressiveness against parameter count:

```
DIAGONAL:  params = dim                       (cheapest — just a scale vector)
GROUPED:   params = h × (dim/h)² = dim²/h    (intermediate — h small dense blocks)
DENSE:     params = dim²                      (most expensive — full projection)
```

In the genome 5-tuple the channel mix type is axis 3. Common pairings from the paper:
- Attention (LOW_RANK token) → GROUPED (multi-head output projection)
- SSM / Conv (SEMI_SEPARABLE / TOEPLITZ token) → DIAGONAL (channel-independent state)
- FFN / SwiGLU (DIAGONAL token) → DENSE (full output projection)

Each LIV block is encoded in a genome as a **5-tuple**: `(featurizer_class, token_mix_type, channel_mix_type, sparsity_pattern, nonlinearity)`.

---

### 2.2 NSGA-II — Evolutionary Architecture Search

NSGA-II simultaneously minimises three objectives without weighting:

| Objective | Measure | Lower = better |
|---|---|---|
| Quality | val loss / MSE after N steps | ✓ |
| Parameters | total unique parameter count | ✓ |
| KV-cache | inference state size estimate | ✓ |

**Genome encoding** — each genome is a list of *N* `LayerGene` tuples:

```
LayerGene = (liv_class, feat_share_group, feat_share_strategy,
             fg_share_group, fg_share_bitmask)
```

- `feat_share_strategy=2` → two layers in the same group share one featurizer instance (weight tying)
- `fg_share_bitmask` → selectively shares B / C / S / V projection weights between layers

**Evolution loop (one generation)**:

```
1. Evaluate population → (quality, params, kv_cache) per individual
2. Non-dominated sort → assign rank (0 = Pareto front)
3. Crowding distance → secondary sort within each rank
4. Tournament selection → pick parents proportional to rank + distance
5. Crossover (multi-point on layer list) → produce offspring genomes
6. Mutation (random per-gene with prob p) → explore new operators
7. repair() → clamp sharing indices to [0, n_layers−1]
8. Replace population, repeat
```

Default: pop=16, gen=18, mutation_prob=0.10, elitism=2, evolution_steps=500 steps/candidate.

Post-evolution: top-K Pareto-front candidates trained to convergence (20 K steps).

---

### 2.3 Existing TS Structures — DMamba, S-Mamba, iTransformer

All three share the same external interface: input `(B, L, C)` → output `(B, H, C)`.
What differs is **how tokens are formed** and **how many backbones are used**.

#### DMamba (Decomposition + Dual Temporal Flow)

```
(B, L, C)  →  RevIN  →  EMA decomposition
                              │                    │
                         Seasonal              Trend
                       embed (C→D)          embed (C→D)
                              │                    │
                    backbone_seasonal      backbone_trend
                    (L tokens, causal)    (L tokens, causal)
                              │                    │
                       proj (D→C)           proj (D→C)
                       temporal (L→H)       temporal (L→H)
                              └──────── + ──────────┘
                                        │
                                   RevIN denorm
                                        │
                                   (B, H, C)
```

- Backbone processes **L temporal tokens** — causal (past cannot see future)
- Genome split: first N/2 layers → seasonal backbone, last N/2 → trend backbone
- Original operators: seasonal=Mamba, trend=MLP

#### S-Mamba (Two-Stage Variate-First)

```
(B, L, C)  →  transpose  →  (B, C, L)
                  │
           variate_embed (L→D)     → (B, C, D)
                  │
         backbone_variate            ← stage 1: inter-variate correlation
         (C tokens, bidirectional)
                  │
         backbone_temporal           ← stage 2: temporal dependency
         (C tokens, bidirectional)
                  │
          proj (D→H)  →  transpose  →  (B, H, C)
```

- Backbone processes **C variate tokens** — bidirectional
- Genome split: first N/2 → variate backbone, last N/2 → temporal backbone
- Original operators: variate=BiMamba, temporal=FFN

#### iTransformer (Single Variate-First Backbone)

```
(B, L, C)  →  transpose  →  (B, C, L)
                  │
           variate_embed (L→D)     → (B, C, D)
                  │
             backbone
         (C tokens, bidirectional)   ← N stacked blocks, each: cross-variate + per-variate
                  │
          proj (D→H)  →  transpose  →  (B, H, C)
```

- Same variate-first orientation as S-Mamba but **single backbone**, no stage split
- Full genome → one backbone
- Original operators: cross-variate=Attention, per-variate=FFN

**Key differences summary:**

| | DMamba | S-Mamba | iTransformer |
|---|---|---|---|
| Token type | Temporal (L tokens) | Variate (C tokens) | Variate (C tokens) |
| Backbones | 2 (seasonal + trend) | 2 (variate + temporal) | 1 |
| Causality | Causal | Bidirectional | Bidirectional |
| Decomposition | EMA + RevIN | None | None |
| Genome split | First/last half | First/last half | Full genome |

---

### 2.4 TiDAR-TS — Think-in-Diffusion for Time Series

TiDAR-TS adapts the LM TiDAR framework to continuous-valued forecasting. The key idea is to **draft all H forecast steps in parallel** (bidirectional/diffusion) and optionally **refine autoregressively**.

**Sequence layout** — a single forward pass processes `L + H` tokens:

```
Position:  0 ──────── L-1  |  L ──────────── L+H-1
Token:     [x_1 ... x_L]   |  [MASK ... MASK]
Attention: causal           |  full (sees all lookback + all mask tokens)
```

The `TIDAR_HYBRID` sparsity mask enforces this pattern on every LIV block.

**Training — joint loss:**

```
L = 1/(1+α) · [α · L_AR  +  L_Diff]

L_AR   = MSE(ar_head(clean_out[:, :-1]),  x[:, 1:])   ← next-step on lookback
L_Diff = MSE(diff_head(mask_out),         y)            ← forecast from mask
```

**Inference modes:**

| Mode | How | Speed |
|---|---|---|
| Draft (fast) | `diff_head(mask_out)` — single forward pass | O(1) |
| AR-refined (k steps) | Slide window, replace first k draft steps with `ar_head` output | O(k) |
| Full AR (slow) | k = H, all steps refined | O(H) |

Unlike LM TiDAR (discrete tokens, cross-entropy), TiDAR-TS uses **MSE** on continuous values. There is no accept/reject step — the AR refinement simply overwrites draft positions with the AR head's prediction.

---

## 3. Core Idea

Use STAR's evolutionary search (NSGA-2) to search over LIV operators within established time series forecasting structures. Instead of manually choosing "Mamba for X, MLP for Y", let the genome decide.

### What We Search (LIV Genome)
For each component slot, the GA searches over:
- **SSM variants**: Rec-1 (Mamba-like, 16x expansion), Rec-2 (compact, 2x expansion)
- **CfC**: Closed-form continuous-time (new LIV class, from Liquid AI)
- **Convolutions**: GConv-1 (short conv), GConv-2 (long conv / Hyena-style)
- **Attention**: SA-1 (standard), SA-2 (+depthwise), SA-3 (MQA), SA-4 (GQA)
- **MLP/FFN**: GMemless (SwiGLU)
- **Differential variants**: All of the above in differential form

### What We Keep Fixed (Not Searched)
- Model width (d_model)
- Number of layers/blocks
- Training hyperparameters (lr, optimizer, etc.)
- Decomposition strategy (EMA for DMamba-style)

---

## 4. Structural Approaches to Compare

We define **4 structural templates**, each inspired by a SOTA model. Within each template, the LIV-GA searches for the best operator combination.

### Approach A: DMamba-Style (Decomposition + Dual Flow)

```
Input (B, L, C)
    │
    ▼
[RevIN Normalization]
    │
    ▼
[EMA Decomposition] ──────────────────────┐
    │                                      │
    ▼ Seasonal                             ▼ Trend
┌─────────────────┐               ┌──────────────────┐
│ LIV Block (GA)  │               │ LIV Block (GA)   │
│ ┌─────────────┐ │               │ ┌──────────────┐ │
│ │ Slot 1: SSM │ │               │ │ Slot 3: MLP  │ │
│ │ or CfC      │ │               │ │ or Conv      │ │
│ │ or Conv     │ │               │ │ or SSM       │ │
│ │ or Attn     │ │               │ └──────────────┘ │
│ └─────────────┘ │               └──────────────────┘
│ ┌─────────────┐ │                        │
│ │ Slot 2: FFN │ │                        │
│ │ or Conv     │ │                        │
│ │ or SSM      │ │                        │
│ └─────────────┘ │                        │
└─────────────────┘                        │
    │                                      │
    ▼                                      ▼
[Aggregate Seasonal + Trend]
    │
    ▼
[RevIN Denormalize]
    │
    ▼
Output (B, H, C)
```

**Genome structure**: 4 slots (seasonal_temporal, seasonal_channel, trend_temporal, trend_channel)
- Original DMamba: seasonal=Mamba, trend=MLP
- GA may discover: seasonal=CfC, trend=ShortConv (or any combination)

### Approach B: S-Mamba-Style (Variate-First + Temporal)

```
Input (B, L, C)
    │
    ▼
[Linear Tokenization] ── variate tokens (B, C, D)
    │
    ▼
┌──────────────────────────────┐
│ Variate Correlation Block    │
│ ┌──────────────────────────┐ │
│ │ Slot 1: Bidirectional    │ │
│ │ SSM / CfC / Attn / Conv │ │
│ └──────────────────────────┘ │
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│ Temporal Dependencies Block  │
│ ┌──────────────────────────┐ │
│ │ Slot 2: FFN / SSM / Conv │ │
│ └──────────────────────────┘ │
└──────────────────────────────┘
    │
    ▼
[Linear Projection]
    │
    ▼
Output (B, H, C)
```

**Genome structure**: 2 slots per block (variate_mixer, temporal_mixer)
- Original S-Mamba: variate=BiMamba, temporal=FFN
- GA may discover: variate=SA-1 (attention), temporal=Rec-1

### Approach C: iTransformer-Style (Inverted Variate Tokens)

```
Input (B, L, C)
    │
    ▼
[Variate Embedding] ── each variate's L timepoints → D-dim token
    │                    Result: (B, C, D) where C variates are tokens
    ▼
┌──────────────────────────────────┐
│ × N Blocks                       │
│  ┌────────────────────────────┐  │
│  │ Slot 1: Cross-Variate      │  │
│  │ Attn / SSM / CfC / Conv   │  │
│  └────────────────────────────┘  │
│  ┌────────────────────────────┐  │
│  │ Slot 2: Per-Variate FFN    │  │
│  │ GMemless / Conv / SSM      │  │
│  └────────────────────────────┘  │
└──────────────────────────────────┘
    │
    ▼
[Linear Projection per variate]
    │
    ▼
Output (B, H, C)
```

**Genome structure**: 2 slots per block × N blocks
- Original iTransformer: cross_variate=Attention, per_variate=FFN
- GA may discover: cross_variate=CfC, per_variate=GConv-2

### Approach D: Hybrid Decomposition + Inverted (Novel)

Combines DMamba's decomposition with iTransformer's inverted tokenization:

```
Input (B, L, C)
    │
    ▼
[RevIN Normalization]
    │
    ▼
[EMA Decomposition]
    │                          │
    ▼ Seasonal                 ▼ Trend
[Variate Embed]            [Variate Embed]
    │                          │
    ▼                          ▼
┌────────────────┐     ┌────────────────┐
│ × N Blocks     │     │ Linear Head    │
│ Slot 1: Cross  │     │ (Slot 3: GA)   │
│ Slot 2: Per-V  │     └────────────────┘
└────────────────┘             │
    │                          │
    ▼                          ▼
[Aggregate + Denormalize]
    │
    ▼
Output (B, H, C)
```

**Genome structure**: 3+ slots (seasonal_cross, seasonal_per_variate, trend_head)
- Novel: No existing paper uses this exact combination

---

## 5. Extended LIV Classes (Rec-3, Rec-4/CfC)

Two additional recurrence-family operators are implemented in `core/liv.py` under the `include_extended` flag. They extend the standard class pool (1–17) with classes 18–21.

### Class 18 — Rec-3: Discretized SSM (S4-style)

```
Featurizer10 + SEMI_SEPARABLE + DIAGONAL, expansion=16

State update (discretized, Mamba/S4-style):
  Δ_t  = softplus(W_dt(x_t))          — input-dependent step size
  A_bar = exp(Δ_t · A_log)             — discretized state transition (A_log learnable)
  B_bar = Δ_t · W_B(x_t)              — discretized input gate
  C_t  = W_C(x_t)                     — output gate

Key difference from Rec-1/Rec-2: explicit Δt discretization makes the
state transition input-dependent in a physically principled way (ZOH).
```

Genome encoding: class 18 = Rec-3, class 20 = Diff-Rec-3

### Class 19 — Rec-4: CfC / Liquid Neural Network

```
Featurizer11 + SEMI_SEPARABLE + DIAGONAL, expansion=16

State update (closed-form, complementary gating):
  gate = σ(W_gate(x_t))
  A_t  = gate                          — forget: how much state to retain
  B_t  = (1 - gate) · g(x_t)          — input:  how much new input to absorb

Key property: A + B_coefficient = 1 per element (complementary).
This is the LIV implementation of CfC (Closed-form Continuous-time networks,
Liquid AI). No ODE solver needed — the complementary gate IS the closed-form solution.
```

Genome encoding: class 19 = Rec-4 (CfC), class 21 = Diff-Rec-4

### Summary

| Class | Name | Featurizer | Key Mechanism | Differential |
|-------|------|-----------|---------------|-------------|
| 18 | Rec-3 | 10 | Δt-discretized SSM (S4/ZOH) | 20 |
| 19 | Rec-4 (CfC) | 11 | Complementary gate A+B=1 | 21 |

### Why CfC (Rec-4) for Time Series
- Complementary gating naturally models **smooth continuous dynamics** (weather, electricity)
- Unlike Rec-1/Rec-2 (fixed A), the forget gate is fully input-dependent
- Low parameter count (no separate A_log parameter) with strong temporal representation
- Works in both causal (DMamba temporal) and bidirectional (S-Mamba variate) modes

---

## 6. TiDAR Integration (Parallel Prediction)

Adapt TiDAR's "Think in Diffusion, Talk in Autoregression" for time series:

### Time Series TiDAR Adaptation

Instead of diffusion for tokens, use **parallel multi-step prediction**:

```
Standard autoregressive forecasting:
  y_1 → y_2 → y_3 → ... → y_H  (sequential, slow)

TiDAR-style for time series:
  Phase 1 (Think/Draft): Predict all H steps in parallel via denoising
    ŷ_1..H = Denoise(noise, context)
  Phase 2 (Refine): Autoregressive refinement pass
    y_t = AR_refine(ŷ_t, y_{t-1}, context)
```

### Where TiDAR Fits in Our Framework
- **Draft model** (parallel): Uses the LIV backbone found by GA
- **Refine model** (optional AR pass): Lightweight LIV layer for correction
- **Benefit**: Faster inference for long horizons (H=720)

### Structured Attention Mask (adapted from TiDAR)
```
For forecast horizon H, lookback L:
  - Lookback tokens (1..L): fully visible to all (AR-style)
  - Forecast tokens (L+1..L+H):
    Draft phase: masked diffusion (each sees only lookback)
    Refine phase: causal AR (each sees lookback + previous forecasts)
```

---

## 7. NSGA-2 Multi-Objective Search

### Objectives
1. **Primary**: Forecasting quality (MSE on validation set)
2. **Secondary**: Parameter count (fewer = better)
3. **Optional 3rd**: Inference latency (FLOPs or wall-clock)

### Genome Encoding for Time Series

Each structural approach has its own genome format:

```python
# Approach A (DMamba-style): 4 searchable slots
genome_A = {
    "seasonal_temporal": LIV_class_id,  # 1-21 (SA, Rec, GConv, GMemless, CfC, Diff)
    "seasonal_channel": LIV_class_id,
    "trend_temporal": LIV_class_id,
    "trend_channel": LIV_class_id,
    # Plus: sharing strategies (from STAR backbone genome)
}

# Approach B (S-Mamba-style): 2 slots × N blocks
genome_B = {
    "blocks": [
        {"variate_mixer": LIV_class_id, "temporal_mixer": LIV_class_id}
        for _ in range(N)
    ]
}

# Approach C (iTransformer-style): 2 slots × N blocks
genome_C = {
    "blocks": [
        {"cross_variate": LIV_class_id, "per_variate": LIV_class_id}
        for _ in range(N)
    ]
}

# Approach D (Hybrid): combination
genome_D = {
    "seasonal_blocks": [{"cross": LIV_class_id, "per_v": LIV_class_id} for _ in range(N)],
    "trend_head": LIV_class_id,
}
```

### Evolution Protocol
- Population size: 16
- Generations: 12-18
- Per-candidate training: 500-2000 steps (quick eval)
- Post-evolution: Top-8 trained for 10K+ steps
- Selection: NSGA-2 (Pareto front of MSE vs. params)

---

## 8. Experimental Plan

### Datasets

| Dataset | Variates | Length | Granularity | Forecast Horizons |
|---------|----------|--------|-------------|-------------------|
| ETTh1   | 7        | 17420  | Hourly      | 96, 192, 336, 720 |
| ETTh2   | 7        | 17420  | Hourly      | 96, 192, 336, 720 |
| ETTm1   | 7        | 69680  | 15-min      | 96, 192, 336, 720 |
| ETTm2   | 7        | 69680  | 15-min      | 96, 192, 336, 720 |
| Electricity | 321  | 26304  | Hourly      | 96, 192, 336, 720 |
| Traffic | 862      | 17544  | Hourly      | 96, 192, 336, 720 |
| Exchange | 8       | 7588   | Daily       | 96, 192, 336, 720 |

### Baselines to Compare Against

| Model | Type | Year | Key Mechanism |
|-------|------|------|---------------|
| DMamba | Mamba + Decomposition | 2025 | EMA decomp + Mamba seasonal + MLP trend |
| S-Mamba | Mamba | 2024 | BiMamba variate + FFN temporal |
| iTransformer | Transformer | 2024 | Inverted variate tokens + attention |
| PatchTST | Transformer | 2023 | Patched temporal tokens + attention |
| TimesNet | CNN | 2023 | FFT + 2D Conv (Inception) |
| DLinear | Linear | 2023 | Decomposition + Linear |
| Autoformer | Transformer | 2022 | Auto-correlation + decomposition |

### Evaluation Metrics
- **MSE** (Mean Squared Error) - primary
- **MAE** (Mean Absolute Error) - secondary
- **Parameter count** - efficiency
- **Inference time** (ms/sample) - practical

### Experiments

#### Exp 1: Architecture Search (Main Result)
- Run NSGA-2 on all 4 structural approaches (A-D)
- Dataset: ETTh1 for search, validate on all 7 datasets
- Report: Best genome for each approach + Pareto fronts

#### Exp 2: Operator Ablation
- Fix structure (e.g., Approach A), vary operator choices:
  - All-SSM, All-CfC, All-Conv, All-Attention, Mixed (GA-found)
  - Shows GA finds better combos than any single operator type

#### Exp 3: CfC vs SSM Head-to-Head
- Same structure, same slots, only vary SSM ↔ CfC
- Shows when CfC wins (irregular data, continuous dynamics)

#### Exp 4: TiDAR Speedup
- Compare sequential vs. TiDAR-style parallel prediction
- Measure wall-clock time for H=720

#### Exp 5: Cross-Dataset Generalization
- Search on ETTh1, evaluate best genome on all other datasets
- Shows whether discovered architectures transfer

---

## 9. Implementation Plan

### Phase 1: Data Pipeline (Week 1)
- [ ] Download datasets via `download_data.sh`
- [ ] Implement `TimeSeriesDataset` class with:
  - Sliding window sampling
  - Train/val/test split (0.7/0.1/0.2 standard)
  - RevIN normalization
  - Configurable lookback (L) and horizon (H)

### Phase 2: Structural Templates (Week 2-3)
- [ ] Implement Approach A: `DMambaLIV` (decomposition + dual flow)
- [ ] Implement Approach B: `SMambaLIV` (variate-first + temporal)
- [ ] Implement Approach C: `iTransformerLIV` (inverted variate tokens)
- [ ] Implement Approach D: `HybridLIV` (decomposition + inverted)
- [ ] Each template uses LIV blocks from existing `liv.py` as drop-in components

### Phase 3: CfC LIV Operator (Week 3)
- [ ] Add CfC as new LIV class (class 18-21) in `liv.py`
- [ ] Verify CfC works in both causal and non-causal modes
- [ ] Unit test against reference CfC implementation

### Phase 4: Time Series Genome + NSGA-2 (Week 4)
- [ ] Extend `nsga.py` genome encoding for time series slots
- [ ] Fitness function: MSE on validation set + param count
- [ ] Quick-eval protocol: short training → MSE measurement

### Phase 5: TiDAR Adaptation (Week 5)
- [ ] Implement parallel draft prediction head
- [ ] Implement optional AR refinement pass
- [ ] Structured attention mask for forecast tokens

### Phase 6: Experiments + Paper (Week 6-8)
- [ ] Run all experiments (Exp 1-5)
- [ ] Generate tables and figures
- [ ] Write paper

---

## 10. Compute Requirements

### Per Architecture Search (1 dataset, 1 approach)
- Population 16 × Generations 12 = 192 candidate evaluations
- Each candidate: ~500-2000 training steps
- Estimated: 4-8 GPU hours on RTX 4060
- Top-8 full training: ~2-4 GPU hours each

### Total Estimate
- 4 approaches × 1 search dataset = 4 searches = 16-32 GPU hours
- Top genomes full eval on 7 datasets × 4 horizons = ~48-96 GPU hours
- Baselines reproduction: ~24 GPU hours
- **Total: ~100-150 GPU hours (~4-6 days on single RTX 4060)**

---

## 11. Expected Contributions

1. **LIV-TS Framework**: First application of LIV evolutionary search to time series forecasting
2. **CfC as LIV operator**: Extending the LIV option pool with continuous-time dynamics
3. **Structural template comparison**: Systematic comparison of DMamba/S-Mamba/iTransformer structures with searchable operators
4. **TiDAR for time series**: First adaptation of hybrid diffusion-AR inference for forecasting speedup
5. **Discovered architectures**: Novel operator combinations that may outperform hand-designed models

---

## References

- [STAR: Synthesis of Tailored Architectures](https://arxiv.org/abs/2411.17800) (Thomas et al., 2024)
- [TiDAR: Think in Diffusion, Act in Autoregression](https://arxiv.org/abs/...) (NVIDIA, 2025)
- [DMamba: Decomposition-enhanced Mamba](https://arxiv.org/abs/2602.09081) (Chen et al., 2025)
- [S-Mamba: Is Mamba Effective for Time Series?](https://arxiv.org/abs/2403.11144) (Wang et al., 2024)
- [iTransformer: Inverted Transformers](https://arxiv.org/abs/2310.06625) (Liu et al., 2024)
- [CfC: Closed-form Continuous-time Neural Networks](https://www.nature.com/articles/s42256-022-00556-7) (Hasani et al., 2022)
- [TimesNet: Temporal 2D-Variation Modeling](https://arxiv.org/abs/2210.02186) (Wu et al., 2023)


Note: Might exclude hybrid approach, focus on approach 1,2 and 3
Endgoal: Adding TTT(test-time training) upon mistakes during diffusion drafting