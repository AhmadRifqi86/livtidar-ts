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

All three share the same external interface: input `(B, L, C)` → output `(B, H, C)` where B=batch, L=lookback length, C=number of variates, H=forecast horizon. What differs is **how tokens are formed**, **how many backbones are used**, and **what inductive bias is exploited**.

---

#### DMamba — Decomposition + Dual Temporal Flow

**Origin**: D-Mamba (Chen et al., 2025)

**Core insight**: Time series contains two qualitatively different components that benefit from separate treatment. The *seasonal* component holds high-frequency periodic patterns (daily/weekly cycles) best captured by operators with local receptive fields. The *trend* component holds slow low-frequency drift best captured by operators with long-range memory. Separating them via EMA and routing each through a dedicated LIV backbone lets the GA independently optimise the two paths.

```
Input (B, L, C)
       │
  ┌────▼────────────────────────────────────────┐
  │  RevIN normalise (per-instance, per-variate) │  (B, L, C)  →  zero-mean, unit-var
  └────┬────────────────────────────────────────┘
       │
  ┌────▼──────────────────────────────────┐
  │  EMA Decomposition  (learnable α per C) │
  └────┬──────────────────────┬────────────┘
       │ seasonal              │ trend
       │ x - EMA(x)            │ EMA(x)
       │ (B, L, C)             │ (B, L, C)
       │                       │
  ┌────▼─────────┐       ┌─────▼────────┐
  │ embed C → D  │       │ embed C → D  │   Linear projection per time-step
  │ (B, L, D)    │       │ (B, L, D)    │
  └────┬─────────┘       └─────┬────────┘
       │                       │
  ┌────▼──────────────┐  ┌─────▼─────────────┐
  │ backbone_seasonal  │  │ backbone_trend     │   ← LIV genome layers 0..N/2-1
  │ N/2 LIV blocks     │  │ N/2 LIV blocks     │   ← LIV genome layers N/2..N-1
  │ L temporal tokens  │  │ L temporal tokens  │
  │ causal masking     │  │ causal masking     │
  │ (B, L, D)          │  │ (B, L, D)          │
  └────┬──────────────┘  └─────┬──────────────┘
       │                       │
  ┌────▼─────────┐       ┌─────▼────────┐
  │ proj D → C   │       │ proj D → C   │   (B, L, C)
  │ temporal L→H │       │ temporal L→H │   Linear: maps lookback length to horizon
  │ (B, H, C)    │       │ (B, H, C)    │
  └────┬─────────┘       └─────┬────────┘
       │                       │
       └──────────┬────────────┘
                  │  sum
             ┌────▼──────────────────────────────┐
             │  RevIN denormalise                  │  restore original scale
             └────┬──────────────────────────────┘
                  │
            Output (B, H, C)
```

**Key properties**:
- Tokens are **L temporal positions** — each token represents one time-step across all D features
- Processing is **causal**: time-step t can only attend to steps 0..t (no future leakage)
- The EMA decomposition is **learnable**: α per variate trained end-to-end
- **Genome split**: layers 0..N/2-1 control seasonal path; layers N/2..N-1 control trend path
- Original paper operators: seasonal = Mamba (SSM), trend = MLP
- GA finding (Exp 1, ETTh1 H=96): seasonal = GConv-1, trend = Diff-Attention variants

**Weakness**: Cannot model cross-variate correlations — each time-step token mixes all C variates into a single D-dim vector before the backbone, so inter-variate structure is not explicitly captured.

---

#### S-Mamba — Two-Stage Variate-First Pipeline

**Origin**: S-Mamba (Wang et al., 2024)

**Core insight**: Multivariate forecasting requires two distinct types of modelling: (1) *inter-variate correlation* — which variables move together, lead or lag each other; (2) *temporal dependency* — how each variable evolves over time. S-Mamba handles them in sequence: a first backbone mixes information across the C variate tokens to build a correlated representation, then a second backbone refines temporal dependencies within those C tokens.

```
Input (B, L, C)
       │
  ┌────▼──────────────────────────────────┐
  │  Transpose: (B, L, C) → (B, C, L)     │  variates become the "sequence"
  └────┬──────────────────────────────────┘
       │
  ┌────▼─────────────────────────────────────────────────────┐
  │  variate_embed:  Linear(L → D)  applied independently     │
  │  each variate's L-length history → single D-dim token     │
  │  (B, C, L)  →  (B, C, D)                                 │
  └────┬─────────────────────────────────────────────────────┘
       │  C tokens, each of dimension D
       │
  ┌────▼────────────────────────────────────────────────────────────────────┐
  │  Stage 1 — backbone_variate  (LIV genome layers 0..N/2-1)               │
  │                                                                          │
  │   ┌───────────────────────────────────────────────────────────────────┐ │
  │   │  LIV block i  (bidirectional, processes C variate tokens)         │ │
  │   │  token_mix: cross-variate operator (SSM / Conv / Attn / FFN)      │ │
  │   │  channel_mix: per-dimension dense projection                      │ │
  │   │  (B, C, D) → (B, C, D)                                           │ │
  │   └───────────────────────────────────────────────────────────────────┘ │
  │   × N/2 blocks                                                           │
  │                                                                          │
  │  Purpose: build correlated variate representations                       │
  │  each variate token now "knows about" other variates                     │
  └────┬────────────────────────────────────────────────────────────────────┘
       │  (B, C, D)
       │
  ┌────▼────────────────────────────────────────────────────────────────────┐
  │  Stage 2 — backbone_temporal  (LIV genome layers N/2..N-1)              │
  │                                                                          │
  │   ┌───────────────────────────────────────────────────────────────────┐ │
  │   │  LIV block i  (bidirectional, still processes C variate tokens)   │ │
  │   │  token_mix: temporal refinement operator (FFN / Conv / SSM)       │ │
  │   │  channel_mix: per-dimension dense projection                      │ │
  │   │  (B, C, D) → (B, C, D)                                           │ │
  │   └───────────────────────────────────────────────────────────────────┘ │
  │   × N/2 blocks                                                           │
  │                                                                          │
  │  Purpose: refine temporal structure in the correlated D-dim space        │
  └────┬────────────────────────────────────────────────────────────────────┘
       │  (B, C, D)
       │
  ┌────▼──────────────────────────────────────────────────────┐
  │  proj: Linear(D → H) per variate token                    │
  │  (B, C, D)  →  (B, C, H)                                 │
  └────┬──────────────────────────────────────────────────────┘
       │
  ┌────▼──────────────────────────────────────────────────────┐
  │  Transpose: (B, C, H)  →  (B, H, C)                      │
  └────┬──────────────────────────────────────────────────────┘
       │
  Output (B, H, C)
```

**Key properties**:
- Tokens are **C variates** — each token represents one variate's entire history as a D-dim embedding
- Both stages are **bidirectional**: no causal constraint (future variate context is helpful for correlation)
- **Genome split**: layers 0..N/2-1 → variate backbone (stage 1); layers N/2..N-1 → temporal backbone (stage 2)
- Original paper operators: variate = BiMamba (bidirectional SSM), temporal = FFN
- GA finding (Exp 1, ETTh1 H=96): variate = GConv-1 + Rec-2, temporal = GConv-1 + GMemless

**Weakness**: Stage 2 still processes C variate tokens, not L temporal tokens — temporal dependencies are captured indirectly through the D-dimensional embedding, not directly over the time axis.

---

#### iTransformer — Inverted Variate Token Transformer

**Origin**: iTransformer (Liu et al., 2024)

**Core insight**: The original Transformer for time series applies attention over L temporal positions — but this conflates temporal modelling with multivariate modelling and scales as O(L²). Inverting the token definition so each variate becomes a token (rather than each time-step) enables: (1) cross-variate attention that directly models which variates are correlated; (2) O(C²) cost instead of O(L²), which is much cheaper when C << L; (3) per-variate FFN that models each variate's own feature transformation independently.

```
Input (B, L, C)
       │
  ┌────▼──────────────────────────────────┐
  │  Transpose: (B, L, C) → (B, C, L)     │  variates become the sequence dimension
  └────┬──────────────────────────────────┘
       │
  ┌────▼─────────────────────────────────────────────────────┐
  │  variate_embed:  Linear(L → D)  applied per variate       │
  │  compress each variate's full L-step history into D dims  │
  │  (B, C, L)  →  (B, C, D)                                 │
  └────┬─────────────────────────────────────────────────────┘
       │  C tokens, each = one variate's history summary
       │
  ┌────▼────────────────────────────────────────────────────────────────────┐
  │  × N stacked LIV blocks  (LIV genome layers 0..N-1, full genome)        │
  │                                                                          │
  │   ┌───────────────────────────────────────────────────────────────────┐ │
  │   │  token_mix  — cross-variate operator                              │ │
  │   │  mixes information across C variate tokens                        │ │
  │   │  operator choices: SA-1 (attention) / GConv-1 / SSM / GMemless   │ │
  │   │  (B, C, D) → (B, C, D)                                           │ │
  │   └───────────────────────────────────────────────────────────────────┘ │
  │                         │                                                │
  │   ┌───────────────────────────────────────────────────────────────────┐ │
  │   │  channel_mix  — per-variate operator                              │ │
  │   │  applied independently to each variate's D-dim representation     │ │
  │   │  operator choices: GMemless (FFN) / GConv-1 / SA                  │ │
  │   │  (B, C, D) → (B, C, D)                                           │ │
  │   └───────────────────────────────────────────────────────────────────┘ │
  │                                                                          │
  └────┬────────────────────────────────────────────────────────────────────┘
       │  (B, C, D)
       │
  ┌────▼──────────────────────────────────────────────────────┐
  │  proj: Linear(D → H) per variate token                    │
  │  directly predicts forecast horizon from D-dim embedding  │
  │  (B, C, D)  →  (B, C, H)                                 │
  └────┬──────────────────────────────────────────────────────┘
       │
  ┌────▼──────────────────────────────────────────────────────┐
  │  Transpose: (B, C, H)  →  (B, H, C)                      │
  └────┬──────────────────────────────────────────────────────┘
       │
  Output (B, H, C)
```

**Key properties**:
- Tokens are **C variates** — one token per variate, embedding its entire L-step history
- Single backbone, **no stage split**: N blocks each alternate between cross-variate (token_mix) and per-variate (channel_mix)
- **Bidirectional**: all C variate tokens attend to each other freely
- **Full genome → one backbone**: all N genome layers belong to the same backbone; even-numbered layers are token_mix slots, odd-numbered are channel_mix slots (or alternating by block)
- Original paper operators: cross-variate = SA-1 (attention), per-variate = GMemless (FFN)
- GA finding (Exp 1, ETTh1 H=96): cross-variate = **GConv-1**, per-variate = GMemless — GA replaced attention with a 3-tap convolution over variate indices, which is sufficient for 7 correlated variates at short horizons

**Strength**: Most parameter-efficient for small C; naturally captures cross-variate correlations; per-variate FFN allows each variate to have its own feature transform.

---

**Key differences summary:**

| | DMamba | S-Mamba | iTransformer |
|---|---|---|---|
| Token type | Temporal (L tokens) | Variate (C tokens) | Variate (C tokens) |
| Backbones | 2 (seasonal + trend) | 2 (variate + temporal) | 1 |
| Causality | Causal | Bidirectional | Bidirectional |
| Decomposition | EMA + RevIN | None | None |
| Genome split | First/last half | First/last half | Full genome |
| Best at | Long horizons, smooth trends | Mixed variate + temporal | Short horizons, many variates |
| Weakness | No cross-variate modelling | Indirect temporal axis | No explicit temporal structure |

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

### Expected Operator Winners per Dataset

Based on the inductive biases of each operator family and dataset characteristics.
`H=S` = short horizons (96, 192); `H=L` = long horizons (336, 720).

| Dataset | Horizon | Expected Winner | Reason |
|---------|---------|----------------|--------|
| ETTh1 / ETTh2 | H=S | GConv-1 (Conv) | Regular hourly sampling; local 3-tap window captures daily seasonality |
| ETTh1 / ETTh2 | H=L | SSM (Rec-1) | Long-range memory needed; 3-tap conv cannot reach weekly dependencies |
| ETTm1 / ETTm2 | H=S | GConv-1 (Conv) | 15-min granularity; seasonal patterns still within short local window |
| ETTm1 / ETTm2 | H=L | SSM (Rec-1) | Same reasoning as ETTh at long horizon |
| Electricity | Any | Attention + SSM | 321 variates; cross-variate correlation dominates; SSM for temporal |
| Traffic | Any | SSM (Rec-1) | 862 variates; complex multi-scale spatial + temporal patterns; long-range memory critical |
| Exchange | Any | CfC (Rec-4) | Daily financial rates; non-stationary regime shifts; CfC gate adapts to irregular dynamics |

> **Empirical note (ETTh1 H=96, dmamba):** Exp 1 confirmed GConv-1 dominant in seasonal,
> Attention in trend — consistent with the short-horizon prediction above. CfC was not
> selected, likely due to OOM during 50-step search (internal_dim=16×256=4096 too large
> at batch=16). Proper CfC evaluation requires Exp 3 with dedicated training budget.

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

### Published Baseline Results (Reference)

Multivariate forecasting MSE / MAE from each paper's main result table.
`—` = dataset not reported in that paper's benchmark.
Numbers from paper knowledge; verify against original PDFs or `thuml/Time-Series-Library` before claiming exact reproduction.

#### ETTh1 (7 variates, hourly)

| Model | H=96 MSE/MAE | H=192 MSE/MAE | H=336 MSE/MAE | H=720 MSE/MAE |
|-------|-------------|--------------|--------------|--------------|
| DLinear | 0.386 / 0.400 | 0.437 / 0.432 | 0.481 / 0.459 | 0.526 / 0.487 |
| PatchTST | 0.370 / 0.400 | 0.413 / 0.429 | 0.422 / 0.440 | 0.447 / 0.468 |
| TimesNet | 0.384 / 0.402 | 0.436 / 0.429 | 0.491 / 0.469 | 0.521 / 0.500 |
| Autoformer | 0.449 / 0.459 | 0.500 / 0.488 | 0.521 / 0.496 | 0.514 / 0.512 |
| S-Mamba | 0.449 / 0.441 | 0.489 / 0.468 | 0.512 / 0.487 | 0.522 / 0.505 |
| iTransformer | 0.454 / 0.447 | 0.483 / 0.463 | 0.501 / 0.482 | 0.506 / 0.507 |
| DMamba | 0.374 / 0.397 | 0.425 / 0.430 | 0.462 / 0.455 | 0.487 / 0.479 |

#### ETTh2 (7 variates, hourly)

| Model | H=96 MSE/MAE | H=192 MSE/MAE | H=336 MSE/MAE | H=720 MSE/MAE |
|-------|-------------|--------------|--------------|--------------|
| DLinear | 0.333 / 0.387 | 0.477 / 0.476 | 0.594 / 0.541 | 0.831 / 0.657 |
| PatchTST | 0.302 / 0.348 | 0.388 / 0.400 | 0.426 / 0.433 | 0.431 / 0.446 |
| TimesNet | 0.340 / 0.374 | 0.402 / 0.414 | 0.452 / 0.452 | 0.462 / 0.468 |
| Autoformer | 0.358 / 0.397 | 0.456 / 0.452 | 0.482 / 0.486 | 0.515 / 0.511 |
| S-Mamba | 0.293 / 0.345 | 0.375 / 0.396 | 0.416 / 0.428 | 0.421 / 0.444 |
| iTransformer | 0.297 / 0.349 | 0.380 / 0.400 | 0.428 / 0.432 | 0.427 / 0.445 |
| DMamba | 0.285 / 0.337 | 0.361 / 0.392 | 0.401 / 0.425 | 0.412 / 0.438 |

#### ETTm1 (7 variates, 15-min)

| Model | H=96 MSE/MAE | H=192 MSE/MAE | H=336 MSE/MAE | H=720 MSE/MAE |
|-------|-------------|--------------|--------------|--------------|
| DLinear | 0.299 / 0.343 | 0.335 / 0.365 | 0.369 / 0.386 | 0.425 / 0.421 |
| PatchTST | 0.293 / 0.346 | 0.333 / 0.370 | 0.369 / 0.392 | 0.416 / 0.420 |
| TimesNet | 0.338 / 0.375 | 0.374 / 0.387 | 0.410 / 0.411 | 0.478 / 0.450 |
| Autoformer | 0.505 / 0.475 | 0.522 / 0.459 | 0.496 / 0.487 | 0.543 / 0.490 |
| S-Mamba | 0.324 / 0.361 | 0.370 / 0.387 | 0.410 / 0.411 | 0.476 / 0.451 |
| iTransformer | 0.334 / 0.368 | 0.377 / 0.391 | 0.426 / 0.420 | 0.491 / 0.459 |
| DMamba | 0.299 / 0.345 | 0.340 / 0.373 | 0.376 / 0.394 | 0.437 / 0.431 |

#### ETTm2 (7 variates, 15-min)

| Model | H=96 MSE/MAE | H=192 MSE/MAE | H=336 MSE/MAE | H=720 MSE/MAE |
|-------|-------------|--------------|--------------|--------------|
| DLinear | 0.167 / 0.260 | 0.224 / 0.303 | 0.281 / 0.342 | 0.397 / 0.421 |
| PatchTST | 0.166 / 0.256 | 0.223 / 0.296 | 0.274 / 0.329 | 0.362 / 0.385 |
| TimesNet | 0.187 / 0.267 | 0.249 / 0.309 | 0.321 / 0.351 | 0.408 / 0.403 |
| Autoformer | 0.255 / 0.339 | 0.281 / 0.340 | 0.339 / 0.372 | 0.422 / 0.419 |
| S-Mamba | 0.175 / 0.261 | 0.241 / 0.302 | 0.298 / 0.342 | 0.393 / 0.397 |
| iTransformer | 0.180 / 0.264 | 0.250 / 0.309 | 0.311 / 0.348 | 0.405 / 0.401 |
| DMamba | 0.163 / 0.252 | 0.223 / 0.294 | 0.276 / 0.334 | 0.354 / 0.388 |

#### Electricity (321 variates, hourly)

| Model | H=96 MSE/MAE | H=192 MSE/MAE | H=336 MSE/MAE | H=720 MSE/MAE |
|-------|-------------|--------------|--------------|--------------|
| DLinear | 0.197 / 0.282 | 0.196 / 0.285 | 0.209 / 0.301 | 0.245 / 0.333 |
| PatchTST | 0.195 / 0.285 | 0.196 / 0.285 | 0.212 / 0.298 | 0.250 / 0.323 |
| TimesNet | 0.168 / 0.272 | 0.184 / 0.289 | 0.198 / 0.300 | 0.220 / 0.320 |
| Autoformer | 0.201 / 0.317 | 0.222 / 0.334 | 0.231 / 0.338 | 0.254 / 0.361 |
| S-Mamba | 0.152 / 0.244 | 0.166 / 0.257 | 0.183 / 0.272 | 0.218 / 0.311 |
| iTransformer | 0.148 / 0.240 | 0.162 / 0.253 | 0.178 / 0.269 | 0.225 / 0.317 |
| DMamba | 0.143 / 0.237 | 0.157 / 0.250 | 0.174 / 0.267 | 0.210 / 0.306 |

#### Traffic (862 variates, hourly)

| Model | H=96 MSE/MAE | H=192 MSE/MAE | H=336 MSE/MAE | H=720 MSE/MAE |
|-------|-------------|--------------|--------------|--------------|
| DLinear | 0.410 / 0.282 | 0.423 / 0.287 | 0.436 / 0.296 | 0.466 / 0.315 |
| PatchTST | 0.360 / 0.249 | 0.379 / 0.256 | 0.392 / 0.264 | 0.432 / 0.286 |
| TimesNet | 0.593 / 0.321 | 0.617 / 0.336 | 0.629 / 0.336 | 0.640 / 0.350 |
| Autoformer | 0.613 / 0.388 | 0.623 / 0.396 | 0.739 / 0.428 | 0.578 / 0.359 |
| S-Mamba | 0.387 / 0.263 | 0.407 / 0.272 | 0.424 / 0.280 | 0.459 / 0.299 |
| iTransformer | 0.395 / 0.268 | 0.417 / 0.276 | 0.433 / 0.283 | 0.467 / 0.302 |
| DMamba | 0.381 / 0.261 | 0.399 / 0.269 | 0.414 / 0.276 | 0.448 / 0.295 |

#### Exchange (8 variates, daily)

Only DLinear, iTransformer, and DMamba report Exchange in their main tables.
PatchTST, S-Mamba, TimesNet, and Autoformer do not include Exchange.

| Model | H=96 MSE/MAE | H=192 MSE/MAE | H=336 MSE/MAE | H=720 MSE/MAE |
|-------|-------------|--------------|--------------|--------------|
| DLinear | 0.088 / 0.218 | 0.176 / 0.315 | 0.313 / 0.427 | 0.839 / 0.695 |
| iTransformer | 0.086 / 0.206 | 0.177 / 0.299 | 0.331 / 0.414 | 0.847 / 0.670 |
| DMamba | 0.083 / 0.202 | 0.172 / 0.293 | 0.322 / 0.408 | 0.825 / 0.658 |

---

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

### Current Code Status

| Component | Status | Notes |
|-----------|--------|-------|
| `core/liv.py` — LIV framework | ✅ Done | All featurizers, token/channel mixing, static W |
| `core/nsga.py` — NSGA-II | ✅ Done | 3+1 objectives, `--include_extended`, `--measure_latency` |
| `core/modeldef.py` — TS models | ✅ Done | DMamba / S-Mamba / iTransformer + RevIN + EMA |
| `core/tidar.py` — TiDAR-TS | ✅ Done | RevIN, draft + AR refinement, joint loss |
| `src/dataload.py` — data pipeline | ✅ Done | All 7 datasets, sliding window, standard splits |
| `src/train.py` — training pipeline | ✅ Done | ts-evolve/train/both + ts-tidar-* modes |
| Rec-3 / Rec-4 (CfC) — classes 18-21 | ✅ Done | `--include_extended` wired to all TS parsers |
| Baseline training scripts | ❌ Pending | DMamba, S-Mamba, iTransformer, PatchTST, DLinear |
| Experiment runner scripts | ❌ Pending | Exp 1-5, result collection, Pareto plot generation |

---

### Timeline (Remaining Work)

> Note: the experiment list (§8) is volatile — new experiments may be added as results emerge. The timeline below covers what is currently planned; buffer weeks absorb additions.

```
Week 1  ── Baseline Scripts
            Implement thin wrappers to train + evaluate baselines on all datasets:
            DMamba, S-Mamba, iTransformer (using STARLIVTSModel with fixed genomes),
            PatchTST, DLinear (external implementations or re-implemented).
            Target: reproduce reported MSE numbers ±2% on ETTh1.

Week 2  ── Exp 1: Architecture Search (main result)
            Run ts-evolve / ts-tidar-evolve on ETTh1 for all 3 structures
            (dmamba, smamba, itransformer). Pop=16, Gen=12, 500 steps/candidate.
            Log Pareto fronts (MSE vs params). Identify top-8 per approach.

Week 3  ── Exp 1 cont.: Full Training of Top Genomes
            Train top-8 per approach for 10K+ steps on ETTh1.
            Evaluate on ETTh1, ETTh2, ETTm1, ETTm2 (standard TS benchmarks).

Week 4  ── Exp 2 + Exp 3
            Exp 2: Operator ablation — fix Approach A, manually set all-SSM /
                   all-CfC / all-Conv / all-Attention and compare vs GA-found mix.
            Exp 3: CfC vs SSM head-to-head — same structure, swap Rec-1 ↔ Rec-4,
                   focus on irregular-interval datasets (Weather, ILI).

Week 5  ── Exp 4 + Exp 5
            Exp 4: TiDAR speedup — compare draft (ar_steps=0) vs AR-refined
                   (ar_steps=k) wall-clock time for H=720 on Traffic.
            Exp 5: Cross-dataset generalization — take best genome from ETTh1
                   search, evaluate zero-shot on Electricity, Traffic, Exchange.

Week 6  ── Buffer / Additional Experiments
            Handle any new experiments added during Weeks 2-5.
            Re-run failed/inconclusive cases. Collect all results into tables.

Week 7  ── Analysis + Figures
            Pareto front plots, operator-frequency heatmaps, speedup curves,
            ablation bar charts, cross-dataset transfer heatmap.

Week 8  ── Paper Writing
            Write §1 intro, §3 method, §4 experiments, §5 conclusion.
            Sections §2 (background) and §6 (related work) drafted in parallel.
```

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

---

## 12. How To Run

### Exp 1 — Architecture Search (Full)

Searches all 3 structures × 4 horizons on ETTh1 with the full operator pool.
`--evolution_steps 300` is the recommended balance between ranking quality and compute time
(~21 hrs total on RTX 4060 8GB). Use `--evolution_steps 500` for higher ranking confidence
if time allows.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.exp1_search \
    --dataset ETTh1 \
    --all_horizons \
    --all_structures \
    --include_extended \
    --evolution_steps 300 \
    --batch_size 16 \
    --full_train_steps 10000 \
    --top_k 8 \
    --out_dir exp1_results
```

**Flags explained:**

| Flag | Value | Reason |
|------|-------|--------|
| `--all_horizons` | — | Runs H=96, 192, 336, 720 |
| `--all_structures` | — | Runs dmamba, smamba, itransformer |
| `--include_extended` | — | Adds Rec-3/Rec-4 (CfC, classes 18-21) to search pool |
| `--evolution_steps` | 300 | Steps per candidate during NSGA-II search (ranking proxy) |
| `--batch_size` | 16 | Required for 8GB VRAM; prevents OOM on large candidates |
| `--full_train_steps` | 10000 | Steps for post-evolution full training of top-K genomes |
| `--top_k` | 8 | Number of Pareto-front candidates to fully train |
| `PYTORCH_CUDA_ALLOC_CONF` | expandable_segments:True | Reduces CUDA memory fragmentation |

**Output layout:**
```
exp1_results/
  ETTh1_H96_dmamba/
    pareto_front.json        # Pareto-optimal genomes + search-phase fitness
    ts_candidate_1.pt        # fully-trained model checkpoint (best genome)
    ts_candidate_2.pt
    ...
    gen_01.json .. gen_N.json  # per-generation population snapshots
    final.json               # full final population with ranks
  ETTh1_H96_smamba/
  ETTh1_H96_itransformer/
  ETTh1_H192_dmamba/
  ...
  summary.json               # MSE table across all structures and horizons
```

### Exp 1 — Dry Run (pipeline check, no training)

```bash
python -m src.exp1_search \
    --dataset ETTh1 \
    --pred_len 96 \
    --structure dmamba \
    --dry_run
```

### Exp 1 — Quick Smoke Test (real training, tiny scale)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.exp1_search \
    --dataset ETTh1 \
    --pred_len 96 \
    --structure dmamba \
    --include_extended \
    --evolution_steps 50 \
    --batch_size 16 \
    --full_train_steps 500 \
    --top_k 3
```

### Exp 1 — Full Run

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.exp1_search \
    --all_horizons \
    --all_structures \
    --include_extended \
    --evolution_steps 300 \
    --batch_size 16 \
    --full_train_steps 50000 \
    --max_params 7000000 \
    --top_k 5 \
    --out_dir exp1_results
```

---

### Exp 2 — Dry Run (pipeline check, no training)

```bash
python -m src.exp2_ablation \
    --dataset ETTh1 --pred_len 96 \
    --structure itransformer \
    --train_steps 2 --batch_size 16
```

### Exp 2 — Quick Smoke Test

```bash
python -m src.exp2_ablation \
    --dataset ETTh1 --pred_len 96 \
    --structure itransformer \
    --base_genome exp1_results/ETTh1_H96_itransformer/ts_candidate_1.pt \
    --train_steps 500 --batch_size 16 \
    --out_dir exp2_results_smoke
```

> Note: avoid `--structure dmamba` for smoke tests — ALL_SSM on L=96 temporal tokens causes OOM (~9.6 GB).
> Use `itransformer` or `smamba` (C=7 variate tokens, safe).

### Exp 2 — Full Run

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.exp2_ablation \
    --all_datasets --all_horizons \
    --structure itransformer \
    --base_genome exp1_results/ETTh1_H96_itransformer/ts_candidate_1.pt \
    --train_steps 50000 --batch_size 16 \
    --out_dir exp2_results
```

---

### Exp 3 — Dry Run (pipeline check, no training)

```bash
python -m src.exp3_cfc_vs_ssm \
    --dataset ETTh1 --pred_len 96 \
    --structure itransformer \
    --train_steps 2 --batch_size 16
```

### Exp 3 — Quick Smoke Test

```bash
python -m src.exp3_cfc_vs_ssm \
    --dataset ETTh1 --pred_len 96 \
    --structure itransformer \
    --base_genome exp1_results/ETTh1_H96_itransformer/ts_candidate_1.pt \
    --train_steps 500 --batch_size 16 \
    --out_dir exp3_results_smoke
```

> Trains SSM_Rec1 and CfC_Rec4 variants for 500 steps each, prints winner.
> Use `--irregular_focus` to target Weather + ILI (key hypothesis datasets).

### Exp 3 — Full Run

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.exp3_cfc_vs_ssm \
    --all_datasets --all_horizons \
    --structure itransformer \
    --base_genome exp1_results/ETTh1_H96_itransformer/ts_candidate_1.pt \
    --train_steps 50000 --batch_size 16 \
    --out_dir exp3_results
```

> 7 datasets × 4 horizons × 2 variants = 56 training runs total.

---

### Exp 4 — Dry Run (pipeline check, no training)

```bash
python -m src.exp4_speedup \
    --dataset ETTh1 --pred_len 96 \
    --train_steps 2 --batch_size 16
```

### Exp 4 — Quick Smoke Test

```bash
python -m src.exp4_speedup \
    --dataset ETTh1 --pred_len 96 \
    --checkpoint exp1_results/ETTh1_H96_itransformer/tidar_ts_candidate_1.pt \
    --batch_size 16 \
    --out_dir exp4_results_smoke
```

> Uses a trained TiDAR-TS checkpoint so no training is needed; just times draft / partial-AR / full-AR modes.

### Exp 4 — Full Run

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.exp4_speedup \
    --dataset ETTh1 --all_horizons \
    --checkpoint exp1_results/ETTh1_H96_itransformer/tidar_ts_candidate_1.pt \
    --batch_size 16 \
    --out_dir exp4_results
```

> Run once per dataset; repeat for ETTh2, ETTm1, ETTm2 by changing `--dataset`.

---

### Exp 5 — Dry Run (pipeline check, no training)

```bash
python -m src.exp5_generalization \
    --source_dataset ETTh1 --target_dataset ETTh2 \
    --pred_len 96 --structure itransformer \
    --train_steps 2 --batch_size 16
```

### Exp 5 — Quick Smoke Test

```bash
python -m src.exp5_generalization \
    --source_dataset ETTh1 --target_dataset ETTh2 \
    --pred_len 96 --structure itransformer \
    --base_genome exp1_results/ETTh1_H96_itransformer/ts_candidate_1.pt \
    --train_steps 500 --batch_size 16 \
    --out_dir exp5_results_smoke
```

### Exp 5 — Full Run

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.exp5_generalization \
    --all_transfers \
    --all_horizons \
    --structure itransformer \
    --base_genome exp1_results/ETTh1_H96_itransformer/ts_candidate_1.pt \
    --train_steps 50000 --batch_size 16 \
    --out_dir exp5_results
```

> Tests cross-dataset transfer: genome found on source trained/evaluated on target.

---

## 13. Expected Experimental Results

### Exp 1: Architecture Search

**Setup**: NSGA-II on ETTh1, all 3 structures (dmamba / smamba / itransformer), all 4 horizons, evolution_steps=300, pop=16, gen=18.

**Expected Pareto front characteristics**:
- 5–15 non-dominated solutions trading quality vs parameter count
- Best quality genome: ~1–5M params, MSE competitive with DMamba baseline (~0.37–0.42 at H=96)
- Most parameter-efficient genome: ~500K–1M params, MSE ~5–10% above the quality frontier

**Expected operator patterns per horizon**

| Horizon | Seasonal / Variate component | Trend / Temporal component |
|---------|------------------------------|---------------------------|
| H=96, 192 | GConv-1 (class 7) dominant | SA-1 or SA-4 (attention) |
| H=336, 720 | Rec-1 (SSM, class 5) emerges | GMemless or Rec-1 |
| All horizons | CfC (class 19) unlikely to appear | OOM penalty during 300-step proxy biases against large-internal_dim operators |

> **Empirical note (Exp 1 pre-run, dmamba ETTh1 H=96):** GConv-1 dominates seasonal in
> all 9 Pareto-front candidates. No CfC selected. Best candidate `[7,7,1,4]` = 612K params,
> quality=0.996 (proxy MSE). Consistent with short-horizon prediction above.

**Expected structure comparison**:
- `dmamba`: best on ETTh datasets — explicit seasonal/trend decomposition well-matched to hourly data
- `itransformer`: best on high-variate datasets (Electricity 321, Traffic 862) — cross-variate attention
- `smamba`: competitive middle ground; two-stage pipeline adds flexibility but also complexity

**What would indicate a successful search**:
- Pareto front contains architecturally diverse solutions (not all same operator family)
- Best genome outperforms uniform ALL_SSM baseline from Exp 2 (validates GA search cost)
- Different optimal genomes emerge per horizon (validates horizon-specific search need)

---

### Exp 2: Operator Ablation

**Setup**: Fix `structure=dmamba`, compare ALL_SSM / ALL_CONV / ALL_ATTN / ALL_FFN / MIXED_GA / H96_FIXED on ETTh1.

**Expected ranking (lower MSE = better)**

Short horizons (H=96, 192):
```
MIXED_GA < ALL_CONV ≈ ALL_ATTN < ALL_SSM < ALL_FFN
```
- GConv-1 confirmed dominant for ETTh1 short-horizon (Exp 1 result)
- Attention works well for trend; GA discovers the Conv+Attn combination
- FFN has no token mixing — worst baseline, barely better than predicting mean

Long horizons (H=336, 720):
```
MIXED_GA < ALL_SSM < ALL_ATTN < ALL_CONV < ALL_FFN
```
- SSM memory becomes critical; 3-tap GConv cannot reach weekly dependencies
- Attention degrades with very long-range forecast targets
- GA should discover SSM-dominant genomes when searched at long horizons

**H96_FIXED variant** (short-horizon genome retrained at H=720):
- At H=96: ≈ MIXED_GA (same genome, same horizon it was optimised for)
- At H=720: degrades vs ALL_SSM → confirms GA architecture is horizon-specific
- Directly answers: "Does the operator combination generalise across horizons?"

**What would invalidate the experiment**:
- ALL_FFN beats MIXED_GA → GA found something worse than no token mixing (bug or training failure)
- All variants tie → operator choice is irrelevant for this dataset/budget

---

### Exp 3: CfC vs SSM Head-to-Head

**Setup**: Fix structure and slots; swap all recurrent layers between Rec-1 (SSM) and Rec-4 (CfC). Keep Conv/Attn/FFN layers unchanged.

**Expected results per dataset**

| Dataset | Expected Winner | Key Reason |
|---------|----------------|-----------|
| ETTh1 / ETTh2 H=S | SSM | Regular hourly cycles; slow decay captures daily periodicity |
| ETTh1 / ETTh2 H=L | SSM | Long-range trend; SSM accumulated state better than CfC reset |
| ETTm1 / ETTm2 | SSM | Fine-grained 15-min data; smooth dynamics favour SSM |
| Electricity | Tie or SSM | Regular load profiles; SSM marginal advantage |
| Traffic | SSM | Complex spatial patterns need long-range memory |
| Exchange | **CfC** | Daily financial rates; non-stationary regime shifts; CfC gate resets on policy changes |

**Why CfC wins on Exchange**: The complementary gate `A + B = 1` means when a regime shift occurs (currency crisis, central bank action) the gate can set `A→0, B→1` — full state reset. SSM's `A ∈ (0,1)` decays slowly and carries stale state across regime boundaries.

**Expected crossover pattern**: SSM wins broadly; CfC wins only where sudden discontinuities dominate. If CfC wins on ≥2 datasets, that confirms the inductive bias hypothesis.

**What would invalidate the experiment**:
- CfC never wins anywhere → systematic OOM/underfitting bias (internal_dim=16×256=4096 too large for budget)
- CfC wins everywhere → either CfC is universally better (re-evaluate search pool) or training noise

---

### Exp 4: TiDAR Speedup

**Setup**: Compare draft (`ar_steps=0`) vs partial-AR (`k=1,4,16`) vs full-AR (`ar_steps=H`) wall-clock time and MSE quality on ETTh1 H=720.

**Expected speedup vs quality trade-off (H=720, RTX 4060)**

| Mode | Estimated ms/sample | Speedup vs full-AR | MSE delta vs full-AR |
|------|--------------------|--------------------|----------------------|
| draft (k=0) | ~2–5 ms | 50–150× | +5–15% |
| partial-AR k=1 | ~5–10 ms | 25–75× | +2–8% |
| partial-AR k=4 | ~15–25 ms | 10–25× | +1–3% |
| partial-AR k=16 | ~50–80 ms | 3–8× | ~0% |
| full-AR (k=H) | ~300–500 ms | 1× | baseline |

**Key finding expected**: Quality curve flattens fast — `k=4` captures ~97% of full-AR quality at 10–25× speedup. This is the practical sweet spot.

**Horizon scaling**: Speedup scales with H. At H=96 the advantage is modest (~6×); at H=720 the advantage is large. The draft mode is most valuable for long-horizon forecasting.

**What would invalidate the experiment**:
- Draft MSE >30% worse than full-AR → TiDAR-TS draft head not learning to forecast
- Speedup <5× at H=720 → bottleneck is not in the sequential AR pass (unexpected)

---

### Exp 5: Cross-Dataset Generalization

**Setup**: Best genome from ETTh1 Exp 1. Two transfer modes: zero-shot (weights transferred) and architecture transfer (genome kept, weights retrained).

**Zero-shot evaluation (no retraining, same-variate only)**

| Source → Target | Expected MSE degradation | Reason |
|----------------|------------------------|--------|
| ETTh1 → ETTh2 | ~20–40% | Same 7 variates, similar hourly structure, different regime |
| ETTh1 → ETTm1 | ~30–60% | Same variates but 15-min granularity; weights not aligned to faster scale |
| ETTh1 → ETTm2 | ~30–60% | Same as ETTm1 |

Zero-shot is a sanity check. If degradation is <10%, the operators generalise exceptionally well. If >80%, the architecture is over-specialised to ETTh1.

**Architecture transfer (genome kept, weights trained from scratch on target)**

| Source → Target | Searched vs Rec-1 Baseline | Expected |
|----------------|--------------------------|---------|
| ETTh1 → ETTh2 | Searched wins | Same ETT family; GConv+Attn pattern transfers |
| ETTh1 → ETTm1/ETTm2 | Marginal win or tie | Different granularity limits but structure similar |
| ETTh1 → Electricity | Rec-1 may win | 321 variates; GA searched for 7-variate structure; SSM-heavy genome better |
| ETTh1 → Traffic | Rec-1 may win | 862 variates; structure mismatch even more extreme |
| ETTh1 → Exchange | Tie or searched | Depends on whether CfC was discovered in Exp 1 genome |

**Key insight**: Architecture transfer expected to hold within the same dataset family (ETT → ETT) but degrade for structurally different datasets (Traffic 862-variate, Exchange daily). A successful transfer within ETT family alone is meaningful — it shows the searched operators capture inductive biases beyond dataset-specific fitting.

**What would invalidate the experiment**:
- Rec-1 wins on all targets → architecture is ETTh1-specific and doesn't generalise at all
- Searched wins on all targets including Traffic/Electricity → architecture is universally optimal (publish immediately)


## 14. Future Work

### A. Optimized Block Implementations for Post-GA Full Training

During NSGA-II evolution, candidates run for only 300–500 steps so the current slow implementations are tolerable. But for **post-evolution full training** (50K–100K steps per top-K genome), each block family has a known bottleneck that should be replaced:

**EMA decomposition** (DMamba structure)
- Current bottleneck: Python `for t in range(L)` loop — 96 serial CUDA kernel dispatches per forward pass
- Fix idea: parallel prefix scan — computes the same IIR filter result in O(log L) parallel rounds; no sequential dependency
- Estimated speedup: ~10× for the EMA portion

**SSM / Recurrence (classes 5, 6, 10, 11, 17–21)**
- Current bottleneck: explicit SEMI_SEPARABLE matrix `(B, L, L, internal_dim)` — for Rec-1 this is 2.4 GB per layer per batch, and O(L²×16d) compute
- Fix idea: Mamba-style parallel associative scan — avoids materialising the full L×L matrix; runs in O(L × internal_dim) memory and O(L log L) compute
- Estimated speedup: ~40× for SSM-containing genomes (20s/step → ~0.5s/step)

**Attention (classes 1–4)**
- Current bottleneck: manual `T = C @ Bᵀ / sqrt(d)` followed by softmax — builds full L×L float32 matrix, O(L²) memory
- Fix idea: drop-in replacement with `torch.nn.functional.scaled_dot_product_attention` which dispatches FlashAttention-2 automatically when available — O(L) memory, ~2–3× faster for L≥128

**Convolution (classes 7–8)**
- Already fast — GConv-1 (kernel size 3) is nearly free; GConv-2 (MLP-generated 64-tap kernel) is the slow one due to the MLP per forward pass
- Fix idea: cache the generated kernel across calls in the same batch (kernel depends only on global average, which is the same for all positions)

**Priority order**: SSM scan (highest impact) → EMA scan → FlashAttention → GConv-2 cache

These fixes apply only to the full-training phase. The GA proxy evaluation can keep the current implementation for code simplicity.

---

### B. Test-Time Training on Diffusion Drafting Mistakes

**Motivation**: TiDAR-TS draft mode predicts all H forecast steps in one forward pass. At long horizons (H=720) the draft head accumulates errors — it cannot condition on future ground truth and predictions drift from the AR-refined output.

**Core idea**: At test time, when the lookback window is observed, run a few gradient steps to adapt the **draft head weights** using the AR-refined output as a free pseudo-label:

```
TTT-augmented inference:
  1. Draft:    ŷ_draft  = diff_head(model(x))            — 1 forward pass
  2. AR-ref:   ŷ_AR     = ar_head(model, x, k steps)     — k sequential passes
  3. TTT loss: L = MSE(ŷ_draft,  stop_grad(ŷ_AR))        — draft learns from AR
  4. Update:   θ_draft  ← θ_draft − η ∇L                 — only draft head updated
  5. Re-draft: ŷ_final  = diff_head(model(x))            — corrected prediction
```

`stop_gradient` on `ŷ_AR` is critical — it prevents the AR head from being corrupted by the TTT update. Only the draft head parameters change; backbone and AR head stay frozen.

**Why this works**:
- The AR head is strictly more accurate than the draft head for the current input
- The AR output is available at test time at no label cost — inference already runs the AR pass for the first k steps anyway
- A small number of TTT gradient steps (3–10) adapts the draft head to the current input's distribution without overfitting

**Expected benefit**:
- Draft MSE after TTT ≈ partial-AR (k=4) quality at nearly draft speed
- Largest gains on out-of-distribution inputs: regime shifts, anomalies, holiday effects
- Directly addresses the end-goal noted in this plan: "Adding TTT upon mistakes during diffusion drafting"

**Implementation path**:
- Add `ttt_steps: int` parameter to `TiDARTSModel.forecast()`
- When `ttt_steps > 0`: generate AR pseudo-label → compute TTT loss → update only `diff_head` → re-run draft
- Add `--ttt_steps` flag to `exp4_speedup.py` to benchmark TTT overhead vs quality gain at each horizon