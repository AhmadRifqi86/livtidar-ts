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

## 2. Core Idea

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

## 3. Structural Approaches to Compare

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

## 4. CfC as New LIV Class

CfC (Closed-form Continuous-time) is a natural addition to the LIV operator pool. It comes from the same Liquid AI group that created LIV.

### CfC LIV Specification

```
CfC operator:
  Token mixing:  Time-continuous (semi-separable with continuous-time dynamics)
  Channel mixing: Diagonal (each channel evolves independently)

  State update (closed-form, no ODE solver):
    h_t = σ(-τ_t · f(x_t)) ⊙ A_t + (1 - σ(-τ_t · f(x_t))) ⊙ B_t

  Where:
    f(x_t)  = learned feature extractor
    τ_t     = input-dependent time constant
    A_t, B_t = learned projections

  Proposed genome encoding:
    LIV class: 18 (CfC-1, standard)
    LIV class: 19 (CfC-2, with backbone parameterization)
    Differential variants: 20-21
```

### Why CfC for Time Series
- Native handling of **irregular time intervals** (unlike SSM which assumes uniform)
- 100x faster than Neural ODE at inference
- Naturally captures **continuous dynamics** in physical systems (weather, electricity)
- Low parameter count with strong feature representation

---

## 5. TiDAR Integration (Parallel Prediction)

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

## 6. NSGA-2 Multi-Objective Search

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

## 7. Experimental Plan

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

## 8. Implementation Plan

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

## 9. Compute Requirements

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

## 10. Expected Contributions

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