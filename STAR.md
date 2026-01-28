# STAR Framework: Matrix Operations Guide

Complete guide showing exact matrix operations for all components with concrete examples.

**Setup:** All examples use:
- Sequence length: `L = 5` tokens
- Channel dimension: `d = 4` features

---

## Input Representation

```
X ∈ R^(5×4) - Input sequence of 5 tokens, each with 4 channels

X = [x₀]   [1.0  2.0  3.0  4.0]
    [x₁]   [1.5  2.5  3.5  4.5]
    [x₂] = [2.0  3.0  4.0  5.0]
    [x₃]   [2.5  3.5  4.5  5.5]
    [x₄]   [3.0  4.0  5.0  6.0]
```

---

# Token Mixing Structures

Token mixing defines how tokens at different positions interact. We look at `T^αβ ∈ R^(L×L)` - the mixing pattern for a specific channel pair.

## 1. Diagonal Token Mixing

**Structure:** `T_ij = { c_i if i=j, else 0 }`

**Matrix form for channel α=0, β=0:**

```
T^00 ∈ R^(5×5)

T^00 = [0.5  0    0    0    0  ]
       [0    0.8  0    0    0  ]
       [0    0    1.2  0    0  ]
       [0    0    0    0.9  0  ]
       [0    0    0    0    1.1]
```

**Operation on channel 0:**

```
x^0 = [1.0, 1.5, 2.0, 2.5, 3.0]ᵀ

y^0 = T^00 · x^0

    = [0.5  0    0    0    0  ] [1.0]   [0.50]
      [0    0.8  0    0    0  ] [1.5]   [1.20]
      [0    0    1.2  0    0  ] [2.0] = [2.40]
      [0    0    0    0.9  0  ] [2.5]   [2.25]
      [0    0    0    0    1.1] [3.0]   [3.30]
```

**Key property:** Each output token only depends on its corresponding input token. No temporal interaction!

---

## 2. Low-Rank Token Mixing (Attention)

**Structure:** `T_ij = C_i · B_j` (outer product)

**For channel α=0, β=0:**

```
C^00 = [0.9, 0.8, 0.7, 0.6, 0.5]ᵀ  (output projection)
B^00 = [0.3, 0.5, 0.7, 0.6, 0.4]   (input projection)

T^00 = C^00 · (B^00)ᵀ

     = [0.9] [0.3  0.5  0.7  0.6  0.4]
       [0.8]
       [0.7]
       [0.6]
       [0.5]

     = [0.27  0.45  0.63  0.54  0.36]
       [0.24  0.40  0.56  0.48  0.32]
       [0.21  0.35  0.49  0.42  0.28]
       [0.18  0.30  0.42  0.36  0.24]
       [0.15  0.25  0.35  0.30  0.20]
```

**Operation:**

```
y^0 = T^00 · x^0

    = [0.27  0.45  0.63  0.54  0.36] [1.0]   [4.86]
      [0.24  0.40  0.56  0.48  0.32] [1.5]   [4.32]
      [0.21  0.35  0.49  0.42  0.28] [2.0] = [3.78]
      [0.18  0.30  0.42  0.36  0.24] [2.5]   [3.24]
      [0.15  0.25  0.35  0.30  0.20] [3.0]   [2.70]
```

**Key property:** Dense matrix - every output token sees ALL input tokens!

---

## 3. Semi-Separable Token Mixing (Recurrence)

**Structure:** `T_ij = C_i · ∏(k=j+1 to i-1) A_k · B_j` for i > j

**Parameters:**

```
C^00 = [0.5, 0.6, 0.7, 0.8, 0.9]ᵀ
A^00 = [0.9, 0.8, 0.7, 0.6]  (transition coefficients)
B^00 = [1.0, 1.1, 1.2, 1.3, 1.4]
```

**Matrix construction:**

```
T^00_00 = C_0·B_0 = 0.5·1.0 = 0.50

T^00_10 = C_1·A_1·B_0 = 0.6·0.9·1.0 = 0.54
T^00_11 = C_1·B_1 = 0.6·1.1 = 0.66

T^00_20 = C_2·A_2·A_1·B_0 = 0.7·0.8·0.9·1.0 = 0.504
T^00_21 = C_2·A_2·B_1 = 0.7·0.8·1.1 = 0.616
T^00_22 = C_2·B_2 = 0.7·1.2 = 0.84

... and so on

T^00 = [0.50   0       0       0       0    ]
       [0.54   0.66    0       0       0    ]
       [0.504  0.616   0.84    0       0    ]
       [0.302  0.369   0.504   1.04    0    ]
       [0.181  0.221   0.302   0.624   1.26 ]
```

**Operation:**

```
y^0 = T^00 · x^0

    = [0.50   0       0       0       0    ] [1.0]   [0.50 ]
      [0.54   0.66    0       0       0    ] [1.5]   [1.53 ]
      [0.504  0.616   0.84    0       0    ] [2.0] = [3.108]
      [0.302  0.369   0.504   1.04    0    ] [2.5]   [5.451]
      [0.181  0.221   0.302   0.624   1.26 ] [3.0]   [8.622]
```

**Key property:** Lower triangular (causal) - each token only sees past!

---

## 4. Scaled Toeplitz Token Mixing (Convolution)

**Structure:** `T_ij = C_i · K_(i-j) · B_j` where K depends on relative position

**Parameters (kernel size 3):**

```
C^00 = [1.0, 1.0, 1.0, 1.0, 1.0]ᵀ
K^00 = [0.3, 0.5, 0.2]  (K_0 for same pos, K_1 for 1-back, K_2 for 2-back)
B^00 = [1.0, 1.0, 1.0, 1.0, 1.0]
```

**Matrix construction:**

```
T^00_ij = C_i · K_(i-j) · B_j if 0 ≤ i-j < 3, else 0

T^00 = [0.3  0    0    0    0  ]  (i-j=0: K_0)
       [0.5  0.3  0    0    0  ]  (i-j=0,1: K_0,K_1)
       [0.2  0.5  0.3  0    0  ]  (i-j=0,1,2: K_0,K_1,K_2)
       [0    0.2  0.5  0.3  0  ]
       [0    0    0.2  0.5  0.3]
```

**Operation:**

```
y^0 = T^00 · x^0

    = [0.3  0    0    0    0  ] [1.0]   [0.30]
      [0.5  0.3  0    0    0  ] [1.5]   [0.95]
      [0.2  0.5  0.3  0    0  ] [2.0] = [1.75]
      [0    0.2  0.5  0.3  0  ] [2.5]   [2.65]
      [0    0    0.2  0.5  0.3] [3.0]   [3.64]
```

**Key property:** Toeplitz structure (constant diagonals) - limited receptive field!

---

# Channel Mixing Structures

Channel mixing defines how features at different dimensions interact. We look at `T_ij ∈ R^(d×d)` - the mixing pattern for a specific token pair.

## 1. Diagonal Channel Mixing

**Structure:** `T_ij` is diagonal

**For token pair (i=2, j=1):**

```
T_21 ∈ R^(4×4)

T_21 = [0.5  0    0    0  ]
       [0    0.6  0    0  ]
       [0    0    0.7  0  ]
       [0    0    0    0.8]
```

**Contribution of x_1 to y_2:**

```
x_1 = [1.5, 2.5, 3.5, 4.5]ᵀ

Contribution = T_21 · x_1

             = [0.5  0    0    0  ] [1.5]   [0.75]
               [0    0.6  0    0  ] [2.5]   [1.50]
               [0    0    0.7  0  ] [3.5] = [2.45]
               [0    0    0    0.8] [4.5]   [3.60]
```

**Key property:** Each output channel depends only on corresponding input channel. Maximum parallelization!

---

## 2. Dense Channel Mixing

**Structure:** `T_ij` is full-rank

**For token pair (i=2, j=1):**

```
T_21 ∈ R^(4×4)

T_21 = [0.5  0.3  0.2  0.1]
       [0.4  0.6  0.3  0.2]
       [0.3  0.4  0.7  0.3]
       [0.2  0.3  0.4  0.8]
```

**Contribution of x_1 to y_2:**

```
x_1 = [1.5, 2.5, 3.5, 4.5]ᵀ

Contribution = T_21 · x_1

             = [0.5  0.3  0.2  0.1] [1.5]   [2.05]
               [0.4  0.6  0.3  0.2] [2.5]   [3.25]
               [0.3  0.4  0.7  0.3] [3.5] = [4.45]
               [0.2  0.3  0.4  0.8] [4.5]   [5.65]
```

**Key property:** All input channels contribute to each output channel. Full cross-channel interaction!

---

## 3. Grouped Channel Mixing (Multi-head)

**Structure:** `T_ij` is block-diagonal (2 heads, each size 2)

**For token pair (i=2, j=1):**

```
T_21 ∈ R^(4×4)

         Head 1          Head 2
T_21 = [0.5  0.3  |  0    0  ]
       [0.4  0.6  |  0    0  ]
       [----------+----------]
       [0    0    |  0.7  0.2]
       [0    0    |  0.3  0.8]
```

**Contribution of x_1 to y_2:**

```
x_1 = [1.5, 2.5, 3.5, 4.5]ᵀ

Contribution = T_21 · x_1

             = [0.5  0.3  0    0  ] [1.5]   [1.50]
               [0.4  0.6  0    0  ] [2.5]   [2.10]
               [0    0    0.7  0.2] [3.5] = [3.35]
               [0    0    0.3  0.8] [4.5]   [4.65]
```

**Key property:** Channels mixed within heads, isolated across heads. Balance between expressiveness and parallelizability!

---

# Featurizers (9 Types)

Featurizers compute feature groups (B, C, etc.) from input. Each has its own token/channel mixing structure.

## Featurizer Class 1: Dense Channel + Diagonal Token

**Description:** Standard linear projection - each token independently, all channels mixed

**Operation:**

```
Input: x_i = [1.5, 2.5, 3.5, 4.5]ᵀ  (token i, 4 channels)
Output dimension: h = 6

Weight matrix W_B ∈ R^(6×4):

W_B = [0.5  0.3  0.2  0.1]
      [0.2  0.7  0.1  0.3]
      [0.1  0.4  0.6  0.2]
      [0.3  0.2  0.5  0.4]
      [0.4  0.1  0.3  0.6]
      [0.2  0.3  0.4  0.5]

B_i = W_B · x_i

    = [0.5  0.3  0.2  0.1] [1.5]   [1.75]
      [0.2  0.7  0.1  0.3] [2.5]   [2.65]
      [0.1  0.4  0.6  0.2] [3.5] = [3.55]
      [0.3  0.2  0.5  0.4] [4.5]   [3.85]
      [0.4  0.1  0.3  0.6]         [4.75]
      [0.2  0.3  0.4  0.5]         [5.20]
```

**Properties:**
- Token mixing: Diagonal (each token processed independently)
- Channel mixing: Dense (all input channels affect all outputs)
- Expansion: 4 → 6 dimensions

---

## Featurizer Class 2: Dense Channel + Toeplitz Token

**Description:** Depthwise convolution across tokens, then dense projection

**Step 1: Temporal convolution on each channel**

```
Channel 0 across all tokens: x^0 = [1.0, 1.5, 2.0, 2.5, 3.0]
Kernel K = [0.3, 0.5, 0.2]

For token i=2:
x̃^0_2 = 0.3·x^0_2 + 0.5·x^0_1 + 0.2·x^0_0
      = 0.3·2.0 + 0.5·1.5 + 0.2·1.0 = 1.55

Similarly for all channels:
x̃_2 = [1.55, 2.55, 3.55, 4.55]ᵀ
```

**Step 2: Dense channel projection**

```
W_B ∈ R^(6×4)  (same as Class 1)

B_2 = W_B · x̃_2

    = [0.5  0.3  0.2  0.1] [1.55]   [1.82]
      [0.2  0.7  0.1  0.3] [2.55]   [2.76]
      [0.1  0.4  0.6  0.2] [3.55] = [3.70]
      [0.3  0.2  0.5  0.4] [4.55]   [4.01]
      [0.4  0.1  0.3  0.6]         [4.95]
      [0.2  0.3  0.4  0.5]         [5.42]
```

**Properties:**
- Token mixing: Toeplitz (temporal convolution with local receptive field)
- Channel mixing: Dense (after convolution)
- Used for attention variants with depthwise convolutions (e.g., SA-2)

---

## Featurizer Class 3: Repeat Factor 4

**Description:** Like Class 1, but last 2 feature groups repeated 4 times

**Operation:**

```
Input: x_i = [1.5, 2.5, 3.5, 4.5]ᵀ

Generate 3 feature groups:
- B^(1)_i ∈ R^4  (normal)
- B^(2)_i ∈ R^1  (repeated 4 times for keys)
- B^(3)_i ∈ R^1  (repeated 4 times for values)

W_Q ∈ R^(4×4):  (Query - normal)
W_Q · x_i = [2.1, 3.2, 4.3, 5.4]ᵀ

W_K ∈ R^(1×4):  (Key - will be repeated)
W_K · x_i = [3.5]
Repeated: K_i = [3.5, 3.5, 3.5, 3.5]ᵀ

W_V ∈ R^(1×4):  (Value - will be repeated)
W_V · x_i = [4.2]
Repeated: V_i = [4.2, 4.2, 4.2, 4.2]ᵀ
```

**Properties:**
- Implements Multi-Query / Grouped-Query Attention
- Reduces cache size by 4× for keys and values
- Used in SA-3

---

## Featurizer Class 4: Repeat Factor 2

**Description:** Same as Class 3, but repeat factor is 2

**Operation:**

```
Input: x_i = [1.5, 2.5, 3.5, 4.5]ᵀ

W_Q ∈ R^(4×4):
Q_i = [2.1, 3.2, 4.3, 5.4]ᵀ

W_K ∈ R^(2×4):
K_i = [3.5, 4.6]ᵀ → Repeated: [3.5, 3.5, 4.6, 4.6]ᵀ

W_V ∈ R^(2×4):
V_i = [4.2, 5.3]ᵀ → Repeated: [4.2, 4.2, 5.3, 5.3]ᵀ
```

**Properties:**
- 2× reduction in KV cache
- Used in SA-4

---

## Featurizer Class 5: Expansion Factor 16

**Description:** For recurrences - expand to 16× dimension

**Operation:**

```
Input: x_i = [1.5, 2.5, 3.5, 4.5]ᵀ  (d=4)
Expansion: 16× → hidden dimension = 64

W_B ∈ R^(64×4):  (only showing first 8 rows)

W_B = [0.5  0.3  0.2  0.1]
      [0.2  0.7  0.1  0.3]
      [0.1  0.4  0.6  0.2]
      [0.3  0.2  0.5  0.4]
      [0.4  0.1  0.3  0.6]
      [0.2  0.3  0.4  0.5]
      [0.1  0.5  0.2  0.6]
      [0.3  0.4  0.1  0.7]
      ... (56 more rows)

B_i = W_B · x_i ∈ R^64

First 8 elements:
B_i = [1.75, 2.65, 3.55, 3.85, 4.75, 5.20, 3.15, 4.95, ...]ᵀ
```

**Properties:**
- Massive capacity expansion (4 → 64)
- Used in Rec-1 (Mamba-like layers)
- Allows rich state representation

---

## Featurizer Class 6: Expansion Factor 2

**Description:** Same as Class 5, but smaller expansion

**Operation:**

```
Input: x_i = [1.5, 2.5, 3.5, 4.5]ᵀ  (d=4)
Expansion: 2× → hidden dimension = 8

W_B ∈ R^(8×4):

W_B = [0.5  0.3  0.2  0.1]
      [0.2  0.7  0.1  0.3]
      [0.1  0.4  0.6  0.2]
      [0.3  0.2  0.5  0.4]
      [0.4  0.1  0.3  0.6]
      [0.2  0.3  0.4  0.5]
      [0.1  0.5  0.2  0.6]
      [0.3  0.4  0.1  0.7]

B_i = W_B · x_i = [1.75, 2.65, 3.55, 3.85, 4.75, 5.20, 3.15, 4.95]ᵀ
```

**Properties:**
- Moderate expansion (4 → 8)
- Used in Rec-2
- Balance between capacity and efficiency

---

## Featurizer Class 7: Diagonal Channel + Toeplitz Token (Explicit Kernel)

**Description:** Depthwise separable convolution with explicit short kernel

**Operation:**

```
Input sequence across all tokens:
X = [[1.0, 2.0, 3.0, 4.0],  (token 0)
     [1.5, 2.5, 3.5, 4.5],  (token 1)
     [2.0, 3.0, 4.0, 5.0],  (token 2)
     [2.5, 3.5, 4.5, 5.5],  (token 3)
     [3.0, 4.0, 5.0, 6.0]]  (token 4)

Kernel K ∈ R^3 per channel (explicit parameters):

K^0 = [0.3, 0.5, 0.2]
K^1 = [0.4, 0.6, 0.1]
K^2 = [0.2, 0.7, 0.3]
K^3 = [0.5, 0.4, 0.2]

For token i=2, channel α=0:
B^0_2 = K^0_0·x^0_2 + K^0_1·x^0_1 + K^0_2·x^0_0
      = 0.3·2.0 + 0.5·1.5 + 0.2·1.0 = 1.55

For all channels at token 2:
B_2 = [1.55, 2.60, 3.05, 4.20]ᵀ
```

**Properties:**
- Diagonal channel (each channel independent)
- Toeplitz token (convolution)
- Explicit kernel parameters
- Used in GConv-1 (short convolutions)
- Output dimension = input dimension (4 → 4)

---

## Featurizer Class 8: Diagonal Channel + Toeplitz Token (Implicit Kernel)

**Description:** Like Class 7, but kernel computed implicitly from input

**Operation:**

```
Step 1: Compute kernel from global context

Global pooling of X:
x_global = mean(X, axis=0) = [2.0, 3.0, 4.0, 5.0]ᵀ

Generate kernel via MLP:
W_kernel ∈ R^(kernel_size × d) = R^(16 × 4)  (kernel_size=16, long conv)

K = W_kernel · x_global ∈ R^16

K = [0.3, 0.5, 0.2, 0.1, 0.4, 0.3, 0.2, 0.5, 
     0.1, 0.3, 0.4, 0.2, 0.5, 0.1, 0.3, 0.2]

Step 2: Apply per-channel convolution (same as Class 7)

For each channel independently with the same long kernel:
B^α_i = Σ_j K_j · x^α_{i-j}
```

**Properties:**
- Input-dependent kernel (implicit parameterization)
- Enables long convolutions (kernel size > 3)
- Used in GConv-2 (long gated convolutions)

---

## Featurizer Class 9: 2 Feature Groups (SwiGLU)

**Description:** Two parallel dense projections for gating

**Operation:**

```
Input: x_i = [1.5, 2.5, 3.5, 4.5]ᵀ  (d=4)
Expansion factor: 4 → hidden = 16

W_gate ∈ R^(16×4):
W_value ∈ R^(16×4):

Gate projection (first 4 elements shown):
G_i = W_gate · x_i = [2.1, 3.5, 4.2, 5.1, ...]ᵀ ∈ R^16

Value projection (first 4 elements shown):
V_i = W_value · x_i = [1.8, 2.9, 3.7, 4.5, ...]ᵀ ∈ R^16

Output (element-wise):
B_i = swish(G_i) ⊙ V_i

where swish(x) = x · sigmoid(x)

swish(G_i) = [1.89, 3.35, 4.12, 5.05, ...]ᵀ

B_i = [1.89·1.8, 3.35·2.9, 4.12·3.7, 5.05·4.5, ...]ᵀ
    = [3.40, 9.72, 15.24, 22.73, ...]ᵀ ∈ R^16
```

**Properties:**
- Two feature groups (gate and value)
- Dense channel mixing
- Diagonal token mixing
- Used in GMemless (SwiGLU/FFN)

---

# Complete LIV Examples

## SA-1: Standard Softmax Attention

**Genome:** `12123`
- Featurizer: Class 1 (dense channel, diagonal token, 3 groups)
- Token mixing: Low-rank (2)
- Sparsity: None (1)
- Nonlinearity: Softmax (2)
- Channel mixing: Grouped (3)

**Full operation with L=5, d=4, num_heads=2:**

```
Input: X ∈ R^(5×4)

Step 1: Featurization (Class 1 - linear projections)

W_Q, W_K, W_V ∈ R^(4×4)

Q = X · W_Q^T ∈ R^(5×4)
K = X · W_K^T ∈ R^(5×4)
V = X · W_V^T ∈ R^(5×4)

Step 2: Split into heads (grouped channel mixing)

Q = [Q_head0, Q_head1] ∈ R^(5×2×2)  (5 tokens, 2 heads, 2 dims each)
K = [K_head0, K_head1] ∈ R^(5×2×2)
V = [V_head0, V_head1] ∈ R^(5×2×2)

Step 3: Token mixing (low-rank + softmax)

For head 0:
Scores^(0) = Q_head0 · K_head0^T ∈ R^(5×5)

Example:
Scores^(0) = [[2.0, 1.5, 1.2, 0.8, 0.5],
              [1.8, 2.2, 1.7, 1.1, 0.7],
              [1.5, 1.9, 2.4, 1.6, 1.0],
              [1.2, 1.5, 2.0, 2.3, 1.4],
              [0.9, 1.2, 1.6, 2.1, 2.6]]

Attention^(0) = softmax(Scores^(0) / √2, dim=-1)

Attention^(0) = [[0.31, 0.24, 0.19, 0.14, 0.12],
                 [0.23, 0.33, 0.25, 0.12, 0.07],
                 [0.16, 0.21, 0.32, 0.21, 0.10],
                 [0.11, 0.15, 0.21, 0.33, 0.20],
                 [0.08, 0.11, 0.16, 0.24, 0.41]]

Output_head0 = Attention^(0) · V_head0 ∈ R^(5×2)

Step 4: Concatenate heads and project

Output = concat([Output_head0, Output_head1]) ∈ R^(5×4)
Y = Output · W_O^T ∈ R^(5×4)
```

---

## Rec-1: Selective State Space (Mamba-like)

**Genome:** `54111`
- Featurizer: Class 5 (dense channel + Toeplitz token, expansion factor 16)
- Token mixing: Semi-separable (4)
- Sparsity: None (1)
- Nonlinearity: None (1)
- Channel mixing: Diagonal (1)

**Full operation with L=5, d=4, state_dim=64 (16× expansion):**

```
Input: X ∈ R^(5×4)

Step 1: Featurization (Class 5 - expansion 16×)

Project to expanded dimension (4 → 64):
W_B ∈ R^(64×4), W_C ∈ R^(64×4)

For each token i:
B_i = W_B · x_i ∈ R^64  (input projection)
C_i = W_C · x_i ∈ R^64  (output projection)

Additionally compute:
A_i ∈ R^64  (state transition, typically log-parameterized)
Δ_i ∈ R^64  (discretization step, input-dependent)

Example for token i=2:
B_2 = [0.5, 0.3, 0.2, ..., 0.4]ᵀ ∈ R^64
C_2 = [0.8, 0.6, 0.4, ..., 0.7]ᵀ ∈ R^64
A_2 = [0.95, 0.92, 0.98, ..., 0.94]ᵀ ∈ R^64
Δ_2 = [0.1, 0.15, 0.08, ..., 0.12]ᵀ ∈ R^64

Step 2: Discretization (continuous → discrete)

Ā_i = exp(Δ_i ⊙ A_i)  (element-wise)
B̄_i = Δ_i ⊙ B_i

Example:
Ā_2 = [0.905, 0.871, 0.924, ..., 0.893]ᵀ
B̄_2 = [0.05, 0.045, 0.016, ..., 0.048]ᵀ

Step 3: Token mixing (semi-separable recurrence)

State evolution (diagonal channel mixing - each channel independent):
h_0 = B̄_0 ⊙ x_0^expanded
h_1 = Ā_1 ⊙ h_0 + B̄_1 ⊙ x_1^expanded
h_2 = Ā_2 ⊙ h_1 + B̄_2 ⊙ x_2^expanded
...

Semi-separable matrix T^αα (for channel α):

T^αα = [C_0·B_0           0                    0          ...]
       [C_1·A_1·B_0       C_1·B_1              0          ...]
       [C_2·A_2·A_1·B_0   C_2·A_2·B_1          C_2·B_2    ...]
       [C_3·∏A·B_0        C_3·A_3·A_2·B_1      ...        ...]

Output at position i:
y_i = C_i ⊙ h_i  (element-wise, then project back)

Step 4: Project back to original dimension

W_out ∈ R^(4×64)
Output_i = W_out · y_i ∈ R^4
```

**Key properties:**
- Semi-separable structure enables O(L) linear recurrence
- Diagonal channel mixing: each of 64 channels evolves independently
- Input-dependent A, B, C provides selectivity (hence "Selective SSM")
- Large state (64) captures long-range dependencies
- No nonlinearity in token mixing (linearity enables parallel scan)

---

## Rec-2: Compact Selective State Space

**Genome:** `64111`
- Featurizer: Class 6 (dense channel + Toeplitz token, expansion factor 2)
- Token mixing: Semi-separable (4)
- Sparsity: None (1)
- Nonlinearity: None (1)
- Channel mixing: Diagonal (1)

**Full operation with L=5, d=4, state_dim=8 (2× expansion):**

```
Input: X ∈ R^(5×4)

Step 1: Featurization (Class 6 - expansion 2×)

Project to moderately expanded dimension (4 → 8):
W_B ∈ R^(8×4), W_C ∈ R^(8×4)

For token i=2:
B_2 = W_B · x_2 = [0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.7, 0.5]ᵀ
C_2 = W_C · x_2 = [0.8, 0.6, 0.4, 0.9, 0.3, 0.7, 0.5, 0.8]ᵀ
A_2 = [0.95, 0.92, 0.98, 0.90, 0.94, 0.96, 0.93, 0.97]ᵀ

Step 2: State evolution (same as Rec-1, smaller state)

h_i ∈ R^8  (vs R^64 in Rec-1)

h_0 = B̄_0 ⊙ x_0^expanded
h_1 = Ā_1 ⊙ h_0 + B̄_1 ⊙ x_1^expanded
h_2 = Ā_2 ⊙ h_1 + B̄_2 ⊙ x_2^expanded

h_2 = [0.42, 0.38, 0.65, 0.28, 0.51, 0.44, 0.58, 0.47]ᵀ

Step 3: Output computation

y_2 = C_2 ⊙ h_2 = [0.34, 0.23, 0.26, 0.25, 0.15, 0.31, 0.29, 0.38]ᵀ

Step 4: Project back

W_out ∈ R^(4×8)
Output_2 = W_out · y_2 ∈ R^4
```

**Key properties:**
- Same structure as Rec-1, smaller state dimension
- Trade-off: less capacity but more parameter efficient
- 8× fewer parameters in state projections vs Rec-1
- Suitable for efficiency-focused architectures

---

## GConv-1: Gated Short Convolution

**Genome:** `73111`
- Featurizer: Class 7 (diagonal channel + Toeplitz token, explicit kernel)
- Token mixing: Scaled Toeplitz (3)
- Sparsity: None (1)
- Nonlinearity: None (1)
- Channel mixing: Diagonal (1)

**Full operation with L=5, d=4, kernel_size=3:**

```
Input: X ∈ R^(5×4)

Step 1: Featurization (Class 7 - explicit short convolution)

Per-channel explicit kernels K^α ∈ R^3:

K^0 = [0.3, 0.5, 0.2]  (weights for pos 0, -1, -2)
K^1 = [0.4, 0.4, 0.2]
K^2 = [0.2, 0.6, 0.2]
K^3 = [0.5, 0.3, 0.2]

Gating projections (input-dependent):
C_i = σ(W_C · x_i) ∈ R^4  (output gate)
B_i = σ(W_B · x_i) ∈ R^4  (input gate)

Step 2: Token mixing (Scaled Toeplitz)

For channel α, the Toeplitz matrix T^αα:

T^αα = [K^α_0    0        0        0        0    ]
       [K^α_1    K^α_0    0        0        0    ]
       [K^α_2    K^α_1    K^α_0    0        0    ]
       [0        K^α_2    K^α_1    K^α_0    0    ]
       [0        0        K^α_2    K^α_1    K^α_0]

For channel 0:

T^00 = [0.3  0    0    0    0  ]
       [0.5  0.3  0    0    0  ]
       [0.2  0.5  0.3  0    0  ]
       [0    0.2  0.5  0.3  0  ]
       [0    0    0.2  0.5  0.3]

Step 3: Gated convolution

Intermediate: z^α = T^αα · x^α

For channel 0:
z^0 = T^00 · [1.0, 1.5, 2.0, 2.5, 3.0]ᵀ
    = [0.30, 0.95, 1.55, 2.15, 2.75]ᵀ

Full scaled Toeplitz (with C, B gates):
y^α_i = C^α_i · z^α_i · B^α_j  (summed over receptive field)

Step 4: Output (diagonal channel mixing)

Each channel processed independently:
Output = [y^0, y^1, y^2, y^3]ᵀ
```

**Key properties:**
- Explicit kernel parameters (learned directly)
- Short receptive field (kernel_size=3)
- Toeplitz structure enables FFT-based computation
- Diagonal channel mixing: no cross-channel interaction in token mixing
- Gating provides input-dependent modulation

---

## GConv-2: Gated Long Convolution

**Genome:** `83111`
- Featurizer: Class 8 (diagonal channel + Toeplitz token, implicit kernel)
- Token mixing: Scaled Toeplitz (3)
- Sparsity: None (1)
- Nonlinearity: None (1)
- Channel mixing: Diagonal (1)

**Full operation with L=5, d=4, kernel_size=16:**

```
Input: X ∈ R^(5×4)

Step 1: Featurization (Class 8 - implicit long kernel)

Unlike GConv-1, kernel is computed from global context:

Global pooling:
x_global = mean(X, axis=0) = [2.0, 3.0, 4.0, 5.0]ᵀ

Generate long kernel via implicit parameterization:
W_kernel ∈ R^(16×4)  (kernel generator network)

K = W_kernel · x_global ∈ R^16

K = [0.30, 0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06,
     0.05, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01]

(Kernel typically decays - captures long-range with diminishing weights)

Step 2: Token mixing (Long Toeplitz via FFT)

For L=5 with kernel_size=16 (causal, padded):

T^αα = [K_0   0     0     0     0   ]
       [K_1   K_0   0     0     0   ]
       [K_2   K_1   K_0   0     0   ]
       [K_3   K_2   K_1   K_0   0   ]
       [K_4   K_3   K_2   K_1   K_0 ]

Efficient computation via FFT:
1. Zero-pad x^α to length 2L
2. FFT(x^α_padded), FFT(K_padded)
3. Multiply in frequency domain
4. IFFT and take first L elements

z^α = IFFT(FFT(x^α) ⊙ FFT(K))[:L]

Step 3: Gating (same as GConv-1)

C_i = σ(W_C · x_i)  (output gate)
y_i = C_i ⊙ z_i

Step 4: Output

Output ∈ R^(5×4)
```

**Key properties:**
- Implicit kernel parameterization (input-dependent or learned generator)
- Long receptive field without explicit L² parameters
- FFT enables O(L log L) computation
- Represents modern long convolution architectures (Hyena-style)

---

## GMemless: Gated Memoryless Unit (SwiGLU)

**Genome:** `91142`
- Featurizer: Class 9 (dense channel + diagonal token, 2 feature groups)
- Token mixing: Diagonal (1)
- Sparsity: None (1)
- Nonlinearity: Swish (4)
- Channel mixing: Dense (2)

**Full operation with L=5, d=4, hidden=16 (4× expansion):**

```
Input: X ∈ R^(5×4)

Step 1: Featurization (Class 9 - two parallel projections)

Gate projection:
W_gate ∈ R^(16×4)

Value projection:
W_value ∈ R^(16×4)

For token i=2, x_2 = [2.0, 3.0, 4.0, 5.0]ᵀ:

G_2 = W_gate · x_2 = [2.1, 3.5, 4.2, 5.1, 2.8, 3.2, 4.5, 5.3,
                      2.4, 3.8, 4.0, 4.9, 2.6, 3.4, 4.3, 5.0]ᵀ

V_2 = W_value · x_2 = [1.8, 2.9, 3.7, 4.5, 2.2, 3.1, 3.9, 4.8,
                       1.9, 3.0, 3.8, 4.6, 2.0, 3.2, 4.0, 4.7]ᵀ

Step 2: Token mixing (Diagonal - each token independent)

T_ij = { swish(G_i) if i=j, else 0 }

The "token mixing" is purely position-wise (diagonal):
- Token 0 only sees token 0
- Token 1 only sees token 1
- etc.

No temporal interaction!

Step 3: Nonlinearity (Swish gating)

swish(x) = x · sigmoid(x)

swish(G_2) = G_2 ⊙ sigmoid(G_2)

sigmoid(G_2) = [0.89, 0.97, 0.99, 0.99, 0.94, 0.96, 0.99, 0.99,
                0.92, 0.98, 0.98, 0.99, 0.93, 0.97, 0.99, 0.99]ᵀ

swish(G_2) = [1.87, 3.40, 4.16, 5.05, 2.63, 3.07, 4.46, 5.25,
              2.21, 3.72, 3.92, 4.85, 2.42, 3.30, 4.26, 4.95]ᵀ

Step 4: Gated output

H_2 = swish(G_2) ⊙ V_2

    = [1.87·1.8, 3.40·2.9, 4.16·3.7, 5.05·4.5, ...]ᵀ
    = [3.37, 9.86, 15.39, 22.73, 5.79, 9.52, 17.39, 25.20,
       4.20, 11.16, 14.90, 22.31, 4.84, 10.56, 17.04, 23.27]ᵀ

Step 5: Channel mixing (Dense - output projection)

W_out ∈ R^(4×16)

Output_2 = W_out · H_2 ∈ R^4

This mixes all 16 hidden dimensions back to 4 output channels.
```

**Key properties:**
- Diagonal token mixing: NO temporal interaction (memoryless)
- Dense channel mixing: full cross-channel interaction
- SwiGLU activation: smooth gating mechanism
- Acts as the "MLP" or "FFN" in transformer-like architectures
- Provides nonlinear channel mixing between temporal layers

---

# Differential Variants (Classes 10-17)

Differential variants apply the same LIV twice in parallel and output their difference:

**Structure:**
```
y = LIV_1(x) - LIV_2(x)
```

where LIV_1 and LIV_2 are identical architectures but with separate parameters.

**Example: SA-1-Diff (Class 10)**

```
Genome: Same as SA-1 (12123) but applied differentially

Input: X ∈ R^(5×4)

Branch 1: Standard SA-1
Y_1 = Attention_1(Q_1, K_1, V_1) · X

Branch 2: Standard SA-1 (separate parameters)
Y_2 = Attention_2(Q_2, K_2, V_2) · X

Output: Y = Y_1 - Y_2
```

**Key properties:**
- Doubles parameters but improves gradient flow
- Difference operation reduces attention noise
- Similar to "Differential Transformer" architecture
- Available for all non-memoryless LIVs (SA, Rec, GConv)

---

# Summary: Complete LIV Option Pool

| Class | Name | Genome | Token Mix | Channel Mix | Key Feature |
|-------|------|--------|-----------|-------------|-------------|
| 1 | SA-1 | 12123 | Low-rank | Grouped | Standard attention |
| 2 | SA-2 | 22123 | Low-rank | Grouped | + depthwise conv |
| 3 | SA-3 | 32123 | Low-rank | Grouped | MQA (repeat 4) |
| 4 | SA-4 | 42123 | Low-rank | Grouped | GQA (repeat 2) |
| 5 | Rec-1 | 54111 | Semi-sep | Diagonal | Mamba-like (×16) |
| 6 | Rec-2 | 64111 | Semi-sep | Diagonal | Compact (×2) |
| 7 | GConv-1 | 73111 | Toeplitz | Diagonal | Short conv |
| 8 | GConv-2 | 83111 | Toeplitz | Diagonal | Long conv |
| 9 | GMemless | 91142 | Diagonal | Dense | SwiGLU/FFN |
| 10-17 | *-Diff | — | — | — | Differential variants |