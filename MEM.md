# Memory Mechanisms in Modern Sequence Models

This document provides comprehensive explanations of three key memory mechanisms:
1. **TTT (Test-Time Training)** - Gradient descent as hidden state update
2. **Google Titan Memory** - Surprise-based neural long-term memory
3. **DeepSeek Engram Memory** - Slot-based memory with competitive writing

All examples use **L=5** (sequence length) and **D=4** (hidden dimension).

---

## 1. TTT (Test-Time Training)

**Paper:** "Learning to (Learn at Test Time): RNNs with Expressive Hidden States" (Yu et al., 2024)

### Core Idea
TTT replaces the traditional hidden state update rule with **gradient descent on a self-supervised loss**. The hidden state IS a neural network's weights, and updating the hidden state means training that network.

### Mathematical Framework

For each token at position t:
1. **Input transformation:** Create key-value pairs from input
2. **Self-supervised loss:** Measure reconstruction error
3. **Gradient update:** Update weights via gradient descent
4. **Output:** Query the updated model

### TTT-Linear: Linear Model as Hidden State

**Hidden State:** Weight matrix W ∈ ℝ^(D×D)

**Update Rule:**
```
W_t = W_{t-1} - η · ∇_W L(W_{t-1}, x_t)
```

Where:
- η = learning rate
- L = self-supervised reconstruction loss
- x_t = input at time t

### Example: TTT-Linear with L=5, D=4

**Setup:**
- Input sequence X ∈ ℝ^(5×4)
- Key projection: W_K ∈ ℝ^(4×4)
- Value projection: W_V ∈ ℝ^(4×4)
- Query projection: W_Q ∈ ℝ^(4×4)
- Initial hidden state: W_0 = 0 ∈ ℝ^(4×4)
- Learning rate: η = 0.1

**Input Sequence:**
```
X = [x_1]   [0.5  0.3  0.1  0.2]
    [x_2] = [0.2  0.8  0.4  0.1]
    [x_3]   [0.7  0.2  0.6  0.3]
    [x_4]   [0.1  0.5  0.3  0.9]
    [x_5]   [0.4  0.6  0.2  0.7]
```

**Step-by-Step Processing (Token 1):**

1. **Compute Key and Value:**
```
k_1 = W_K · x_1 = [0.6, 0.4, 0.2, 0.3]
v_1 = W_V · x_1 = [0.5, 0.3, 0.4, 0.2]
```

2. **Self-Supervised Loss (Reconstruction):**
```
L(W_0, x_1) = ||W_0 · k_1 - v_1||²
            = ||[0,0,0,0] - [0.5, 0.3, 0.4, 0.2]||²
            = 0.54
```

3. **Compute Gradient:**
```
∇_W L = -2(v_1 - W_0 · k_1) · k_1^T
      = -2 · v_1 · k_1^T

      = -2 · [0.5]   · [0.6, 0.4, 0.2, 0.3]
            [0.3]
            [0.4]
            [0.2]

      = [-0.60  -0.40  -0.20  -0.30]
        [-0.36  -0.24  -0.12  -0.18]
        [-0.48  -0.32  -0.16  -0.24]
        [-0.24  -0.16  -0.08  -0.12]
```

4. **Update Hidden State:**
```
W_1 = W_0 - η · ∇_W L
    = 0 - 0.1 · ∇_W L

    = [0.060  0.040  0.020  0.030]
      [0.036  0.024  0.012  0.018]
      [0.048  0.032  0.016  0.024]
      [0.024  0.016  0.008  0.012]
```

5. **Compute Output:**
```
q_1 = W_Q · x_1 = [0.4, 0.5, 0.3, 0.2]
o_1 = W_1 · q_1 = [0.053, 0.032, 0.042, 0.021]
```

**Continuing for Token 2:**

Using updated W_1:
```
k_2 = W_K · x_2 = [0.3, 0.7, 0.5, 0.2]
v_2 = W_V · x_2 = [0.4, 0.6, 0.2, 0.5]

# Prediction with current weights
pred_2 = W_1 · k_2 = [0.049, 0.029, 0.039, 0.020]

# Loss
L(W_1, x_2) = ||pred_2 - v_2||² = 0.72

# Gradient and update → W_2
# Continue pattern...
```

### TTT-MLP: MLP as Hidden State

Instead of a linear layer, use a 2-layer MLP:
```
f(x; θ) = W_2 · ReLU(W_1 · x + b_1) + b_2
```

Hidden state θ = {W_1, b_1, W_2, b_2}

**Advantages:**
- More expressive hidden state
- Can learn non-linear patterns
- Better long-range dependencies

**Update:**
```
θ_t = θ_{t-1} - η · ∇_θ L(θ_{t-1}, x_t)
```

### Mini-Batch TTT

Process tokens in mini-batches for efficiency:
```
For batch [x_t, x_{t+1}, ..., x_{t+B-1}]:
    L_batch = Σ L(W, x_i)
    W_new = W - η · ∇_W L_batch
```

---

## 2. Google Titan Memory

**Paper:** "Titans: Learning to Memorize at Test Time" (Behrouz et al., 2025)

### Core Idea
Titan introduces a **Neural Long-Term Memory** module that learns to memorize important information based on **surprise**. High surprise = worth remembering.

### Architecture Components

1. **Core (Short-Term):** Sliding window attention
2. **Long-Term Memory:** Persistent memory module
3. **Persistent Memory:** Learnable parameters (task-level)

### Memory Update Mechanism

**Key Insight:** Use the gradient of surprise as memory update signal.

**Surprise Definition:**
```
S_t = L(M_{t-1}, x_t) = ||M_{t-1} · k_t - v_t||²
```

**Memory Update:**
```
M_t = M_{t-1} - η · ∇_M S_t + decay_term
```

### Detailed Example: L=5, D=4

**Setup:**
- Memory matrix: M ∈ ℝ^(4×4)
- Learning rate: η = 0.1
- Decay factor: α = 0.9
- Initial memory: M_0 = I (identity matrix)

**Input Sequence:**
```
X = [x_1]   [0.5  0.3  0.1  0.2]
    [x_2] = [0.2  0.8  0.4  0.1]
    [x_3]   [0.7  0.2  0.6  0.3]
    [x_4]   [0.1  0.5  0.3  0.9]
    [x_5]   [0.4  0.6  0.2  0.7]
```

**Projections (for this example, assume identity for simplicity):**
```
k_t = x_t (key)
v_t = x_t (value)
q_t = x_t (query)
```

**Processing Token 1:**

1. **Retrieve from Memory:**
```
retrieved_1 = M_0 · q_1 = I · [0.5, 0.3, 0.1, 0.2]
            = [0.5, 0.3, 0.1, 0.2]
```

2. **Compute Surprise:**
```
prediction_1 = M_0 · k_1 = [0.5, 0.3, 0.1, 0.2]
surprise_1 = ||prediction_1 - v_1||² = 0  (perfect prediction)
```

3. **Compute Gradient:**
```
∇_M S_1 = -2(v_1 - M_0 · k_1) · k_1^T = 0
```

4. **Update Memory (with decay):**
```
M_1 = α · M_0 - η · ∇_M S_1
    = 0.9 · I - 0 = 0.9 · I
```

**Processing Token 2 (more interesting):**

1. **Retrieve:**
```
retrieved_2 = M_1 · q_2 = 0.9 · [0.2, 0.8, 0.4, 0.1]
            = [0.18, 0.72, 0.36, 0.09]
```

2. **Compute Surprise:**
```
prediction_2 = M_1 · k_2 = [0.18, 0.72, 0.36, 0.09]
surprise_2 = ||prediction_2 - v_2||²
           = ||(0.18-0.2, 0.72-0.8, 0.36-0.4, 0.09-0.1)||²
           = ||(-0.02, -0.08, -0.04, -0.01)||²
           = 0.0085
```

3. **Gradient:**
```
∇_M S_2 = -2(v_2 - M_1 · k_2) · k_2^T
        = -2 · [-0.02, -0.08, -0.04, -0.01]^T · [0.2, 0.8, 0.4, 0.1]

        = [0.008   0.032   0.016   0.004]
          [0.032   0.128   0.064   0.016]
          [0.016   0.064   0.032   0.008]
          [0.004   0.016   0.008   0.002]
```

4. **Update:**
```
M_2 = α · M_1 - η · ∇_M S_2
    = 0.9 · (0.9 · I) - 0.1 · ∇_M S_2
    = 0.81 · I - 0.1 · ∇_M S_2
```

### Memory Gating

Titan uses **forget gates** to control information flow:

```
f_t = σ(W_f · [h_{t-1}, x_t])  # Forget gate
i_t = σ(W_i · [h_{t-1}, x_t])  # Input gate

M_t = f_t ⊙ M_{t-1} + i_t ⊙ ΔM_t
```

Where ΔM_t is the surprise-based update.

### Three Architectures

**MAC (Memory as Context):**
```
output = Attention(Q, K, V) + Memory_Retrieval(q)
```

**MAG (Memory as Gate):**
```
g = σ(Memory_Retrieval(q))
output = g ⊙ Attention(Q, K, V)
```

**MAL (Memory as Layer):**
- Stack memory layer and attention layer alternately
- Each layer has independent parameters

---

## 3. DeepSeek Engram Memory

**Paper:** "Native Sparse Attention with Intra-Block Linear Attention" (DeepSeek, 2025)

### Core Idea
Engram Memory uses a **fixed set of memory slots** that compete for storage through a gating mechanism. Unlike TTT/Titan which use continuous weight updates, Engram uses **discrete slot allocation**.

### Architecture

**Components:**
- Memory slots: M ∈ ℝ^(N_slots × D) where N_slots is fixed (e.g., 64)
- Write gates: Determine which slots to update
- Read mechanism: Soft attention over slots

### Mathematical Framework

**Memory Slots:**
```
M = [m_1, m_2, ..., m_N]^T ∈ ℝ^(N×D)
```

**Write Operation:**
```
# Compute write weights (competitive)
w_write = softmax(M · k_t / √D)  ∈ ℝ^N

# Compute erase vector
e_t = σ(W_e · x_t)  ∈ ℝ^D

# Compute add vector
a_t = tanh(W_a · x_t)  ∈ ℝ^D

# Update memory
M_t[i] = M_{t-1}[i] ⊙ (1 - w_write[i] · e_t) + w_write[i] · a_t
```

**Read Operation:**
```
# Compute read weights
w_read = softmax(M · q_t / √D)  ∈ ℝ^N

# Read from memory
r_t = M^T · w_read  ∈ ℝ^D
```

### Detailed Example: L=5, D=4, N_slots=3

**Setup:**
- Memory slots: M ∈ ℝ^(3×4)
- Initial memory (random small values):
```
M_0 = [0.1  0.2  0.1  0.3]  # Slot 1
      [0.2  0.1  0.3  0.1]  # Slot 2
      [0.3  0.1  0.2  0.2]  # Slot 3
```

**Input Sequence:**
```
X = [x_1]   [0.5  0.3  0.1  0.2]
    [x_2] = [0.2  0.8  0.4  0.1]
    [x_3]   [0.7  0.2  0.6  0.3]
    [x_4]   [0.1  0.5  0.3  0.9]
    [x_5]   [0.4  0.6  0.2  0.7]
```

**Processing Token 1:**

1. **Compute Key:**
```
k_1 = W_K · x_1 = [0.5, 0.3, 0.1, 0.2]  (assume identity projection)
```

2. **Compute Write Weights (which slot to write to):**
```
scores = M_0 · k_1 / √4
       = [0.1·0.5 + 0.2·0.3 + 0.1·0.1 + 0.3·0.2] / 2
         [0.2·0.5 + 0.1·0.3 + 0.3·0.1 + 0.1·0.2] / 2
         [0.3·0.5 + 0.1·0.3 + 0.2·0.1 + 0.2·0.2] / 2
       = [0.18/2, 0.18/2, 0.24/2]
       = [0.09, 0.09, 0.12]

w_write = softmax([0.09, 0.09, 0.12])
        = [0.32, 0.32, 0.36]  # Slot 3 gets slightly more
```

3. **Compute Erase and Add Vectors:**
```
e_1 = σ(W_e · x_1) = [0.6, 0.5, 0.4, 0.5]  # What to forget
a_1 = tanh(W_a · x_1) = [0.4, 0.3, 0.1, 0.2]  # What to add
```

4. **Update Each Memory Slot:**
```
# Slot 1 update (w_write[1] = 0.32):
M_1[1] = M_0[1] ⊙ (1 - 0.32 · e_1) + 0.32 · a_1
       = [0.1, 0.2, 0.1, 0.3] ⊙ [0.81, 0.84, 0.87, 0.84] + [0.13, 0.10, 0.03, 0.06]
       = [0.081, 0.168, 0.087, 0.252] + [0.13, 0.10, 0.03, 0.06]
       = [0.21, 0.27, 0.12, 0.31]

# Slot 2 update (w_write[2] = 0.32):
M_1[2] = [0.21, 0.18, 0.29, 0.15]  (similar calculation)

# Slot 3 update (w_write[3] = 0.36):
M_1[3] = [0.26, 0.17, 0.21, 0.24]  (similar calculation)
```

5. **Read from Memory (for output):**
```
q_1 = W_Q · x_1 = [0.5, 0.3, 0.1, 0.2]

read_scores = M_1 · q_1 / √4
w_read = softmax(read_scores) = [0.34, 0.33, 0.33]

r_1 = Σ w_read[i] · M_1[i]
    = 0.34 · [0.21, 0.27, 0.12, 0.31]
    + 0.33 · [0.21, 0.18, 0.29, 0.15]
    + 0.33 · [0.26, 0.17, 0.21, 0.24]
    = [0.23, 0.21, 0.21, 0.23]
```

### Competitive Writing Mechanism

The key insight is **content-based addressing**:
- Slots with content similar to the key get updated more
- This creates **clustering** of related information
- Over time, slots specialize for different types of content

**Sharpening (optional):**
```
w_write = softmax(scores / τ)  # τ < 1 makes distribution sharper
```

With τ = 0.5:
```
w_write = softmax([0.09, 0.09, 0.12] / 0.5)
        = softmax([0.18, 0.18, 0.24])
        = [0.30, 0.30, 0.40]  # More concentrated on Slot 3
```

---

## Comparison Table

| Aspect | TTT | Titan Memory | Engram Memory |
|--------|-----|--------------|---------------|
| **Memory Type** | Neural network weights | Weight matrix | Fixed slots |
| **Update Signal** | Gradient descent | Surprise-based gradient | Content-addressed gating |
| **Capacity** | O(D²) | O(D²) | O(N_slots × D) |
| **Update Complexity** | O(D²) per token | O(D²) per token | O(N_slots × D) per token |
| **Forgetting** | Implicit (overwriting) | Explicit decay | Explicit erase gate |
| **Interpretability** | Low | Medium | High (discrete slots) |
| **Long-range** | Excellent | Excellent | Limited by N_slots |

### When to Use Each

**TTT:**
- Need to adapt to specific patterns in context
- Long sequences with evolving patterns
- Tasks requiring in-context learning

**Titan Memory:**
- Need surprise-based selective memorization
- Want to combine with attention
- Need interpretable memory updates

**Engram Memory:**
- Fixed memory budget is acceptable
- Want discrete, interpretable storage
- Need fast O(N_slots) retrieval

---

## Implementation Patterns

### TTT-Linear in PyTorch
```python
class TTTLinear(nn.Module):
    def __init__(self, d_model, lr=0.1):
        super().__init__()
        self.d = d_model
        self.lr = lr
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_q = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        W = torch.zeros(B, D, D, device=x.device)
        outputs = []

        for t in range(L):
            k = self.W_k(x[:, t])  # [B, D]
            v = self.W_v(x[:, t])  # [B, D]
            q = self.W_q(x[:, t])  # [B, D]

            # Compute prediction and loss gradient
            pred = torch.bmm(W, k.unsqueeze(-1)).squeeze(-1)
            error = v - pred

            # Gradient: d/dW ||Wk - v||^2 = -2(v - Wk)k^T
            grad = -2 * torch.bmm(error.unsqueeze(-1), k.unsqueeze(1))

            # Update
            W = W - self.lr * grad

            # Output
            out = torch.bmm(W, q.unsqueeze(-1)).squeeze(-1)
            outputs.append(out)

        return torch.stack(outputs, dim=1)
```

### Titan Memory in PyTorch
```python
class TitanMemory(nn.Module):
    def __init__(self, d_model, lr=0.1, decay=0.9):
        super().__init__()
        self.d = d_model
        self.lr = lr
        self.decay = decay
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_q = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        M = torch.eye(D, device=x.device).unsqueeze(0).expand(B, -1, -1)
        outputs = []

        for t in range(L):
            k = self.W_k(x[:, t])
            v = self.W_v(x[:, t])
            q = self.W_q(x[:, t])

            # Surprise-based update
            pred = torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)
            surprise = v - pred
            grad = -2 * torch.bmm(surprise.unsqueeze(-1), k.unsqueeze(1))

            # Decay + update
            M = self.decay * M - self.lr * grad

            # Retrieve
            out = torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)
            outputs.append(out)

        return torch.stack(outputs, dim=1)
```

### Engram Memory in PyTorch
```python
class EngramMemory(nn.Module):
    def __init__(self, d_model, n_slots=64):
        super().__init__()
        self.d = d_model
        self.n_slots = n_slots
        self.M = nn.Parameter(torch.randn(n_slots, d_model) * 0.1)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_e = nn.Linear(d_model, d_model)
        self.W_a = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        M = self.M.unsqueeze(0).expand(B, -1, -1).clone()
        outputs = []

        for t in range(L):
            k = self.W_k(x[:, t])  # [B, D]
            q = self.W_q(x[:, t])
            e = torch.sigmoid(self.W_e(x[:, t]))  # Erase
            a = torch.tanh(self.W_a(x[:, t]))     # Add

            # Write weights
            w_write = F.softmax(torch.bmm(M, k.unsqueeze(-1)).squeeze(-1) / (D ** 0.5), dim=-1)

            # Update memory
            erase = torch.bmm(w_write.unsqueeze(-1), e.unsqueeze(1))  # [B, N, D]
            add = torch.bmm(w_write.unsqueeze(-1), a.unsqueeze(1))
            M = M * (1 - erase) + add

            # Read
            w_read = F.softmax(torch.bmm(M, q.unsqueeze(-1)).squeeze(-1) / (D ** 0.5), dim=-1)
            out = torch.bmm(w_read.unsqueeze(1), M).squeeze(1)
            outputs.append(out)

        return torch.stack(outputs, dim=1)
```

---

## Key Insights

### Why These Memory Mechanisms?

1. **Transformers' Limitation:** Attention has O(L²) complexity and no persistent state
2. **RNN's Limitation:** Fixed-size hidden state limits expressiveness
3. **Solution:** Learnable, adaptive memory that balances capacity and efficiency

### Common Themes

1. **Key-Value Storage:** All three use some form of key-value association
2. **Selective Writing:** Not all information is equally important
3. **Gated Reading:** Output is a weighted combination of stored information
4. **Online Learning:** Memory updates happen per-token during inference

### Future Directions

- **Hybrid architectures:** Combining multiple memory mechanisms
- **Sparse memory:** Only updating relevant portions
- **Hierarchical memory:** Multi-scale temporal storage
- **Memory consolidation:** Compressing old memories