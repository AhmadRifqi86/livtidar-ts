"""
Unified LIV Framework with Consistent Feature Interface

Core equation: Y = T(x) · x
Token mixing: T_ij = C_i · M_ij · B_j  (unified for ALL types)

All token mixing types use exactly 3 feature groups:
- B: input projection  [batch, seq_len, internal_dim]
- C: output projection [batch, seq_len, internal_dim]
- S: structure param   [batch, seq_len, internal_dim] or [internal_dim, kernel_size]

The M_ij matrix is determined by type:
- Diagonal:      M_ij = δ_ij           (S unused, or extra scaling)
- Low-rank:      M_ij = 1              (S unused, outer product C·B^T)
- Toeplitz:      M_ij = S_{i-j}        (S = convolution kernel)
- Semi-separable: M_ij = ∏A from S    (S = transition coefficients A)

Featurizer classes 1-9 each produce {B, C, S, V} with class-specific logic:
- Classes 1-4: Attention variants (MHA, conv-pre, MQA, GQA)
- Classes 5-6: Recurrence variants (expansion 16×, 2×)
- Classes 7-8: Convolution variants (explicit, implicit kernel)
- Class 9: SwiGLU / FFN (gated memoryless)

Non-linearity: configurable via NonLinearity enum on UnifiedLIV.
Sparsity: implemented as SparsityMask (see commented usage in UnifiedLIV).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import ABC, abstractmethod
from enum import Enum


class TokenMixType(Enum):
    DIAGONAL = 1       # T_ij = C_i · δ_ij · B_j = C_i · B_i · δ_ij
    LOW_RANK = 2       # T_ij = C_i · 1 · B_j = C_i · B_j
    TOEPLITZ = 3       # T_ij = C_i · K_{i-j} · B_j
    SEMI_SEPARABLE = 4 # T_ij = C_i · (∏A) · B_j


class ChannelMixType(Enum):
    DIAGONAL = 1  # W_{αβ} = w_α · δ_{αβ}
    DENSE = 2     # W_{αβ} = W[α,β]
    GROUPED = 3   # W = block_diag(W_1, ..., W_h)


class NonLinearity(Enum):
    """5-tuple axis 5: activation applied after token mixing.

    SOFTMAX is special — applied to T matrix (row-wise) inside TokenMixWeightGenerator.
    All others are applied element-wise to the mixed output (after T @ V, before channel mix).
    """
    NONE = 0
    SOFTMAX = 1    # Row-wise on T (handled in TokenMixWeightGenerator)
    SIGMOID = 2    # σ(mixed)
    SILU = 3       # SiLU(mixed)
    GELU = 4       # GELU(mixed)
    RELU = 5       # ReLU(mixed)


_NONLINEARITY_FN = {
    NonLinearity.NONE: lambda x: x,
    NonLinearity.SIGMOID: torch.sigmoid,
    NonLinearity.SILU: F.silu,
    NonLinearity.GELU: F.gelu,
    NonLinearity.RELU: F.relu,
}


class SparsityType(Enum):
    """5-tuple axis 4: sparsity pattern on token mixing matrix T."""
    NONE = 0       # Dense T
    CAUSAL = 1     # Lower-triangular (already handled by causal flag)
    BANDED = 2     # Entries within bandwidth of diagonal
    TOP_K = 3      # Keep only top-k values per row
    TIDAR_HYBRID = 4  # Causal prefix + bidirectional mask block (TiDAR)


# =============================================================================
# Featurizer Base Class
# =============================================================================

class FeaturizerBase(nn.Module, ABC):
    """Abstract base for all featurizers.

    Every featurizer produces the same interface:
        {B, C, S, V} where:
        - B: [batch, seq_len, internal_dim] — input projection
        - C: [batch, seq_len, internal_dim] — output projection
        - S: structure param (shape varies by token mixing type)
        - V: [batch, seq_len, internal_dim] — values to be mixed

    Subclasses define HOW these are computed from input x.
    """

    def __init__(self, dim: int, internal_dim: int, num_heads: int = 1):
        super().__init__()
        self.dim = dim
        self._internal_dim = internal_dim
        self.num_heads = num_heads

    @property
    def internal_dim(self) -> int:
        return self._internal_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> dict:
        """Compute B, C, S, V from input x: [batch, seq_len, dim]"""
        ...


# =============================================================================
# Unified Featurizer (generic fallback, kept as-is)
# =============================================================================

class UnifiedFeaturizer(FeaturizerBase):
    """Generic featurizer that produces (B, C, S, V) based on token_mix_type.

    This is the original generic implementation. For paper-accurate
    featurizers, use Featurizer1-Featurizer9 instead.
    """

    def __init__(self, dim: int, num_heads: int = 1,
                 token_mix_type: TokenMixType = TokenMixType.LOW_RANK,
                 kernel_size: int = 3):
        super().__init__(dim, dim, num_heads)
        self.token_mix_type = token_mix_type
        self.kernel_size = kernel_size

        self.W_B = nn.Linear(dim, dim)
        self.W_C = nn.Linear(dim, dim)

        if token_mix_type == TokenMixType.DIAGONAL:
            self.W_S = nn.Linear(dim, dim)
        elif token_mix_type == TokenMixType.LOW_RANK:
            self.W_S = nn.Linear(dim, dim)
        elif token_mix_type == TokenMixType.TOEPLITZ:
            self.W_S = nn.Linear(dim, dim * kernel_size)
        elif token_mix_type == TokenMixType.SEMI_SEPARABLE:
            self.W_S = nn.Linear(dim, dim)
            self.A_log = nn.Parameter(torch.zeros(dim))

        self.W_V = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor):
        batch, seq_len, dim = x.shape
        B = self.W_B(x)
        C = self.W_C(x)
        V = self.W_V(x)

        if self.token_mix_type == TokenMixType.DIAGONAL:
            S = torch.sigmoid(self.W_S(x))
        elif self.token_mix_type == TokenMixType.LOW_RANK:
            S = self.W_S(x)
        elif self.token_mix_type == TokenMixType.TOEPLITZ:
            S_raw = self.W_S(x)
            S = S_raw.mean(dim=1).view(batch, self.dim, self.kernel_size)
        elif self.token_mix_type == TokenMixType.SEMI_SEPARABLE:
            S = torch.sigmoid(self.W_S(x) + self.A_log)

        return {'B': B, 'C': C, 'S': S, 'V': V}


# =============================================================================
# Featurizer Classes 1-9
# =============================================================================

class Featurizer1(FeaturizerBase):
    """Class 1: Dense channel + Diagonal token, 3 groups (Q, K, V).

    Standard MHA featurizer. Each token projected independently (diagonal token),
    all channels mixed (dense channel). Produces Q→C, K→B, V→V.
    Used by SA-1.
    """

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__(dim, dim, num_heads)
        self.W_Q = nn.Linear(dim, dim)
        self.W_K = nn.Linear(dim, dim)
        self.W_V = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor):
        B = self.W_K(x)
        C = self.W_Q(x)
        S = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
        V = self.W_V(x)
        return {'B': B, 'C': C, 'S': S, 'V': V}


class Featurizer2(FeaturizerBase):
    """Class 2: Dense channel + Toeplitz token, 3 groups.

    Like Class 1, but applies depthwise causal convolution to input
    BEFORE the Q, K, V projections. Used by SA-2.
    """

    def __init__(self, dim: int, num_heads: int = 8, conv_kernel: int = 3):
        super().__init__(dim, dim, num_heads)
        self.conv_kernel = conv_kernel
        # Depthwise conv (per-channel)
        self.dw_conv = nn.Conv1d(dim, dim, conv_kernel, padding=conv_kernel - 1, groups=dim)
        self.W_Q = nn.Linear(dim, dim)
        self.W_K = nn.Linear(dim, dim)
        self.W_V = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor):
        # Step 1: depthwise causal convolution
        x_t = x.transpose(1, 2)  # [B, dim, L]
        x_conv = self.dw_conv(x_t)[..., :x.shape[1]]  # causal truncation
        x_conv = x_conv.transpose(1, 2)  # [B, L, dim]

        # Step 2: project from conv'd input
        B = self.W_K(x_conv)
        C = self.W_Q(x_conv)
        S = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
        V = self.W_V(x_conv)
        return {'B': B, 'C': C, 'S': S, 'V': V}


class Featurizer3(FeaturizerBase):
    """Class 3: Dense channel + Diagonal token, repeat factor 4 (MQA).

    Q is full dim. K, V are projected to dim/repeat_factor then repeated.
    Implements Multi-Query Attention. Used by SA-3.
    """

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__(dim, dim, num_heads)
        self.repeat_factor = num_heads  # 1 KV head repeated to all heads
        self.kv_dim = dim // num_heads  # single head dim
        self.W_Q = nn.Linear(dim, dim)
        self.W_K = nn.Linear(dim, self.kv_dim)
        self.W_V = nn.Linear(dim, self.kv_dim)

    def forward(self, x: torch.Tensor):
        C = self.W_Q(x)  # [B, L, dim] full Q

        # K, V: project to small dim then repeat
        K_small = self.W_K(x)  # [B, L, kv_dim]
        V_small = self.W_V(x)  # [B, L, kv_dim]
        B = K_small.repeat(1, 1, self.repeat_factor)  # [B, L, dim]
        V = V_small.repeat(1, 1, self.repeat_factor)  # [B, L, dim]

        S = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
        return {'B': B, 'C': C, 'S': S, 'V': V}


class Featurizer4(FeaturizerBase):
    """Class 4: Dense channel + Diagonal token, repeat factor 2 (GQA).

    Q is full dim. K, V have num_heads/2 KV heads, each repeated 2×.
    Implements Grouped-Query Attention. Used by SA-4.
    """

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__(dim, dim, num_heads)
        self.repeat_factor = 2
        self.kv_heads = num_heads // self.repeat_factor
        self.kv_dim = dim // self.repeat_factor  # half-dim
        self.W_Q = nn.Linear(dim, dim)
        self.W_K = nn.Linear(dim, self.kv_dim)
        self.W_V = nn.Linear(dim, self.kv_dim)

    def forward(self, x: torch.Tensor):
        C = self.W_Q(x)  # [B, L, dim]

        K_small = self.W_K(x)  # [B, L, kv_dim]
        V_small = self.W_V(x)  # [B, L, kv_dim]

        # Repeat-interleave: [a, b] -> [a, a, b, b] for head alignment
        batch, seq_len, _ = K_small.shape
        head_dim = self.dim // self.num_heads
        K_heads = K_small.view(batch, seq_len, self.kv_heads, head_dim)
        V_heads = V_small.view(batch, seq_len, self.kv_heads, head_dim)
        B = K_heads.repeat_interleave(self.repeat_factor, dim=2).reshape(batch, seq_len, self.dim)
        V = V_heads.repeat_interleave(self.repeat_factor, dim=2).reshape(batch, seq_len, self.dim)

        S = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
        return {'B': B, 'C': C, 'S': S, 'V': V}


class Featurizer5(FeaturizerBase):
    """Class 5: Dense channel + expansion 16×.

    Projects B, C, S to expanded state dimension (dim * 16).
    Input-dependent A (state transition), B (input gate), C (output gate).
    Used by Rec-1 (Mamba-like SSM).
    """

    def __init__(self, dim: int, num_heads: int = 1, expansion: int = 16):
        internal = dim * expansion
        super().__init__(dim, internal, num_heads)
        self.expansion = expansion
        self.W_B = nn.Linear(dim, internal)
        self.W_C = nn.Linear(dim, internal)
        self.W_A = nn.Linear(dim, internal)
        self.W_V = nn.Linear(dim, internal)
        self.A_log = nn.Parameter(torch.zeros(internal))

    def forward(self, x: torch.Tensor):
        B = self.W_B(x)
        C = self.W_C(x)
        S = torch.sigmoid(self.W_A(x) + self.A_log)  # A in (0,1) for stability
        V = self.W_V(x)
        return {'B': B, 'C': C, 'S': S, 'V': V}


class Featurizer6(FeaturizerBase):
    """Class 6: Dense channel + expansion 2×.

    Same as Class 5 with smaller expansion. Used by Rec-2.
    """

    def __init__(self, dim: int, num_heads: int = 1, expansion: int = 2):
        internal = dim * expansion
        super().__init__(dim, internal, num_heads)
        self.expansion = expansion
        self.W_B = nn.Linear(dim, internal)
        self.W_C = nn.Linear(dim, internal)
        self.W_A = nn.Linear(dim, internal)
        self.W_V = nn.Linear(dim, internal)
        self.A_log = nn.Parameter(torch.zeros(internal))

    def forward(self, x: torch.Tensor):
        B = self.W_B(x)
        C = self.W_C(x)
        S = torch.sigmoid(self.W_A(x) + self.A_log)
        V = self.W_V(x)
        return {'B': B, 'C': C, 'S': S, 'V': V}


class Featurizer7(FeaturizerBase):
    """Class 7: Diagonal channel + Toeplitz token, explicit short kernel.

    B, C are input-dependent gates (sigmoid). S is a learned explicit kernel.
    V is the raw input x (no projection). Used by GConv-1.
    """

    def __init__(self, dim: int, num_heads: int = 1, kernel_size: int = 3):
        super().__init__(dim, dim, num_heads)
        self.kernel_size = kernel_size
        self.W_B = nn.Linear(dim, dim)
        self.W_C = nn.Linear(dim, dim)
        # Explicit per-channel kernel (learned parameters, not input-dependent)
        self.kernel = nn.Parameter(torch.randn(dim, kernel_size) * 0.02)

    def forward(self, x: torch.Tensor):
        batch = x.shape[0]
        B = torch.sigmoid(self.W_B(x))
        C = torch.sigmoid(self.W_C(x))
        # S = explicit kernel, expand for batch dim: [batch, dim, kernel_size]
        S = self.kernel.unsqueeze(0).expand(batch, -1, -1)
        V = x  # Raw input, no projection
        return {'B': B, 'C': C, 'S': S, 'V': V}


class Featurizer8(FeaturizerBase):
    """Class 8: Diagonal channel + Toeplitz token, implicit long kernel.

    Like Class 7, but kernel is computed from global context via MLP.
    Enables long convolutions with input-dependent kernel. Used by GConv-2.
    """

    def __init__(self, dim: int, num_heads: int = 1, kernel_size: int = 64):
        super().__init__(dim, dim, num_heads)
        self.kernel_size = kernel_size
        self.W_B = nn.Linear(dim, dim)
        self.W_C = nn.Linear(dim, dim)
        # Implicit kernel generator: global context -> kernel
        self.kernel_net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim * kernel_size),
        )

    def forward(self, x: torch.Tensor):
        batch, seq_len, dim = x.shape
        B = torch.sigmoid(self.W_B(x))
        C = torch.sigmoid(self.W_C(x))
        # S = implicit kernel from global pooling
        x_global = x.mean(dim=1)  # [batch, dim]
        S = self.kernel_net(x_global).view(batch, dim, self.kernel_size)
        V = x  # Raw input
        return {'B': B, 'C': C, 'S': S, 'V': V}


class Featurizer9(FeaturizerBase):
    """Class 9: Dense channel + Diagonal token, 2 groups (SwiGLU).

    Two parallel projections: gate and value, with expansion.
    B = swish(gate), C = ones (not used in diagonal T), V = value.
    The diagonal token mixing applies B element-wise to V: output = B ⊙ V.
    Channel mixing (dense) projects back. Used by GMemless.
    """

    def __init__(self, dim: int, num_heads: int = 1, expansion: int = 4):
        internal = dim * expansion
        super().__init__(dim, internal, num_heads)
        self.expansion = expansion
        self.W_gate = nn.Linear(dim, internal)
        self.W_value = nn.Linear(dim, internal)

    def forward(self, x: torch.Tensor):
        gate = self.W_gate(x)
        value = self.W_value(x)
        # B = swish(gate), so diagonal T[i,i] = C_i · B_i = B_i when C=1
        B = F.silu(gate)
        C = torch.ones_like(B)
        S = torch.zeros_like(B)
        V = value
        return {'B': B, 'C': C, 'S': S, 'V': V}


class Featurizer10(FeaturizerBase):
    """Class 10: SSM with discretization (S4/Mamba-style).

    Continuous-to-discrete conversion:
      A_bar = exp(Δ · A_log)    — discretized state transition
      B_bar = Δ · W_B(x)       — discretized input gate

    A_log is a learnable parameter (negative for stability).
    Δ (discretization step) is input-dependent via softplus.
    Paired with semi-separable token mixing + diagonal channel mixing.
    """

    def __init__(self, dim: int, num_heads: int = 1, expansion: int = 16):
        internal = dim * expansion
        super().__init__(dim, internal, num_heads)
        self.expansion = expansion
        self.W_B = nn.Linear(dim, internal)
        self.W_C = nn.Linear(dim, internal)
        self.W_dt = nn.Linear(dim, internal)
        # Log-parameterized A, negative for stability (decaying state)
        self.A_log = nn.Parameter(-torch.rand(internal))
        self.W_V = nn.Linear(dim, internal)

    def forward(self, x: torch.Tensor):
        dt = F.softplus(self.W_dt(x))               # Δ: [batch, L, internal]
        A_bar = torch.exp(dt * self.A_log)           # exp(Δ·A): state transition
        B_bar = dt * self.W_B(x)                     # Δ·B: discretized input
        C = self.W_C(x)
        V = self.W_V(x)
        return {'B': B_bar, 'C': C, 'S': A_bar, 'V': V}


class Featurizer11(FeaturizerBase):
    """Class 11: CfC / Liquid Neural Network with complementary gating.

    Complementary forget/input gate:
      gate = σ(W_gate(x))
      A = gate               — forget (how much state to retain)
      B = (1 - gate) · g(x)  — input  (how much new input to absorb)

    Key property: A + B_coefficient = 1 per element, so the state update
    h_t = gate · h_{t-1} + (1-gate) · g(x_t) is a convex interpolation
    between memory and new input.
    Paired with semi-separable token mixing + diagonal channel mixing.
    """

    def __init__(self, dim: int, num_heads: int = 1, expansion: int = 16):
        internal = dim * expansion
        super().__init__(dim, internal, num_heads)
        self.expansion = expansion
        self.W_gate = nn.Linear(dim, internal)
        self.W_input = nn.Linear(dim, internal)
        self.W_C = nn.Linear(dim, internal)
        self.W_V = nn.Linear(dim, internal)

    def forward(self, x: torch.Tensor):
        gate = torch.sigmoid(self.W_gate(x))         # σ(f(x)) ∈ (0,1)
        B = (1 - gate) * self.W_input(x)             # (1-σ) · g(x)
        C = self.W_C(x)
        V = self.W_V(x)
        return {'B': B, 'C': C, 'S': gate, 'V': V}


FEATURIZER_REGISTRY = {
    1: Featurizer1,
    2: Featurizer2,
    3: Featurizer3,
    4: Featurizer4,
    5: Featurizer5,
    6: Featurizer6,
    7: Featurizer7,
    8: Featurizer8,
    9: Featurizer9,
    10: Featurizer10,
    11: Featurizer11,
}


# =============================================================================
# Token Mixing Weight Generator (Unified Interface)
# =============================================================================

class TokenMixWeightGenerator(nn.Module):
    """Generates L×L token mixing matrix T from features (B, C, S).

    Core formula: T_ij = C_i · M_ij · B_j

    The M_ij structure depends on type:
    - DIAGONAL:      M_ij = δ_ij
    - LOW_RANK:      M_ij = 1
    - TOEPLITZ:      M_ij = K_{i-j}
    - SEMI_SEPARABLE: M_ij = ∏_{k=j+1}^{i-1} A_k
    """

    def __init__(self, internal_dim: int, mix_type: TokenMixType,
                 num_heads: int = 1, kernel_size: int = 3,
                 causal: bool = True, use_softmax: bool = False):
        super().__init__()
        self.internal_dim = internal_dim
        self.mix_type = mix_type
        self.num_heads = num_heads
        self.head_dim = internal_dim // num_heads
        self.kernel_size = kernel_size
        self.causal = causal
        self.use_softmax = use_softmax

    def forward(self, B: torch.Tensor, C: torch.Tensor,
                S: torch.Tensor) -> torch.Tensor:
        """Generate token mixing matrix T from features.

        Args:
            B: [batch, seq_len, internal_dim] - input projection
            C: [batch, seq_len, internal_dim] - output projection
            S: structure param (shape varies by type)

        Returns:
            T: [batch, num_heads, seq_len, seq_len]
        """
        batch, seq_len, _ = B.shape
        H = self.num_heads
        d = self.head_dim

        # Reshape to heads: [batch, heads, seq_len, head_dim]
        B_h = B.view(batch, seq_len, H, d).permute(0, 2, 1, 3)
        C_h = C.view(batch, seq_len, H, d).permute(0, 2, 1, 3)

        if self.mix_type == TokenMixType.DIAGONAL:
            T = self._diagonal(B_h, C_h, S)
        elif self.mix_type == TokenMixType.LOW_RANK:
            T = self._low_rank(B_h, C_h, S)
        elif self.mix_type == TokenMixType.TOEPLITZ:
            T = self._toeplitz(B_h, C_h, S)
        elif self.mix_type == TokenMixType.SEMI_SEPARABLE:
            T = self._semi_separable(B_h, C_h, S)

        # Apply causal mask if needed
        if self.causal and self.mix_type != TokenMixType.DIAGONAL:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=T.device), diagonal=1).bool()
            T = T.masked_fill(mask, 0.0 if not self.use_softmax else float('-inf'))

        return T

    def _diagonal(self, B_h, C_h, S):
        """T_ij = (C_i ⊙ B_i) · δ_ij"""
        diag_vals = (C_h * B_h).sum(dim=-1)  # [batch, H, L]
        return torch.diag_embed(diag_vals)    # [batch, H, L, L]

    def _low_rank(self, B_h, C_h, S):
        """T_ij = C_i · B_j^T / √d"""
        scale = math.sqrt(self.head_dim)
        return torch.matmul(C_h, B_h.transpose(-2, -1)) / scale

    def _toeplitz(self, B_h, C_h, S):
        """T_ij = C_i · K_{i-j} · B_j — built without in-place ops for autograd."""
        batch, H, L, d = B_h.shape
        # S: [batch, internal_dim, kernel_size]
        K_avg = S.mean(dim=1)  # [batch, kernel_size]

        # Build each entry of T out-of-place
        rows = []
        for i in range(L):
            row = []
            for j in range(L):
                k = i - j
                if k < 0 or k >= self.kernel_size:
                    row.append(torch.zeros(batch, H, device=B_h.device, dtype=B_h.dtype))
                else:
                    c_i = C_h[:, :, i, :]  # [batch, H, d]
                    b_j = B_h[:, :, j, :]
                    cb = (c_i * b_j).sum(dim=-1)  # [batch, H]
                    k_val = K_avg[:, k]  # [batch]
                    row.append(cb * k_val.unsqueeze(1))
            rows.append(torch.stack(row, dim=-1))
        T = torch.stack(rows, dim=-2)  # [batch, H, L, L]
        return T

    def _semi_separable(self, B_h, C_h, S):
        """T_ij = C_i · (∏A) · B_j — built without in-place ops for autograd."""
        batch, H, L, d = B_h.shape
        A_h = S.view(batch, L, H, d).permute(0, 2, 1, 3)  # [batch, H, L, d]

        # Cumulative product of A (out-of-place)
        cum_A = torch.cumprod(A_h, dim=2)  # [batch, H, L, d]

        # Build T entries as list, then stack
        rows = []
        for i in range(L):
            row = []
            for j in range(L):
                if j > i:
                    row.append(torch.zeros(batch, H, device=B_h.device, dtype=B_h.dtype))
                else:
                    c_i = C_h[:, :, i, :]
                    b_j = B_h[:, :, j, :]
                    if i == j:
                        prod = torch.ones_like(c_i)
                    elif j == 0:
                        prod = cum_A[:, :, i-1]
                    else:
                        prod = cum_A[:, :, i-1] / (cum_A[:, :, j-1] + 1e-8)
                    row.append((c_i * prod * b_j).sum(dim=-1))
            rows.append(torch.stack(row, dim=-1))  # [batch, H, L]
        T = torch.stack(rows, dim=-2)  # [batch, H, L, L]
        return T


# =============================================================================
# Channel Mixing Weight Generator
# =============================================================================

class ChannelMixWeightGeneratorOld(nn.Module):
    """Generates d_out × d_in channel mixing matrix."""

    def __init__(self, dim_in: int, dim_out: int, mix_type: ChannelMixType,
                 num_heads: int = 1):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.mix_type = mix_type
        self.num_heads = num_heads

        if mix_type == ChannelMixType.DIAGONAL:
            self.weight = nn.Parameter(torch.ones(min(dim_in, dim_out)))

        elif mix_type == ChannelMixType.DENSE:
            self.weight = nn.Parameter(torch.randn(dim_out, dim_in) * 0.02)

        elif mix_type == ChannelMixType.GROUPED:
            assert dim_in % num_heads == 0 and dim_out % num_heads == 0
            h_in, h_out = dim_in // num_heads, dim_out // num_heads
            self.weight = nn.Parameter(torch.randn(num_heads, h_out, h_in) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply static channel mixing: x @ W^T.

        Args:
            x: [batch, seq_len, dim_in]
        Returns:
            y: [batch, seq_len, dim_out]
        """
        if self.mix_type == ChannelMixType.DIAGONAL:
            d = len(self.weight)
            W = torch.diag(self.weight)
            if self.dim_out != d or self.dim_in != d:
                W = F.pad(W, (0, max(0, self.dim_in - d), 0, max(0, self.dim_out - d)))
            W = W[:self.dim_out, :self.dim_in]
        elif self.mix_type == ChannelMixType.DENSE:
            W = self.weight
        elif self.mix_type == ChannelMixType.GROUPED:
            W = torch.block_diag(*[self.weight[h] for h in range(self.num_heads)])
        return F.linear(x, W)


class ChannelMixGenerator(nn.Module):
    """Input-dependent channel mixing: W(x) is computed from input.

    Unlike ChannelMixWeightGenerator where W is a static nn.Parameter,
    here W is generated per-token from the input, making the full
    T_ij^{αβ}(x) truly input-varying in BOTH token and channel dimensions.

    Generates a [batch, seq_len, dim_out, dim_in] matrix — a different
    W per position, computed by a lightweight network from x.
    """

    def __init__(self, dim_in: int, dim_out: int, mix_type: ChannelMixType,
                 num_heads: int = 1, bottleneck: int = None):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.mix_type = mix_type
        self.num_heads = num_heads

        # Input dim for the generator is dim_in (takes the mixed features as context)
        # Bottleneck keeps param count manageable
        neck = bottleneck or max(dim_in // 4, 1)

        if mix_type == ChannelMixType.DIAGONAL:
            # Generate dim_out diagonal entries from input
            d = min(dim_in, dim_out)
            self.net = nn.Sequential(
                nn.Linear(dim_in, neck),
                nn.SiLU(),
                nn.Linear(neck, d),
            )

        elif mix_type == ChannelMixType.DENSE:
            # Generate full dim_out × dim_in matrix from input
            self.net = nn.Sequential(
                nn.Linear(dim_in, neck),
                nn.SiLU(),
                nn.Linear(neck, dim_out * dim_in),
            )

        elif mix_type == ChannelMixType.GROUPED:
            # Generate block-diagonal: h blocks of (dim_out/h × dim_in/h)
            assert dim_in % num_heads == 0 and dim_out % num_heads == 0
            h_in = dim_in // num_heads
            h_out = dim_out // num_heads
            self.net = nn.Sequential(
                nn.Linear(dim_in, neck),
                nn.SiLU(),
                nn.Linear(neck, num_heads * h_out * h_in),
            )

    def inner_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate input-dependent channel mixing matrix.

        Args:
            x: [batch, seq_len, dim_in] — the mixed features (after token mixing)

        Returns:
            W: [batch, seq_len, dim_out, dim_in] — per-position mixing matrix
        """
        if self.mix_type == ChannelMixType.DIAGONAL:
            return self._diagonal(x)
        elif self.mix_type == ChannelMixType.DENSE:
            return self._dense(x)
        elif self.mix_type == ChannelMixType.GROUPED:
            return self._grouped(x)

    def _diagonal(self, x: torch.Tensor) -> torch.Tensor:
        batch, L, _ = x.shape
        d = min(self.dim_in, self.dim_out)
        diag_vals = self.net(x)  # [batch, L, d]
        W = torch.diag_embed(diag_vals)  # [batch, L, d, d]
        if self.dim_out != d or self.dim_in != d:
            W = F.pad(W, (0, max(0, self.dim_in - d), 0, max(0, self.dim_out - d)))
        return W[:, :, :self.dim_out, :self.dim_in]

    def _dense(self, x: torch.Tensor) -> torch.Tensor:
        batch, L, _ = x.shape
        W = self.net(x)  # [batch, L, dim_out * dim_in]
        return W.view(batch, L, self.dim_out, self.dim_in)

    def _grouped(self, x: torch.Tensor) -> torch.Tensor:
        batch, L, _ = x.shape
        H = self.num_heads
        h_in = self.dim_in // H
        h_out = self.dim_out // H
        blocks = self.net(x).view(batch, L, H, h_out, h_in)  # [B, L, H, ho, hi]

        # Assemble block-diagonal per position
        W = torch.zeros(batch, L, self.dim_out, self.dim_in,
                        device=x.device, dtype=x.dtype)
        for h in range(H):
            r = h * h_out
            c = h * h_in
            W[:, :, r:r+h_out, c:c+h_in] = blocks[:, :, h]
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate W(x) and apply it: output_i = W(x_i) @ x_i.

        Args:
            x: [batch, seq_len, dim_in]
        Returns:
            y: [batch, seq_len, dim_out]
        """
        W = self.inner_forward(x)  # [batch, L, dim_out, dim_in]
        # Per-position matmul: y_i = W_i @ x_i
        return torch.einsum('bloi,bli->blo', W, x)


# =============================================================================
# Sparsity Mask (5-tuple axis 4) — NOT attached to UnifiedLIV
# =============================================================================

class SparsityMask(nn.Module):
    """Applies sparsity pattern to token mixing matrix T: [batch, heads, L, L].

    Args:
        sparsity_type: Type of sparsity to apply.
        bandwidth: For BANDED, half-width around diagonal.
        top_k: For TOP_K, entries to keep per row.
    """

    def __init__(self, sparsity_type: SparsityType = SparsityType.NONE,
                 bandwidth: int = 64, top_k: int = 32, use_softmax: bool = False):
        super().__init__()
        self.sparsity_type = sparsity_type
        self.bandwidth = bandwidth
        self.top_k = top_k
        self.use_softmax = use_softmax

    def forward(self, T: torch.Tensor, clean_len: int = None) -> torch.Tensor:
        if self.sparsity_type == SparsityType.NONE:
            return T

        if self.sparsity_type == SparsityType.CAUSAL:
            L = T.size(-1)
            mask = torch.triu(torch.ones(L, L, device=T.device), diagonal=1).bool()
            return T.masked_fill(mask, 0.0)

        if self.sparsity_type == SparsityType.BANDED:
            L = T.size(-1)
            rows = torch.arange(L, device=T.device).unsqueeze(1)
            cols = torch.arange(L, device=T.device).unsqueeze(0)
            mask = (rows - cols).abs() > self.bandwidth
            return T.masked_fill(mask, 0.0)

        if self.sparsity_type == SparsityType.TOP_K:
            k = min(self.top_k, T.size(-1))
            vals, idx = T.topk(k, dim=-1)
            sparse = torch.zeros_like(T)
            sparse.scatter_(-1, idx, vals)
            return sparse

        if self.sparsity_type == SparsityType.TIDAR_HYBRID:
            L = T.size(-1)
            cl = clean_len if clean_len is not None else L // 2
            fill = float('-inf') if self.use_softmax else 0.0
            mask = torch.zeros(L, L, device=T.device, dtype=torch.bool)
            # Clean prefix: causal among themselves
            causal_mask = torch.triu(
                torch.ones(cl, cl, device=T.device), diagonal=1
            ).bool()
            mask[:cl, :cl] = causal_mask
            # Clean tokens cannot attend to mask tokens
            mask[:cl, cl:] = True
            # Mask tokens attend to everything (prefix + each other): no mask
            return T.masked_fill(mask, fill)

        return T


# Usage example (NOT wired into UnifiedLIV):
#
#   sparsity = SparsityMask(SparsityType.BANDED, bandwidth=128)
#   T = token_mix_gen(B, C, S)        # [batch, heads, L, L]
#   T = sparsity(T)                   # apply band mask
#   mixed = T @ V_h                   # sparse token mixing
#
#   # Or combined causal + banded:
#   sparsity = SparsityMask(SparsityType.BANDED, bandwidth=64)
#   T = token_mix_gen(B, C, S)        # already causal from token_mix_gen
#   T = sparsity(T)                   # further restrict to band


# =============================================================================
# Unified LIV Operator
# =============================================================================

class UnifiedLIV(nn.Module):
    """Unified LIV: Y = C_mix @ (T @ V)

    Where T = token_mixing_weight(B, C, S) is ALWAYS an explicit L×L matrix.

    Accepts any FeaturizerBase subclass. The featurizer determines:
    - What B, C, S, V are and how they're computed
    - The internal_dim (may differ from dim due to expansion)

    Token/channel mixing weight generators operate on internal_dim,
    with channel mixing projecting internal_dim → dim as output.
    """

    def __init__(self, dim: int,
                 token_mix_type: TokenMixType = TokenMixType.LOW_RANK,
                 channel_mix_type: ChannelMixType = ChannelMixType.GROUPED,
                 num_heads: int = 8,
                 kernel_size: int = 3,
                 causal: bool = True,
                 use_softmax: bool = True,
                 featurizer: FeaturizerBase = None,
                 featurizer_cls: int = None,
                 expansion: int = 1,
                 nonlinearity: NonLinearity = NonLinearity.NONE,
                 sparsity_type: SparsityType = None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.nonlinearity = nonlinearity

        # Build featurizer
        if featurizer is not None:
            self.featurizer = featurizer
        elif featurizer_cls is not None and featurizer_cls in FEATURIZER_REGISTRY:
            cls = FEATURIZER_REGISTRY[featurizer_cls]
            # Pass appropriate args based on class
            if featurizer_cls in (1, 2, 3, 4):
                self.featurizer = cls(dim, num_heads)
            elif featurizer_cls in (5, 6, 10, 11):
                self.featurizer = cls(dim, num_heads, expansion)
            elif featurizer_cls == 7:
                self.featurizer = cls(dim, num_heads, kernel_size)
            elif featurizer_cls == 8:
                self.featurizer = cls(dim, num_heads, kernel_size)
            elif featurizer_cls == 9:
                self.featurizer = cls(dim, num_heads, expansion)
        else:
            self.featurizer = UnifiedFeaturizer(dim, num_heads, token_mix_type, kernel_size)

        internal_dim = self.featurizer.internal_dim
        # For expanded dims, use 1 head so the full expanded dim is one "head"
        tmix_heads = num_heads if internal_dim == dim else 1

        # Softmax: via nonlinearity enum or legacy use_softmax flag
        _use_softmax = (nonlinearity == NonLinearity.SOFTMAX) or \
                       (token_mix_type == TokenMixType.LOW_RANK and use_softmax)

        self.token_mix_gen = TokenMixWeightGenerator(
            internal_dim, token_mix_type, tmix_heads, kernel_size, causal,
            use_softmax=_use_softmax,
        )

        # Static channel mixing: paper-accurate W as nn.Parameter (ChannelMixWeightGeneratorOld).
        # ChannelMixGenerator (input-dependent W(x)) is kept in the file for experimentation
        # but is not the paper's definition — paper W is a fixed learned matrix.
        self.channel_mix_gen = ChannelMixWeightGeneratorOld(
            internal_dim, dim, channel_mix_type,
            num_heads if internal_dim == dim else 1,
        )

        self.token_mix_type = token_mix_type
        self.channel_mix_type = channel_mix_type
        self._tmix_heads = tmix_heads
        self._use_softmax = _use_softmax

        # External sparsity mask (e.g., TiDAR hybrid causal+bidirectional)
        if sparsity_type is not None:
            self.sparsity = SparsityMask(
                sparsity_type, use_softmax=_use_softmax,
            )
            if sparsity_type == SparsityType.TIDAR_HYBRID:
                self.token_mix_gen.causal = False
        else:
            self.sparsity = None

    def forward(self, x: torch.Tensor, clean_len: int = None) -> torch.Tensor:
        """Compute Y = σ(T(x) · V) through W(x).

        Steps: featurize → token mix → sparsity → softmax → nonlinearity → channel mix
        """
        batch, seq_len, dim = x.shape

        # 1. Featurizer: x -> (B, C, S, V) at internal_dim
        feat = self.featurizer(x)
        B, C, S, V = feat['B'], feat['C'], feat['S'], feat['V']

        # 2. Token mixing matrix T: [batch, heads, L, L]
        T = self.token_mix_gen(B, C, S)

        # 3. Sparsity mask (e.g., TiDAR hybrid causal+bidirectional)
        if self.sparsity is not None:
            T = self.sparsity(T, clean_len=clean_len)

        # 4. Softmax normalization over unmasked positions
        if self._use_softmax:
            T = F.softmax(T, dim=-1)

        # 5. Element-wise nonlinearity on T (non-softmax types only)
        if self.nonlinearity not in (NonLinearity.NONE, NonLinearity.SOFTMAX):
            T = _NONLINEARITY_FN[self.nonlinearity](T)

        # 4. Reshape V for multi-head, apply σ(T) @ V
        H = self._tmix_heads
        head_dim = self.featurizer.internal_dim // H
        V_h = V.view(batch, seq_len, H, head_dim).permute(0, 2, 1, 3)
        mixed = torch.matmul(T, V_h)  # [batch, H, L, head_dim]
        mixed = mixed.permute(0, 2, 1, 3).reshape(batch, seq_len, self.featurizer.internal_dim)

        # 5. Input-dependent channel mixing: internal_dim → dim
        output = self.channel_mix_gen(mixed)

        return output


# =============================================================================
# Factory & Presets
# =============================================================================

def create_liv(dim: int, token_mix: int, channel_mix: int, **kwargs) -> UnifiedLIV:
    """Create UnifiedLIV from integer codes.

    token_mix: 1=diagonal, 2=low_rank, 3=toeplitz, 4=semi_separable
    channel_mix: 1=diagonal, 2=dense, 3=grouped
    """
    return UnifiedLIV(
        dim,
        TokenMixType(token_mix),
        ChannelMixType(channel_mix),
        **kwargs
    )


# --- Attention variants ---

def SA1(dim: int, num_heads: int = 8, **kw) -> UnifiedLIV:
    """SA-1 (12123): Standard MHA. Featurizer 1 + low-rank + grouped."""
    return UnifiedLIV(dim, TokenMixType.LOW_RANK, ChannelMixType.GROUPED,
                      num_heads=num_heads, featurizer_cls=1, **kw)

def SA2(dim: int, num_heads: int = 8, **kw) -> UnifiedLIV:
    """SA-2 (22123): Conv-pre attention. Featurizer 2 + low-rank + grouped."""
    return UnifiedLIV(dim, TokenMixType.LOW_RANK, ChannelMixType.GROUPED,
                      num_heads=num_heads, featurizer_cls=2, **kw)

def SA3(dim: int, num_heads: int = 8, **kw) -> UnifiedLIV:
    """SA-3 (32123): MQA. Featurizer 3 + low-rank + grouped."""
    return UnifiedLIV(dim, TokenMixType.LOW_RANK, ChannelMixType.GROUPED,
                      num_heads=num_heads, featurizer_cls=3, **kw)

def SA4(dim: int, num_heads: int = 8, **kw) -> UnifiedLIV:
    """SA-4 (42123): GQA. Featurizer 4 + low-rank + grouped."""
    return UnifiedLIV(dim, TokenMixType.LOW_RANK, ChannelMixType.GROUPED,
                      num_heads=num_heads, featurizer_cls=4, **kw)

# --- Recurrence variants ---

def Rec1(dim: int, expansion: int = 16, **kw) -> UnifiedLIV:
    """Rec-1 (54111): Mamba-like SSM. Featurizer 5 + semi-sep + diagonal."""
    return UnifiedLIV(dim, TokenMixType.SEMI_SEPARABLE, ChannelMixType.DIAGONAL,
                      num_heads=1, featurizer_cls=5, expansion=expansion,
                      use_softmax=False, **kw)

def Rec2(dim: int, expansion: int = 2, **kw) -> UnifiedLIV:
    """Rec-2 (64111): Compact SSM. Featurizer 6 + semi-sep + diagonal."""
    return UnifiedLIV(dim, TokenMixType.SEMI_SEPARABLE, ChannelMixType.DIAGONAL,
                      num_heads=1, featurizer_cls=6, expansion=expansion,
                      use_softmax=False, **kw)

def Rec3(dim: int, expansion: int = 16, **kw) -> UnifiedLIV:
    """Rec-3: SSM with discretization. Featurizer 10 + semi-sep + diagonal."""
    return UnifiedLIV(dim, TokenMixType.SEMI_SEPARABLE, ChannelMixType.DIAGONAL,
                      num_heads=1, featurizer_cls=10, expansion=expansion,
                      use_softmax=False, **kw)

def Rec4(dim: int, expansion: int = 16, **kw) -> UnifiedLIV:
    """Rec-4: CfC/LNN with complementary gating. Featurizer 11 + semi-sep + diagonal."""
    return UnifiedLIV(dim, TokenMixType.SEMI_SEPARABLE, ChannelMixType.DIAGONAL,
                      num_heads=1, featurizer_cls=11, expansion=expansion,
                      use_softmax=False, **kw)

def Rec5(dim: int, expansion: int = 2, **kw) -> UnifiedLIV:
    """Rec-5: Compact CfC/LNN. Same gating as Rec-4 but expansion=2 (matches Rec-2 size).

    Analogous to Rec-2 being the compact version of Rec-1:
      Rec-1 (SSM, expansion=16)  ↔  Rec-2 (SSM, expansion=2)
      Rec-4 (CfC, expansion=16)  ↔  Rec-5 (CfC, expansion=2)

    Allows CfC to compete fairly alongside attention/conv in DMamba temporal paths
    without the 16.9M param explosion of Rec-4.
    """
    return UnifiedLIV(dim, TokenMixType.SEMI_SEPARABLE, ChannelMixType.DIAGONAL,
                      num_heads=1, featurizer_cls=11, expansion=expansion,
                      use_softmax=False, **kw)

# --- Convolution variants ---

def GConv1(dim: int, kernel_size: int = 3, **kw) -> UnifiedLIV:
    """GConv-1 (73111): Gated short conv. Featurizer 7 + toeplitz + diagonal."""
    return UnifiedLIV(dim, TokenMixType.TOEPLITZ, ChannelMixType.DIAGONAL,
                      num_heads=1, featurizer_cls=7, kernel_size=kernel_size,
                      use_softmax=False, **kw)

def GConv2(dim: int, kernel_size: int = 64, **kw) -> UnifiedLIV:
    """GConv-2 (83111): Gated long conv. Featurizer 8 + toeplitz + diagonal."""
    return UnifiedLIV(dim, TokenMixType.TOEPLITZ, ChannelMixType.DIAGONAL,
                      num_heads=1, featurizer_cls=8, kernel_size=kernel_size,
                      use_softmax=False, **kw)

# --- Memoryless ---

def GMemless(dim: int, expansion: int = 4, **kw) -> UnifiedLIV:
    """GMemless (91142): SwiGLU/FFN. Featurizer 9 + diagonal + dense."""
    return UnifiedLIV(dim, TokenMixType.DIAGONAL, ChannelMixType.DENSE,
                      num_heads=1, featurizer_cls=9, expansion=expansion,
                      use_softmax=False, **kw)


# =============================================================================
# LIV Block with Residual
# =============================================================================

class UnifiedLIVBlock(nn.Module):
    """Pre-norm residual: y = LIV(norm(x)) + x"""

    def __init__(self, dim: int, liv: UnifiedLIV):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.liv = liv

    def forward(self, x: torch.Tensor, clean_len: int = None) -> torch.Tensor:
        return x + self.liv(self.norm(x), clean_len=clean_len)


class DifferentialLIV(nn.Module):
    """Differential variant: y = LIV_1(x) - LIV_2(x)"""

    def __init__(self, build_fn, *args, **kwargs):
        super().__init__()
        self.liv1 = build_fn(*args, **kwargs)
        self.liv2 = build_fn(*args, **kwargs)

    def forward(self, x: torch.Tensor, clean_len: int = None) -> torch.Tensor:
        return self.liv1(x, clean_len=clean_len) - self.liv2(x, clean_len=clean_len)


class STARBackbone(nn.Module):
    """Stack of UnifiedLIV blocks.

    configs: list of (featurizer_cls, token_mix, channel_mix) tuples
    """

    def __init__(self, configs: list, dim: int, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList()
        for cfg in configs:
            if len(cfg) == 3:
                feat_cls, t, c = cfg
                liv = UnifiedLIV(
                    dim, TokenMixType(t), ChannelMixType(c),
                    featurizer_cls=feat_cls, **kwargs
                )
            else:
                t, c = cfg
                liv = create_liv(dim, t, c, **kwargs)
            self.blocks.append(UnifiedLIVBlock(dim, liv))

    def forward(self, x: torch.Tensor, clean_len: int = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, clean_len=clean_len)
        return x