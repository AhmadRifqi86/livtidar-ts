"""
LIV (Linear Input-Varying) Framework Building Blocks

Based on STAR: Synthesis of Tailored Architectures (arXiv:2411.17800v1)

Core equation: Y = T(x) · x
where T(x) is a structured matrix computed from the input,
decomposed into token mixing (T_ij in R^(L×L)) and channel mixing (T_ij in R^(d×d)).

Operator genome: 5 integers
  1. Featurizer class (1-9)
  2. Token mixing: 1=diagonal, 2=low-rank, 3=scaled-toeplitz, 4=semi-separable
  3. Sparsity: 1=none, 2=banded
  4. Nonlinearity: 1=none, 2=softmax, 3=relu, 4=swish
  5. Channel mixing: 1=diagonal, 2=dense, 3=grouped
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# Nonlinearities
# =============================================================================

NONLINEARITIES = {
    1: lambda x: x,                    # none
    2: lambda x: F.softmax(x, dim=-1), # softmax over last (token) dim
    3: lambda x: F.relu(x),
    4: lambda x: F.silu(x),            # swish
}


# =============================================================================
# Token Mixing
# =============================================================================

class DiagonalTokenMixing(nn.Module):
    """Token mixing = diagonal: T_ij = c_i * delta_ij
    Each token is scaled independently — no temporal interaction.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, B=None, C=None) -> torch.Tensor:
        # x: [batch, seq_len, dim]
        # Diagonal: y_i = scale * x_i (per-channel scaling, no cross-token)
        return x * self.scale


class LowRankTokenMixing(nn.Module):
    """Token mixing = low-rank: T_ij = C_i · B_j^T (outer product)
    This gives a dense L×L attention-like matrix from rank-d factors.
    B, C are provided by the featurizer (query/key projections).
    Nonlinearity (e.g. softmax) is applied to the score matrix.
    """

    def __init__(self, nonlinearity: int = 2, causal: bool = False):
        super().__init__()
        self.nonlinearity_id = nonlinearity
        self.causal = causal

    def forward(self, x: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # B: [batch, (heads), seq_len, head_dim] — "keys"
        # C: [batch, (heads), seq_len, head_dim] — "queries"
        # x here is V: [batch, (heads), seq_len, head_dim] — "values"
        # Score matrix: S_ij = C_i · B_j  (low-rank structure)
        scale = math.sqrt(B.shape[-1])
        scores = torch.matmul(C, B.transpose(-2, -1)) / scale  # [..., L, L]

        if self.causal:
            L = scores.shape[-1]
            mask = torch.triu(torch.ones(L, L, device=scores.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float('-inf'))

        nl = NONLINEARITIES[self.nonlinearity_id]
        scores = nl(scores)

        return torch.matmul(scores, x)  # [..., L, head_dim]


class SemiSeparableTokenMixing(nn.Module):
    """Token mixing = semi-separable (causal recurrence):
    T_ij = C_i · prod(A_k, k=j+1..i-1) · B_j  for i >= j, else 0

    Implemented as a sequential scan:
      h_t = A_t * h_{t-1} + B_t * x_t
      y_t = C_t * h_t

    A, B, C are input-dependent (provided by featurizer).
    Channel mixing is diagonal (each channel evolves independently).
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                A: torch.Tensor) -> torch.Tensor:
        # B: [batch, seq_len, state_dim] — input gate
        # C: [batch, seq_len, state_dim] — output gate
        # A: [batch, seq_len, state_dim] — state transition (should be in [0,1])
        # x is unused directly; B already encodes the input projection
        batch, seq_len, state_dim = B.shape

        h = torch.zeros(batch, state_dim, device=B.device, dtype=B.dtype)
        outputs = []
        for t in range(seq_len):
            h = A[:, t] * h + B[:, t]
            outputs.append(C[:, t] * h)

        return torch.stack(outputs, dim=1)  # [batch, seq_len, state_dim]


class ScaledToeplitzTokenMixing(nn.Module):
    """Token mixing = scaled Toeplitz (convolution):
    T_ij = C_i · K_{i-j} · B_j  where K is a convolution kernel

    The Toeplitz structure means constant diagonals — translation equivariance.
    Can be computed efficiently via FFT for long kernels.

    K is either provided explicitly (short kernel) or implicitly (long kernel).
    B, C are input-dependent scaling factors from featurizer.
    """

    def __init__(self, kernel_size: int, dim: int, use_fft: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.use_fft = use_fft
        # Per-channel kernels
        self.kernel = nn.Parameter(torch.randn(dim, kernel_size) * 0.02)

    def forward(self, x: torch.Tensor, B: torch.Tensor = None,
                C: torch.Tensor = None, kernel: torch.Tensor = None) -> torch.Tensor:
        # x: [batch, seq_len, dim]
        # B: [batch, seq_len, dim] — input gate (optional)
        # C: [batch, seq_len, dim] — output gate (optional)
        # kernel: [dim, kernel_size] — override kernel (for implicit parameterization)

        if B is not None:
            x = x * B

        K = kernel if kernel is not None else self.kernel  # [dim, kernel_size]

        # Depthwise causal convolution
        batch, seq_len, dim = x.shape
        x_t = x.transpose(1, 2)  # [batch, dim, seq_len]

        if self.use_fft and seq_len > self.kernel_size * 4:
            # FFT-based convolution for long kernels
            n_fft = 2 * seq_len
            X_f = torch.fft.rfft(x_t, n=n_fft, dim=-1)
            # Pad kernel to match
            K_padded = F.pad(K, (0, n_fft - K.shape[-1]))  # [dim, n_fft]
            K_f = torch.fft.rfft(K_padded, dim=-1)
            y_t = torch.fft.irfft(X_f * K_f.unsqueeze(0), n=n_fft, dim=-1)[..., :seq_len]
        else:
            # Direct convolution (causal padding)
            padding = self.kernel_size - 1
            x_padded = F.pad(x_t, (padding, 0))  # causal: pad left
            # Groups=dim for depthwise
            K_conv = K.unsqueeze(1)  # [dim, 1, kernel_size]
            y_t = F.conv1d(x_padded, K_conv, groups=dim)

        y = y_t.transpose(1, 2)  # [batch, seq_len, dim]

        if C is not None:
            y = y * C

        return y


# =============================================================================
# Channel Mixing
# =============================================================================

class DiagonalChannelMixing(nn.Module):
    """Channel mixing = diagonal: each channel processed independently.
    T_ij^{alpha,beta} = 0 for alpha != beta.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class DenseChannelMixing(nn.Module):
    """Channel mixing = dense: full cross-channel interaction.
    Implemented as linear projection (optionally with expansion).
    """

    def __init__(self, dim: int, out_dim: int = None):
        super().__init__()
        out_dim = out_dim or dim
        self.proj = nn.Linear(dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class GroupedChannelMixing(nn.Module):
    """Channel mixing = grouped (block-diagonal / multi-head).
    Channels are split into groups (heads); mixing happens within each group
    independently, then concatenated. Output projection merges heads.
    """

    def __init__(self, dim: int, num_heads: int, out_dim: int = None):
        super().__init__()
        out_dim = out_dim or dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0
        self.out_proj = nn.Linear(dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, dim]
        # The grouped structure is implicit: the token mixing operates
        # per-head (block-diagonal), then we project back.
        return self.out_proj(x)


# =============================================================================
# Sparsity
# =============================================================================

def apply_banded_mask(scores: torch.Tensor, bandwidth: int) -> torch.Tensor:
    """Apply banded sparsity to an L×L score matrix (keep only nearby tokens)."""
    L = scores.shape[-1]
    idx = torch.arange(L, device=scores.device)
    mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() > bandwidth
    return scores.masked_fill(mask, float('-inf'))


# =============================================================================
# Featurizers
# =============================================================================

class Featurizer1(nn.Module):
    """Class 1: Dense channel + Diagonal token, 3 feature groups.

    Standard linear projection producing Q, K, V from input.
    Token mixing in featurizer: diagonal (each token projected independently).
    Channel mixing in featurizer: dense (linear layer mixes all channels).
    Used by SA-1 (standard attention).
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0
        self.W_Q = nn.Linear(dim, dim)
        self.W_K = nn.Linear(dim, dim)
        self.W_V = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor):
        """Returns Q, K, V reshaped for multi-head attention.
        Each output: [batch, num_heads, seq_len, head_dim]
        """
        B, L, D = x.shape
        Q = self.W_Q(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        return {'Q': Q, 'K': K, 'V': V}


class Featurizer9(nn.Module):
    """Class 9: Dense channel + Diagonal token, 2 feature groups (SwiGLU).

    Two parallel dense projections: gate and value.
    Token mixing in featurizer: diagonal (each token independent).
    Channel mixing in featurizer: dense (linear mixes channels).
    Used by GMemless (SwiGLU / FFN).
    """

    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        hidden = dim * expansion
        self.W_gate = nn.Linear(dim, hidden)
        self.W_value = nn.Linear(dim, hidden)

    def forward(self, x: torch.Tensor):
        """Returns gate and value projections.
        Each: [batch, seq_len, hidden_dim]
        """
        return {
            'gate': self.W_gate(x),
            'value': self.W_value(x),
        }


# =============================================================================
# Base LIV Operator
# =============================================================================

class LIVOperator(nn.Module):
    """Base class for Linear Input-Varying operators.

    Core equation: y_i^alpha = sum_j sum_beta T_ij^{alpha,beta}(x) * x_j^beta

    The T(x) matrix is decomposed into:
      - Token mixing structure (how positions i,j interact)
      - Channel mixing structure (how channels alpha,beta interact)

    T(x) is input-dependent, computed by the featurizer neural network,
    enabling the same framework to represent attention, recurrence,
    convolution, or memoryless (FFN) operations.

    Subclasses define specific combinations via the operator genome:
      (featurizer, token_mixing, sparsity, nonlinearity, channel_mixing)
    """

    def __init__(self, dim: int, genome: tuple = None):
        super().__init__()
        self.dim = dim
        self.genome = genome  # (featurizer, token_mix, sparsity, nonlinearity, channel_mix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Y = T(x) · x.

        Args:
            x: [batch, seq_len, dim]
        Returns:
            y: [batch, seq_len, dim]
        """
        raise NotImplementedError


class LIVBlock(nn.Module):
    """Pre-norm residual wrapper: y = LIV(norm(x)) + x"""

    def __init__(self, dim: int, liv: LIVOperator):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.liv = liv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.liv(self.norm(x))


class DifferentialLIV(LIVOperator):
    """Differential variant: y = LIV_1(x) - LIV_2(x)
    Two identical-architecture LIVs with separate parameters.
    """

    def __init__(self, liv_cls, *args, **kwargs):
        dim = args[0] if args else kwargs.get('dim')
        super().__init__(dim)
        self.liv1 = liv_cls(*args, **kwargs)
        self.liv2 = liv_cls(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.liv1(x) - self.liv2(x)


# =============================================================================
# Concrete LIV Classes
# =============================================================================

class SA1(LIVOperator):
    """Standard Softmax Attention — Genome: 12123
    Featurizer 1, Low-rank token mixing, No sparsity, Softmax, Grouped channel mixing.
    """

    def __init__(self, dim: int, num_heads: int = 8, causal: bool = False,
                 sparsity: int = 1, bandwidth: int = 128):
        super().__init__(dim, genome=(1, 2, sparsity, 2, 3))
        self.featurizer = Featurizer1(dim, num_heads)
        self.token_mix = LowRankTokenMixing(nonlinearity=2, causal=causal)
        self.channel_mix = GroupedChannelMixing(dim, num_heads)
        self.sparsity = sparsity
        self.bandwidth = bandwidth
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.featurizer(x)  # Q, K, V: [B, H, L, d]
        Q, K, V = feat['Q'], feat['K'], feat['V']

        # Low-rank token mixing with softmax: T_ij = softmax(Q_i · K_j / sqrt(d))
        # Then Y = T · V (per head — grouped channel mixing)
        scale = math.sqrt(Q.shape[-1])
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        if self.token_mix.causal:
            L = scores.shape[-1]
            mask = torch.triu(torch.ones(L, L, device=scores.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float('-inf'))

        if self.sparsity == 2:
            scores = apply_banded_mask(scores, self.bandwidth)

        scores = F.softmax(scores, dim=-1)
        out = torch.matmul(scores, V)  # [B, H, L, d]

        B, H, L, d = out.shape
        out = out.transpose(1, 2).reshape(B, L, H * d)  # [B, L, D]
        return self.channel_mix(out)


class GMemless(LIVOperator):
    """SwiGLU / FFN — Genome: 91142
    Featurizer 9, Diagonal token mixing, No sparsity, Swish, Dense channel mixing.
    """

    def __init__(self, dim: int, expansion: int = 4):
        super().__init__(dim, genome=(9, 1, 1, 4, 2))
        self.featurizer = Featurizer9(dim, expansion)
        # Token mixing is diagonal (implicit — each token independent)
        # Nonlinearity is swish (applied to gate)
        # Channel mixing is dense (the down projection)
        self.channel_mix = DenseChannelMixing(dim * expansion, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.featurizer(x)
        # Diagonal token mixing: each token processed independently
        # Swish nonlinearity on gate, element-wise multiply with value
        hidden = F.silu(feat['gate']) * feat['value']
        # Dense channel mixing: project back to dim
        return self.channel_mix(hidden)


class Rec1(LIVOperator):
    """Mamba-like Selective SSM — Genome: 54111
    Featurizer 5, Semi-separable token mixing, No sparsity, None, Diagonal channel mixing.
    """

    def __init__(self, dim: int, expansion: int = 16):
        super().__init__(dim, genome=(5, 4, 1, 1, 1))
        self.state_dim = dim * expansion
        # Featurizer 5: dense channel, expansion 16x
        self.W_B = nn.Linear(dim, self.state_dim)
        self.W_C = nn.Linear(dim, self.state_dim)
        self.W_A = nn.Linear(dim, self.state_dim)  # input-dependent transition
        self.W_dt = nn.Linear(dim, self.state_dim)  # discretization step
        # Log-parameterized A bias for stability
        self.A_log = nn.Parameter(torch.log(torch.rand(self.state_dim) * 0.5 + 0.5))
        self.token_mix = SemiSeparableTokenMixing()
        # Diagonal channel mixing (each state channel independent)
        self.out_proj = nn.Linear(self.state_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape

        # Input-dependent projections (featurizer 5)
        B_feat = self.W_B(x)   # [batch, L, state_dim]
        C_feat = self.W_C(x)   # [batch, L, state_dim]
        dt = F.softplus(self.W_dt(x))  # [batch, L, state_dim]

        # Discretize: A_bar = exp(dt * A), B_bar = dt * B
        A = -torch.exp(self.A_log)  # negative for stability
        A_bar = torch.exp(dt * A.unsqueeze(0).unsqueeze(0))  # [batch, L, state_dim]
        B_bar = dt * B_feat

        # Semi-separable recurrence: h_t = A_bar_t * h_{t-1} + B_bar_t
        y = self.token_mix(x, B=B_bar, C=C_feat, A=A_bar)  # [batch, L, state_dim]

        # Project back (diagonal channel mixing is implicit in per-channel recurrence)
        return self.out_proj(y)


class GConv1(LIVOperator):
    """Gated Short Convolution — Genome: 73111
    Featurizer 7, Scaled Toeplitz token mixing, No sparsity, None, Diagonal channel mixing.
    """

    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__(dim, genome=(7, 3, 1, 1, 1))
        # Featurizer 7: explicit per-channel short kernel (diagonal channel + toeplitz token)
        self.token_mix = ScaledToeplitzTokenMixing(kernel_size, dim)
        # Input-dependent gates
        self.W_B = nn.Linear(dim, dim)  # input gate
        self.W_C = nn.Linear(dim, dim)  # output gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_gate = torch.sigmoid(self.W_B(x))  # [batch, L, dim]
        C_gate = torch.sigmoid(self.W_C(x))  # [batch, L, dim]
        # Scaled Toeplitz: y = C * conv(B * x, K)
        return self.token_mix(x, B=B_gate, C=C_gate)


class Rec2(LIVOperator):
    """Compact SSM — Genome: 64111
    Featurizer 6, Semi-separable token mixing, No sparsity, None, Diagonal channel mixing.
    """

    def __init__(self, dim: int, expansion: int = 2):
        super().__init__(dim, genome=(6, 4, 1, 1, 1))
        self.state_dim = dim * expansion
        self.W_B = nn.Linear(dim, self.state_dim)
        self.W_C = nn.Linear(dim, self.state_dim)
        self.W_dt = nn.Linear(dim, self.state_dim)
        self.A_log = nn.Parameter(torch.log(torch.rand(self.state_dim) * 0.5 + 0.5))
        self.token_mix = SemiSeparableTokenMixing()
        self.out_proj = nn.Linear(self.state_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        B_feat = self.W_B(x)
        C_feat = self.W_C(x)
        dt = F.softplus(self.W_dt(x))
        A = -torch.exp(self.A_log)
        A_bar = torch.exp(dt * A.unsqueeze(0).unsqueeze(0))
        B_bar = dt * B_feat
        y = self.token_mix(x, B=B_bar, C=C_feat, A=A_bar)
        return self.out_proj(y)


class GConv2(LIVOperator):
    """Gated Long Convolution — Genome: 83111
    Featurizer 8, Scaled Toeplitz token mixing, No sparsity, None, Diagonal channel mixing.
    Kernel is implicitly parameterized (computed from input).
    """

    def __init__(self, dim: int, max_kernel_size: int = 64):
        super().__init__(dim, genome=(8, 3, 1, 1, 1))
        self.max_kernel_size = max_kernel_size
        # Implicit kernel generator
        self.kernel_net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim * max_kernel_size),
        )
        self.W_C = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        # Implicit kernel from global context (featurizer 8)
        x_global = x.mean(dim=1)  # [batch, dim]
        kernel = self.kernel_net(x_global)  # [batch, dim * kernel_size]
        kernel = kernel.view(batch, dim, self.max_kernel_size)

        C_gate = torch.sigmoid(self.W_C(x))
        x_t = x.transpose(1, 2)  # [batch, dim, seq_len]

        # FFT-based causal convolution with per-batch implicit kernel
        n_fft = 2 * seq_len
        X_f = torch.fft.rfft(x_t, n=n_fft, dim=-1)  # [batch, dim, n_fft//2+1]
        K_padded = F.pad(kernel, (0, n_fft - self.max_kernel_size))
        K_f = torch.fft.rfft(K_padded, dim=-1)  # [batch, dim, n_fft//2+1]
        y_t = torch.fft.irfft(X_f * K_f, n=n_fft, dim=-1)[..., :seq_len]

        y = y_t.transpose(1, 2)  # [batch, seq_len, dim]
        return y * C_gate


# =============================================================================
# Registry & Factory
# =============================================================================

LIV_REGISTRY = {
    1: SA1,       # 12123
    5: Rec1,      # 54111
    6: Rec2,      # 64111
    7: GConv1,    # 73111
    8: GConv2,    # 83111
    9: GMemless,  # 91142
}


def create_liv(genome: tuple, dim: int, **kwargs) -> LIVOperator:
    """Create a LIV operator from a 5-integer genome.

    Args:
        genome: (featurizer, token_mix, sparsity, nonlinearity, channel_mix)
        dim: model dimension
        **kwargs: additional arguments for the specific LIV class
    """
    import inspect
    featurizer_id = genome[0]
    if featurizer_id not in LIV_REGISTRY:
        raise ValueError(f"Featurizer class {featurizer_id} not implemented. "
                         f"Available: {list(LIV_REGISTRY.keys())}")
    cls = LIV_REGISTRY[featurizer_id]
    # Filter kwargs to only those accepted by the constructor
    sig = inspect.signature(cls.__init__)
    valid = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(dim, **valid)


def create_liv_block(genome: tuple, dim: int, **kwargs) -> LIVBlock:
    """Create a pre-norm residual LIV block from genome."""
    return LIVBlock(dim, create_liv(genome, dim, **kwargs))


# =============================================================================
# Backbone
# =============================================================================

class STARBackbone(nn.Module):
    """Stack of LIV blocks forming a complete backbone.

    Takes a list of genomes (one per layer) and builds the architecture.
    Pre-norm residual connection: y = LIV(norm(x)) + x
    """

    def __init__(self, genomes: list, dim: int, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([
            create_liv_block(g, dim, **kwargs) for g in genomes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x