"""
Unified LIV Framework with Consistent Feature Interface

Core equation: Y = T(x) · x
Token mixing: T_ij = C_i · M_ij · B_j  (unified for ALL types)

All token mixing types use exactly 3 feature groups:
- B: input projection  [batch, seq_len, dim]
- C: output projection [batch, seq_len, dim]
- S: structure param   [batch, seq_len, dim] or [dim, kernel_size]

The M_ij matrix is determined by type:
- Diagonal:      M_ij = δ_ij           (S unused, or extra scaling)
- Low-rank:      M_ij = 1              (S unused, outer product C·B^T)
- Toeplitz:      M_ij = S_{i-j}        (S = convolution kernel)
- Semi-separable: M_ij = ∏A from S    (S = transition coefficients A)

NOTE: 
-   Maybe pairing featurizer and token mixer in the same class for clear coupling, because in original paper
    the number of featurizer output is based on token mixing type --> 3 for low rank, 4 for recur, 2 for diagonal
-   also non-linearity and sparsity genome is not yet implemented
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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


# =============================================================================
# Unified Featurizer: Always produces (B, C, S)
# =============================================================================

class UnifiedFeaturizer(nn.Module):
    """Produces exactly 3 feature groups for token mixing: B, C, S.

    B: input projection  - how each input position contributes
    C: output projection - how each output position is formed
    S: structure param   - type-specific (A for recurrence, K for conv, etc.)

    This ensures all token mixing types have the same interface.
    """

    def __init__(self, dim: int, num_heads: int = 1,
                 token_mix_type: TokenMixType = TokenMixType.LOW_RANK,
                 kernel_size: int = 3):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.token_mix_type = token_mix_type
        self.kernel_size = kernel_size

        # B and C projections (used by ALL types)
        self.W_B = nn.Linear(dim, dim)  # Input gate/projection
        self.W_C = nn.Linear(dim, dim)  # Output gate/projection

        # S projection (structure-specific)
        if token_mix_type == TokenMixType.DIAGONAL:
            # S acts as additional scaling (or can be ignored)
            self.W_S = nn.Linear(dim, dim)

        elif token_mix_type == TokenMixType.LOW_RANK:
            # S not needed for outer product, but we keep it for uniformity
            # Can be used as learned bias or scaling
            self.W_S = nn.Linear(dim, dim)

        elif token_mix_type == TokenMixType.TOEPLITZ:
            # S generates the convolution kernel
            # Input-dependent kernel: x -> [dim, kernel_size]
            self.W_S = nn.Linear(dim, dim * kernel_size)

        elif token_mix_type == TokenMixType.SEMI_SEPARABLE:
            # S generates state transition A
            self.W_S = nn.Linear(dim, dim)
            # Learnable decay bias for stability
            self.A_log = nn.Parameter(torch.zeros(dim))

        # Value projection (what gets mixed)
        self.W_V = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor):
        """Compute B, C, S, V from input.

        Args:
            x: [batch, seq_len, dim]

        Returns:
            dict with B, C, S, V tensors
        """
        batch, seq_len, dim = x.shape

        # B and C: always [batch, seq_len, dim]
        B = self.W_B(x)
        C = self.W_C(x)
        V = self.W_V(x)

        # S: structure-specific
        if self.token_mix_type == TokenMixType.DIAGONAL:
            S = torch.sigmoid(self.W_S(x))  # [batch, L, dim] scaling

        elif self.token_mix_type == TokenMixType.LOW_RANK:
            S = self.W_S(x)  # [batch, L, dim] (optional, can be identity-like)

        elif self.token_mix_type == TokenMixType.TOEPLITZ:
            # Generate per-position kernel, then average for shared kernel
            S_raw = self.W_S(x)  # [batch, L, dim * kernel_size]
            S = S_raw.mean(dim=1)  # [batch, dim * kernel_size]
            S = S.view(batch, self.dim, self.kernel_size)  # [batch, dim, K]

        elif self.token_mix_type == TokenMixType.SEMI_SEPARABLE:
            # A = sigmoid(proj) for stability in (0, 1)
            S = torch.sigmoid(self.W_S(x) + self.A_log)  # [batch, L, dim]

        return {'B': B, 'C': C, 'S': S, 'V': V}


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

    def __init__(self, dim: int, mix_type: TokenMixType,
                 num_heads: int = 1, kernel_size: int = 3,
                 causal: bool = True, use_softmax: bool = False):
        super().__init__()
        self.dim = dim
        self.mix_type = mix_type
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = kernel_size
        self.causal = causal
        self.use_softmax = use_softmax

    def forward(self, B: torch.Tensor, C: torch.Tensor,
                S: torch.Tensor) -> torch.Tensor:
        """Generate token mixing matrix T from features.

        Args:
            B: [batch, seq_len, dim] - input projection
            C: [batch, seq_len, dim] - output projection
            S: structure param (shape varies by type)

        Returns:
            T: [batch, num_heads, seq_len, seq_len]
        """
        batch, seq_len, dim = B.shape
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

        # Optional softmax (for attention)
        if self.use_softmax:
            T = F.softmax(T, dim=-1)

        return T

    def _diagonal(self, B_h, C_h, S):
        """T_ij = (C_i ⊙ B_i) · δ_ij - diagonal matrix"""
        # S: [batch, L, dim] additional scaling
        batch, H, L, d = B_h.shape

        # Diagonal: T[i,i] = C_i · B_i (element-wise, summed over head_dim)
        # [batch, H, L]
        diag_vals = (C_h * B_h).sum(dim=-1)

        # Build diagonal matrix
        T = torch.diag_embed(diag_vals)  # [batch, H, L, L]
        return T

    def _low_rank(self, B_h, C_h, S):
        """T_ij = C_i · B_j^T - outer product (attention-like)"""
        # [batch, H, L, L] = [batch, H, L, d] @ [batch, H, d, L]
        scale = math.sqrt(self.head_dim)
        T = torch.matmul(C_h, B_h.transpose(-2, -1)) / scale
        return T

    def _toeplitz(self, B_h, C_h, S):
        """T_ij = C_i · K_{i-j} · B_j - convolution structure"""
        # S: [batch, dim, kernel_size] - convolution kernel
        batch, H, L, d = B_h.shape
        K = S  # [batch, dim, kernel_size]

        # Build Toeplitz matrix from kernel
        # Average kernel over dim to get [batch, kernel_size]
        K_avg = K.mean(dim=1)  # [batch, kernel_size]

        T = torch.zeros(batch, H, L, L, device=B_h.device, dtype=B_h.dtype)

        for k in range(min(self.kernel_size, L)):
            # Diagonal k: positions (i, i-k) for i >= k
            if L - k > 0:
                # Scale by C_i and B_{i-k}
                # C_h[:, :, k:, :] and B_h[:, :, :L-k, :]
                C_part = C_h[:, :, k:, :]       # [batch, H, L-k, d]
                B_part = B_h[:, :, :L-k, :]     # [batch, H, L-k, d]
                # Element-wise product summed over d, scaled by kernel
                cb = (C_part * B_part).sum(dim=-1)  # [batch, H, L-k]
                k_val = K_avg[:, k:k+1]  # [batch, 1]

                # Fill diagonal
                idx = torch.arange(L - k, device=T.device)
                T[:, :, idx + k, idx] = cb * k_val.unsqueeze(1)

        return T

    def _semi_separable(self, B_h, C_h, S):
        """T_ij = C_i · (∏_{k=j+1}^{i-1} A_k) · B_j - recurrence structure"""
        # S: [batch, L, dim] - state transition A
        batch, H, L, d = B_h.shape

        # Reshape A to heads
        A_h = S.view(batch, L, H, d).permute(0, 2, 1, 3)  # [batch, H, L, d]

        # Build lower triangular T explicitly
        T = torch.zeros(batch, H, L, L, device=B_h.device, dtype=B_h.dtype)

        # Precompute cumulative products of A for efficiency
        # cum_A[i] = A_1 * A_2 * ... * A_i
        cum_A = torch.ones(batch, H, L, d, device=A_h.device, dtype=A_h.dtype)
        for i in range(1, L):
            cum_A[:, :, i] = cum_A[:, :, i-1] * A_h[:, :, i]

        for i in range(L):
            for j in range(i + 1):
                c_i = C_h[:, :, i, :]  # [batch, H, d]
                b_j = B_h[:, :, j, :]  # [batch, H, d]

                if i == j:
                    # T[i,i] = C_i · B_i (no A's)
                    prod = torch.ones_like(c_i)
                else:
                    # T[i,j] = C_i · (A_{j+1} · ... · A_{i-1}) · B_j
                    # = C_i · (cum_A[i-1] / cum_A[j]) · B_j
                    if j == 0:
                        prod = cum_A[:, :, i-1] if i > 0 else torch.ones_like(c_i)
                    else:
                        prod = cum_A[:, :, i-1] / (cum_A[:, :, j-1] + 1e-8)

                T[:, :, i, j] = (c_i * prod * b_j).sum(dim=-1)

        return T


# =============================================================================
# Channel Mixing Weight Generator
# =============================================================================

class ChannelMixWeightGenerator(nn.Module):
    """Generates d×d channel mixing matrix."""

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

    def forward(self) -> torch.Tensor:
        """Generate channel mixing matrix W: [dim_out, dim_in]"""
        if self.mix_type == ChannelMixType.DIAGONAL:
            d = len(self.weight)
            W = torch.diag(self.weight)
            if self.dim_out != d or self.dim_in != d:
                W = F.pad(W, (0, max(0, self.dim_in - d), 0, max(0, self.dim_out - d)))
            return W[:self.dim_out, :self.dim_in]

        elif self.mix_type == ChannelMixType.DENSE:
            return self.weight

        elif self.mix_type == ChannelMixType.GROUPED:
            return torch.block_diag(*[self.weight[h] for h in range(self.num_heads)])


# =============================================================================
# Unified LIV Operator
# =============================================================================

class UnifiedLIV(nn.Module):
    """Unified LIV: Y = C_mix @ (T @ V)

    Where T = token_mixing_weight(B, C, S) is ALWAYS an explicit L×L matrix.

    All token mixing types use the same featurizer interface:
    - B: input projection
    - C: output projection
    - S: structure parameter

    This allows swapping token/channel mixing types while keeping
    the core computation (explicit matmul) fixed.
    """

    def __init__(self, dim: int,
                 token_mix_type: TokenMixType = TokenMixType.LOW_RANK,
                 channel_mix_type: ChannelMixType = ChannelMixType.GROUPED,
                 num_heads: int = 8,
                 kernel_size: int = 3,
                 causal: bool = True,
                 use_softmax: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Featurizer: produces B, C, S, V (same interface for all types)
        self.featurizer = UnifiedFeaturizer(
            dim, num_heads, token_mix_type, kernel_size
        )

        # Token mixing weight generator
        self.token_mix_gen = TokenMixWeightGenerator(
            dim, token_mix_type, num_heads, kernel_size, causal,
            use_softmax=(token_mix_type == TokenMixType.LOW_RANK and use_softmax)
        )

        # Channel mixing weight generator
        self.channel_mix_gen = ChannelMixWeightGenerator(
            dim, dim, channel_mix_type, num_heads
        )

        self.token_mix_type = token_mix_type
        self.channel_mix_type = channel_mix_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Y = C_mix @ (T @ V).

        1. Featurizer: x -> (B, C, S, V)
        2. Token mixing: T = f(B, C, S), then T @ V
        3. Channel mixing: result @ W^T
        """
        batch, seq_len, dim = x.shape

        # 1. Extract features (unified interface)
        feat = self.featurizer(x)
        B, C, S, V = feat['B'], feat['C'], feat['S'], feat['V']

        # 2. Generate token mixing matrix T: [batch, heads, L, L]
        T = self.token_mix_gen(B, C, S)

        # 3. Reshape V for multi-head and apply token mixing
        V_h = V.view(batch, seq_len, self.num_heads, self.head_dim)
        V_h = V_h.permute(0, 2, 1, 3)  # [batch, heads, L, head_dim]

        # Token mixing: T @ V (explicit matmul!)
        mixed = torch.matmul(T, V_h)  # [batch, heads, L, head_dim]

        # Reshape back
        mixed = mixed.permute(0, 2, 1, 3).reshape(batch, seq_len, dim)

        # 4. Channel mixing (explicit matmul!)
        W = self.channel_mix_gen()  # [dim, dim]
        output = F.linear(mixed, W)

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


# Preset configurations matching paper's LIV classes
def SA1(dim: int, num_heads: int = 8, **kw) -> UnifiedLIV:
    """Attention: low-rank token + grouped channel"""
    return UnifiedLIV(dim, TokenMixType.LOW_RANK, ChannelMixType.GROUPED,
                      num_heads=num_heads, use_softmax=True, **kw)

def Rec1(dim: int, num_heads: int = 8, **kw) -> UnifiedLIV:
    """SSM: semi-separable token + diagonal channel"""
    return UnifiedLIV(dim, TokenMixType.SEMI_SEPARABLE, ChannelMixType.DIAGONAL,
                      num_heads=num_heads, use_softmax=False, **kw)

def GConv1(dim: int, num_heads: int = 8, kernel_size: int = 3, **kw) -> UnifiedLIV:
    """Conv: toeplitz token + diagonal channel"""
    return UnifiedLIV(dim, TokenMixType.TOEPLITZ, ChannelMixType.DIAGONAL,
                      num_heads=num_heads, kernel_size=kernel_size, use_softmax=False, **kw)

def GMemless(dim: int, num_heads: int = 1, **kw) -> UnifiedLIV:
    """FFN: diagonal token + dense channel"""
    return UnifiedLIV(dim, TokenMixType.DIAGONAL, ChannelMixType.DENSE,
                      num_heads=num_heads, use_softmax=False, **kw)


# =============================================================================
# LIV Block with Residual
# =============================================================================

class UnifiedLIVBlock(nn.Module):
    """Pre-norm residual: y = LIV(norm(x)) + x"""

    def __init__(self, dim: int, liv: UnifiedLIV):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.liv = liv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.liv(self.norm(x))


class STARBackbone(nn.Module):
    """Stack of UnifiedLIV blocks."""

    def __init__(self, configs: list, dim: int, **kwargs):
        """
        Args:
            configs: list of (token_mix_type, channel_mix_type) tuples
            dim: model dimension
        """
        super().__init__()
        self.blocks = nn.ModuleList([
            UnifiedLIVBlock(dim, create_liv(dim, t, c, **kwargs))
            for t, c in configs
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x