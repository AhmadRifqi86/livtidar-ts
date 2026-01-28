"""
Unified LIV Framework with Explicit Weight Matrix Generation

Core equation: Y = T(x) · x
where T(x) is ALWAYS an explicit matrix, and mixing is ALWAYS matmul.

The weight generators produce the T matrix based on:
- Token mixing type: diagonal, low_rank, semi_separable, toeplitz
- Channel mixing type: diagonal, dense, grouped

This allows architecture search to swap weight generators while keeping
the core computation (matmul) fixed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from enum import Enum
from typing import Dict, Optional, Tuple


class TokenMixType(Enum):
    DIAGONAL = 1       # T_ij = c_i * δ_ij
    LOW_RANK = 2       # T_ij = C_i · B_j  (outer product)
    TOEPLITZ = 3       # T_ij = K_{i-j}    (convolution)
    SEMI_SEPARABLE = 4 # T_ij = C_i · ∏A · B_j (recurrence)


class ChannelMixType(Enum):
    DIAGONAL = 1  # T^{αβ} = w_α * δ_{αβ}
    DENSE = 2     # T^{αβ} = W_{αβ}
    GROUPED = 3   # T^{αβ} = block_diag(W_1, ..., W_h)


# =============================================================================
# Token Mixing Weight Generators
# =============================================================================

class TokenMixWeightGenerator(nn.Module):
    """Generates the L×L token mixing matrix T based on type.

    The output is ALWAYS an explicit [batch, L, L] matrix (or [batch, heads, L, L]).
    The actual mixing is always: output = T @ values
    """

    def __init__(self, dim: int, mix_type: TokenMixType,
                 num_heads: int = 1, kernel_size: int = 3,
                 causal: bool = True):
        super().__init__()
        self.dim = dim
        self.mix_type = mix_type
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = kernel_size
        self.causal = causal

        if mix_type == TokenMixType.DIAGONAL:
            # Learnable per-position scaling (or input-dependent)
            self.scale_proj = nn.Linear(dim, dim)

        elif mix_type == TokenMixType.LOW_RANK:
            # Q, K projections for outer product T = softmax(QK^T)
            self.W_Q = nn.Linear(dim, dim)
            self.W_K = nn.Linear(dim, dim)

        elif mix_type == TokenMixType.TOEPLITZ:
            # Per-channel convolution kernels
            self.kernel = nn.Parameter(torch.randn(num_heads, self.head_dim, kernel_size) * 0.02)

        elif mix_type == TokenMixType.SEMI_SEPARABLE:
            # A, B, C projections for recurrence
            self.W_A = nn.Linear(dim, dim)
            self.W_B = nn.Linear(dim, dim)
            self.W_C = nn.Linear(dim, dim)
            # Log-parameterized decay for stability
            self.A_log = nn.Parameter(torch.log(torch.rand(dim) * 0.5 + 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate the token mixing matrix T.

        Args:
            x: [batch, seq_len, dim]
        Returns:
            T: [batch, (heads), seq_len, seq_len] - the explicit mixing matrix
        """
        B, L, D = x.shape

        if self.mix_type == TokenMixType.DIAGONAL:
            return self._diagonal_weight(x)
        elif self.mix_type == TokenMixType.LOW_RANK:
            return self._low_rank_weight(x)
        elif self.mix_type == TokenMixType.TOEPLITZ:
            return self._toeplitz_weight(x)
        elif self.mix_type == TokenMixType.SEMI_SEPARABLE:
            return self._semi_separable_weight(x)

    def _diagonal_weight(self, x: torch.Tensor) -> torch.Tensor:
        """T_ij = c_i * δ_ij → diagonal L×L matrix"""
        B, L, D = x.shape
        # Input-dependent diagonal scaling
        c = torch.sigmoid(self.scale_proj(x))  # [B, L, D]
        # Construct diagonal matrix: T[i,i] = c[i], T[i,j≠i] = 0
        # For per-head: [B, H, L, L] diagonal
        c = c.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, L, d]
        # We return per-head diagonal matrices, averaged over head_dim
        c_avg = c.mean(dim=-1)  # [B, H, L]
        T = torch.diag_embed(c_avg)  # [B, H, L, L]
        return T

    def _low_rank_weight(self, x: torch.Tensor) -> torch.Tensor:
        """T_ij = softmax(Q_i · K_j / √d) → dense L×L from outer product"""
        B, L, D = x.shape
        Q = self.W_Q(x).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.W_K(x).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # [B, H, L, L]
        scale = math.sqrt(self.head_dim)
        T = torch.matmul(Q, K.transpose(-2, -1)) / scale

        if self.causal:
            mask = torch.triu(torch.ones(L, L, device=T.device, dtype=torch.bool), diagonal=1)
            T = T.masked_fill(mask, float('-inf'))

        T = F.softmax(T, dim=-1)
        return T

    def _toeplitz_weight(self, x: torch.Tensor) -> torch.Tensor:
        """T_ij = K_{i-j} for |i-j| < kernel_size → banded Toeplitz L×L"""
        B, L, D = x.shape
        H = self.num_heads

        # Construct Toeplitz matrix from kernel
        # kernel: [H, head_dim, kernel_size]
        # We average over head_dim to get [H, kernel_size]
        k = self.kernel.mean(dim=1)  # [H, kernel_size]

        # Build L×L Toeplitz matrix per head
        T = torch.zeros(H, L, L, device=x.device, dtype=x.dtype)
        for i in range(self.kernel_size):
            if self.causal:
                # Only fill lower diagonals (causal)
                diag_idx = i
                diag_len = L - i
                T[:, torch.arange(i, L), torch.arange(0, diag_len)] = k[:, i:i+1].expand(-1, diag_len)
            else:
                # Fill both diagonals (non-causal)
                if i == 0:
                    T[:, torch.arange(L), torch.arange(L)] = k[:, 0:1].expand(-1, L)
                else:
                    T[:, torch.arange(i, L), torch.arange(0, L-i)] = k[:, i:i+1].expand(-1, L-i)
                    T[:, torch.arange(0, L-i), torch.arange(i, L)] = k[:, i:i+1].expand(-1, L-i)

        return T.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, L, L]

    def _semi_separable_weight(self, x: torch.Tensor) -> torch.Tensor:
        """T_ij = C_i · (∏_{k=j+1}^{i-1} A_k) · B_j → lower triangular L×L"""
        B, L, D = x.shape

        # Input-dependent A, B, C
        A = torch.sigmoid(self.W_A(x))  # [B, L, D] in (0, 1) for stability
        B_feat = self.W_B(x)            # [B, L, D]
        C = self.W_C(x)                 # [B, L, D]

        # Average over dim to get scalar transitions per position
        # For simplicity, we compute per-head with averaged features
        A = A.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, L, d]
        B_feat = B_feat.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        C = C.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Construct lower triangular T explicitly
        # T[i,j] = C[i] · A[i-1] · A[i-2] · ... · A[j+1] · B[j]  for i >= j
        # This is O(L²) but explicit
        T = torch.zeros(B, self.num_heads, L, L, device=x.device, dtype=x.dtype)

        for i in range(L):
            for j in range(i + 1):
                # Compute C_i · (prod of A's) · B_j
                c_i = C[:, :, i, :]      # [B, H, d]
                b_j = B_feat[:, :, j, :] # [B, H, d]

                # Product of A's from j+1 to i-1
                if i == j:
                    # T[i,i] = C_i · B_i (no A's in between)
                    prod = torch.ones_like(c_i)
                else:
                    prod = torch.ones_like(c_i)
                    for k in range(j + 1, i):
                        prod = prod * A[:, :, k, :]

                # T[i,j] = sum over d of (c_i * prod * b_j)
                T[:, :, i, j] = (c_i * prod * b_j).sum(dim=-1)

        return T  # [B, H, L, L]


# =============================================================================
# Channel Mixing Weight Generators
# =============================================================================

class ChannelMixWeightGenerator(nn.Module):
    """Generates the d×d channel mixing matrix based on type.

    The output is ALWAYS an explicit [d_out, d_in] matrix.
    The actual mixing is always: output = input @ W.T
    """

    def __init__(self, dim_in: int, dim_out: int, mix_type: ChannelMixType,
                 num_heads: int = 1):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.mix_type = mix_type
        self.num_heads = num_heads

        if mix_type == ChannelMixType.DIAGONAL:
            # Diagonal: only d parameters
            self.weight = nn.Parameter(torch.ones(min(dim_in, dim_out)))

        elif mix_type == ChannelMixType.DENSE:
            # Dense: full d_out × d_in matrix
            self.weight = nn.Parameter(torch.randn(dim_out, dim_in) * 0.02)

        elif mix_type == ChannelMixType.GROUPED:
            # Block diagonal: h blocks of (d_out/h) × (d_in/h)
            assert dim_in % num_heads == 0 and dim_out % num_heads == 0
            head_dim_in = dim_in // num_heads
            head_dim_out = dim_out // num_heads
            # Store as [H, d_out/H, d_in/H]
            self.weight = nn.Parameter(torch.randn(num_heads, head_dim_out, head_dim_in) * 0.02)

    def forward(self) -> torch.Tensor:
        """Generate the channel mixing matrix W.

        Returns:
            W: [d_out, d_in] - the explicit mixing matrix
        """
        if self.mix_type == ChannelMixType.DIAGONAL:
            return self._diagonal_weight()
        elif self.mix_type == ChannelMixType.DENSE:
            return self._dense_weight()
        elif self.mix_type == ChannelMixType.GROUPED:
            return self._grouped_weight()

    def _diagonal_weight(self) -> torch.Tensor:
        """W_{αβ} = w_α * δ_{αβ} → diagonal matrix"""
        d = len(self.weight)
        W = torch.diag(self.weight)
        # Pad if dim_out != dim_in
        if self.dim_out > d:
            W = F.pad(W, (0, 0, 0, self.dim_out - d))
        if self.dim_in > d:
            W = F.pad(W, (0, self.dim_in - d, 0, 0))
        return W[:self.dim_out, :self.dim_in]

    def _dense_weight(self) -> torch.Tensor:
        """W_{αβ} = learned dense matrix"""
        return self.weight

    def _grouped_weight(self) -> torch.Tensor:
        """W = block_diag(W_1, W_2, ..., W_h) → block diagonal"""
        # weight: [H, d_out/H, d_in/H]
        H = self.num_heads
        W = torch.block_diag(*[self.weight[h] for h in range(H)])
        return W


# =============================================================================
# Unified Featurizer
# =============================================================================

class UnifiedFeaturizer(nn.Module):
    """Featurizer that produces features for both token and channel mixing.

    Given input x, produces:
    - Features for token mixing weight generation (Q, K for low-rank; A, B, C for recurrence; etc.)
    - Value tensor V that will be mixed by the token mixing matrix
    """

    def __init__(self, dim: int, num_heads: int = 1, expansion: int = 1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.expansion = expansion

        # Value projection (what gets mixed)
        self.W_V = nn.Linear(dim, dim * expansion)

        # Output projection (after mixing)
        if expansion != 1:
            self.W_O = nn.Linear(dim * expansion, dim)
        else:
            self.W_O = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from input.

        Returns:
            dict with 'V' (values to be mixed) and 'x' (original for weight gen)
        """
        B, L, D = x.shape
        V = self.W_V(x)  # [B, L, D*expansion]

        if self.expansion == 1:
            V = V.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        return {'V': V, 'x': x}


# =============================================================================
# Unified LIV Operator
# =============================================================================

class UnifiedLIV(nn.Module):
    """Unified LIV operator where mixing is ALWAYS explicit matrix multiplication.

    Y = C @ (T @ V)

    where:
    - T = token_mix_weight_gen(x) produces [B, H, L, L]
    - V = featurizer(x)['V'] is [B, H, L, d]
    - C = channel_mix_weight_gen() produces [d_out, d_in]

    The weight generators can be swapped during architecture search
    while the core computation (matmul) stays fixed.
    """

    def __init__(self, dim: int,
                 token_mix_type: TokenMixType,
                 channel_mix_type: ChannelMixType,
                 num_heads: int = 8,
                 kernel_size: int = 3,
                 expansion: int = 1,
                 causal: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Weight generators (these define the structure of T)
        self.token_mix_gen = TokenMixWeightGenerator(
            dim, token_mix_type, num_heads, kernel_size, causal
        )
        self.channel_mix_gen = ChannelMixWeightGenerator(
            dim * expansion, dim, channel_mix_type, num_heads
        )

        # Featurizer (produces values to be mixed)
        self.featurizer = UnifiedFeaturizer(dim, num_heads, expansion)

        self.expansion = expansion
        self.token_mix_type = token_mix_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Y = C @ (T @ V) with explicit matrices.

        Args:
            x: [batch, seq_len, dim]
        Returns:
            y: [batch, seq_len, dim]
        """
        B, L, D = x.shape

        # 1. Generate token mixing matrix T: [B, H, L, L]
        T = self.token_mix_gen(x)

        # 2. Get values from featurizer
        feat = self.featurizer(x)
        V = feat['V']  # [B, H, L, head_dim] or [B, L, D*exp]

        # 3. Token mixing: T @ V (explicit matmul!)
        if self.expansion == 1:
            # V: [B, H, L, head_dim], T: [B, H, L, L]
            mixed = torch.matmul(T, V)  # [B, H, L, head_dim]
            mixed = mixed.permute(0, 2, 1, 3).reshape(B, L, D)  # [B, L, D]
        else:
            # For expansion > 1, T is applied per-position (diagonal token mix case)
            # V: [B, L, D*exp]
            if T.dim() == 4:
                # Reshape for batched matmul
                T_avg = T.mean(dim=1)  # [B, L, L] average over heads
                mixed = torch.matmul(T_avg, V)  # [B, L, D*exp]
            else:
                mixed = torch.matmul(T, V)

        # 4. Channel mixing: mixed @ C.T (explicit matmul!)
        C = self.channel_mix_gen()  # [D, D*exp] or [D, D]
        y = F.linear(mixed, C)  # [B, L, D]

        return y


# =============================================================================
# Factory Functions
# =============================================================================

def create_unified_liv(
    dim: int,
    token_mix: int,    # 1=diagonal, 2=low_rank, 3=toeplitz, 4=semi_separable
    channel_mix: int,  # 1=diagonal, 2=dense, 3=grouped
    **kwargs
) -> UnifiedLIV:
    """Create a UnifiedLIV from integer type codes (like genome)."""
    token_type = TokenMixType(token_mix)
    channel_type = ChannelMixType(channel_mix)
    return UnifiedLIV(dim, token_type, channel_type, **kwargs)


# =============================================================================
# Pre-defined Configurations (matching paper's LIV classes)
# =============================================================================

def SA1(dim: int, num_heads: int = 8, causal: bool = True) -> UnifiedLIV:
    """Standard Attention: low-rank token + grouped channel"""
    return UnifiedLIV(dim, TokenMixType.LOW_RANK, ChannelMixType.GROUPED,
                      num_heads=num_heads, causal=causal)

def Rec1(dim: int, num_heads: int = 8, causal: bool = True) -> UnifiedLIV:
    """SSM-style Recurrence: semi-separable token + diagonal channel"""
    return UnifiedLIV(dim, TokenMixType.SEMI_SEPARABLE, ChannelMixType.DIAGONAL,
                      num_heads=num_heads, causal=causal)

def GConv1(dim: int, num_heads: int = 8, kernel_size: int = 3) -> UnifiedLIV:
    """Gated Convolution: toeplitz token + diagonal channel"""
    return UnifiedLIV(dim, TokenMixType.TOEPLITZ, ChannelMixType.DIAGONAL,
                      num_heads=num_heads, kernel_size=kernel_size)

def GMemless(dim: int, expansion: int = 4) -> UnifiedLIV:
    """SwiGLU/FFN: diagonal token + dense channel"""
    return UnifiedLIV(dim, TokenMixType.DIAGONAL, ChannelMixType.DENSE,
                      num_heads=1, expansion=expansion)


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
