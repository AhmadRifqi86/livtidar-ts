import torch
import torch.nn as nn

#NOTE: The featurizer, channel mixing and token mixing is not general enough

class DiagonalTokenMix(nn.Module):
    """Memoryless - each token independent"""
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # x: [B, L, D]
        return x * self.scale

class LowRankTokenMix(nn.Module):
    """Attention-style - all tokens interact"""
    def __init__(self, dim, rank=64):
        super().__init__()
        self.B = nn.Linear(dim, rank, bias=False)
        self.C = nn.Linear(rank, dim, bias=False)
    
    def forward(self, x):
        # x: [B, L, D]
        b = self.B(x)  # [B, L, rank]
        scores = b @ b.transpose(-2, -1)  # [B, L, L]
        scores = torch.softmax(scores, -1)
        return self.C(scores @ b)  # [B, L, D]

class SemiSeparableTokenMix(nn.Module):
    """Recurrence - causal sequential"""
    def __init__(self, dim):
        super().__init__()
        self.B = nn.Linear(dim, dim, bias=False)
        self.C = nn.Linear(dim, dim, bias=False)
        self.A = nn.Parameter(torch.randn(dim))
    
    def forward(self, x):
        # x: [B, L, D]
        b = self.B(x)
        a = torch.sigmoid(self.A)
        
        # Parallel scan
        h = torch.zeros_like(b[:, 0])
        outputs = []
        for t in range(b.shape[1]):
            h = a * h + b[:, t]
            outputs.append(h)
        
        outputs = torch.stack(outputs, 1)  # [B, L, D]
        return self.C(outputs)

class ToeplitzTokenMix(nn.Module):
    """Convolution - local receptive field"""
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size, 
                              padding=kernel_size-1, groups=dim)
    
    def forward(self, x):
        # x: [B, L, D]
        x = x.transpose(1, 2)  # [B, D, L]
        x = self.conv(x)[:, :, :x.shape[-1]]  # causal
        return x.transpose(1, 2)

class DiagonalChannelMix(nn.Module):
    """Each channel independent"""
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # x: [B, L, D]
        return x * self.weight

class DenseChannelMix(nn.Module):
    """Full channel interaction"""
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.fc2 = nn.Linear(dim * expansion, dim)
    
    def forward(self, x):
        # x: [B, L, D]
        return self.fc2(torch.silu(self.fc1(x)))

class GroupedChannelMix(nn.Module):
    """Multi-head style"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.fc = nn.Linear(dim, dim)
    
    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        x = x.view(B, L, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)  # [B, H, L, d]
        x = x.reshape(B, L, D)
        return self.fc(x)

class Featurizer1(nn.Module):
    """Dense channel + diagonal token"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(dim, hidden_dim)
    
    def forward(self, x):
        # x: [B, L, D]
        return self.proj(x)  # [B, L, H]

class Featurizer2(nn.Module):
    """Dense channel + Toeplitz token"""
    def __init__(self, dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size, 
                              padding=kernel_size-1, groups=dim)
        self.proj = nn.Linear(dim, hidden_dim)
    
    def forward(self, x):
        # x: [B, L, D]
        x_t = x.transpose(1, 2)
        x_t = self.conv(x_t)[:, :, :x.shape[1]]
        x = x_t.transpose(1, 2)
        return self.proj(x)

class Featurizer5(nn.Module):
    """Expansion factor 16 for recurrence"""
    def __init__(self, dim, expansion=16):
        super().__init__()
        self.B = nn.Linear(dim, dim * expansion)
        self.C = nn.Linear(dim * expansion, dim)
    
    def forward(self, x):
        # Returns both projections
        return self.B(x), self.C(x)

class SA1(nn.Module):
    """Softmax Attention - genome: 12123"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [B, H, L, d]
        
        # Token mixing: low-rank + softmax
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, -1)
        
        # Channel mixing: grouped
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.out(out)

class Rec1(nn.Module):
    """Recurrence - genome: 54111"""
    def __init__(self, dim, expansion=16):
        super().__init__()
        hidden = dim * expansion
        # Featurizer 5: expansion 16
        self.B = nn.Linear(dim, hidden)
        self.C = nn.Linear(hidden, dim)
        self.A = nn.Parameter(torch.randn(hidden))
    
    def forward(self, x):
        B, L, D = x.shape
        b = self.B(x)  # [B, L, H]
        a = torch.sigmoid(self.A)
        
        # Semi-separable token mixing
        h = torch.zeros(B, b.shape[-1], device=x.device)
        outputs = []
        for t in range(L):
            h = a * h + b[:, t]  # Diagonal channel mixing
            outputs.append(h)
        
        outputs = torch.stack(outputs, 1)
        return self.C(outputs)

class GMemless(nn.Module):
    """SwiGLU - genome: 91142"""
    def __init__(self, dim, expansion=4):
        super().__init__()
        hidden = dim * expansion
        self.gate = nn.Linear(dim, hidden)
        self.up = nn.Linear(dim, hidden)
        self.down = nn.Linear(hidden, dim)
    
    def forward(self, x):
        # Diagonal token + dense channel + swish
        return self.down(torch.silu(self.gate(x)) * self.up(x))

class GConv1(nn.Module):
    """Short gated conv - genome: 73111"""
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        # Featurizer 7: diagonal channel + toeplitz token
        self.gate = nn.Conv1d(dim, dim, kernel_size, 
                              padding=kernel_size-1, groups=dim)
        self.value = nn.Conv1d(dim, dim, kernel_size,
                               padding=kernel_size-1, groups=dim)
    
    def forward(self, x):
        x = x.transpose(1, 2)  # [B, D, L]
        L = x.shape[-1]
        g = self.gate(x)[:, :, :L]
        v = self.value(x)[:, :, :L]
        return (g * v).transpose(1, 2)


class STARBackbone(nn.Module):
    def __init__(self, genome, dim):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # Parse genome: groups of 5 integers
        num_livs = len(genome) // 5
        
        # Track shared featurizers/groups
        shared_feat = {}
        shared_groups = {}
        
        for i in range(num_livs):
            g = genome[i*5:(i+1)*5]
            liv_class, feat_idx, feat_strat, group_idx, group_strat = g
            
            # Create or reuse LIV
            if feat_idx in shared_feat and feat_strat > 1:
                # Share featurizer weights
                layer = shared_feat[feat_idx]
            else:
                layer = self._create_liv(liv_class, dim)
                if feat_strat > 1:
                    shared_feat[feat_idx] = layer
            
            self.layers.append(layer)
            self.norms.append(nn.LayerNorm(dim))
    
    def _create_liv(self, liv_class, dim):
        if liv_class == 1: return SA1(dim)
        elif liv_class == 5: return Rec1(dim)
        elif liv_class == 7: return GConv1(dim)
        elif liv_class == 9: return GMemless(dim)
        # ... other classes
    
    def forward(self, x):
        for norm, layer in zip(self.norms, self.layers):
            x = x + layer(norm(x))  # Pre-norm residual
        return x