"""
Mamba (S6) Block Implementation
Simplified implementation of Mamba for traffic feature extraction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MambaBlock(nn.Module):
    """
    Simplified Mamba (S6) block with selective mechanism
    
    This is a lightweight implementation focusing on the core
    selective state space mechanism for traffic analysis
    """
    
    def __init__(self, d_model=64, d_state=16, d_conv=4, expand=2, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            d_state: SSM state dimension
            d_conv: Local convolution kernel size
            expand: Expansion factor for inner dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.d_conv = d_conv
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        # Convolutional layer for local dependencies
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner  # Depthwise convolution
        )
        
        # Selective mechanism: make B, C, dt input-dependent
        self.x_proj = nn.Linear(self.d_inner, d_state + d_state + 1)  # B, C, dt
        
        # SSM parameters
        self.A_log = nn.Parameter(torch.randn(d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        # Initialize A_log with negative values for stability
        nn.init.uniform_(self.A_log, -math.log(self.d_state), -1.0)
    
    def forward(self, x):
        """
        Args:
            x: (B, L, d_model)
            
        Returns:
            output: (B, L, d_model)
        """
        B, L, D = x.shape
        
        # Input projection
        x_and_res = self.in_proj(x)  # (B, L, 2*d_inner)
        x_inner, res = x_and_res.chunk(2, dim=-1)  # Each (B, L, d_inner)
        
        # Local convolution
        x_conv = x_inner.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :L]  # (B, d_inner, L)
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        
        # Activation
        x_conv = F.silu(x_conv)
        
        # Selective SSM
        y = self.selective_scan(x_conv)  # (B, L, d_inner)
        
        # Residual connection with gate
        y = y * F.silu(res)
        
        # Output projection
        output = self.out_proj(y)
        output = self.dropout(output)
        
        return output
    
    def selective_scan(self, x):
        """
        Simplified selective scan mechanism
        
        Args:
            x: (B, L, d_inner)
            
        Returns:
            y: (B, L, d_inner)
        """
        B, L, D_inner = x.shape
        
        # Get selective parameters B, C, dt
        BC_dt = self.x_proj(x)  # (B, L, d_state*2 + 1)
        
        B_sel = BC_dt[..., :self.d_state]  # (B, L, d_state)
        C_sel = BC_dt[..., self.d_state:2*self.d_state]  # (B, L, d_state)
        dt = F.softplus(BC_dt[..., -1:])  # (B, L, 1)
        
        # Discretize A
        A = -torch.exp(self.A_log)  # (d_state,)
        A_bar = torch.exp(dt * A.unsqueeze(0).unsqueeze(0))  # (B, L, d_state)
        
        # Simplified SSM computation (parallel scan approximation)
        # For efficiency, we use a simplified version
        h = torch.zeros(B, self.d_state, D_inner, device=x.device)  # (B, d_state, d_inner)
        outputs = []
        
        for t in range(L):
            x_t = x[:, t, :]  # (B, d_inner)
            B_t = B_sel[:, t, :]  # (B, d_state)
            C_t = C_sel[:, t, :]  # (B, d_state)
            A_t = A_bar[:, t, :]  # (B, d_state)
            
            # State update: h_t = A_t * h_{t-1} + B_t * x_t
            h = A_t.unsqueeze(-1) * h + B_t.unsqueeze(-1) * x_t.unsqueeze(1)
            
            # Output: y_t = C_t * h_t
            y_t = torch.einsum('bd,bdi->bi', C_t, h)  # (B, d_inner)
            
            # Add skip connection
            y_t = y_t + self.D * x_t
            
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)  # (B, L, d_inner)
        
        return y


class MambaLayer(nn.Module):
    """Mamba layer with residual connection and normalization"""
    
    def __init__(self, d_model=64, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        
        self.norm = nn.LayerNorm(d_model)
        self.mamba = MambaBlock(d_model, d_state, d_conv, expand, dropout)
        
    def forward(self, x):
        """
        Args:
            x: (B, L, d_model)
            
        Returns:
            output: (B, L, d_model)
        """
        return x + self.mamba(self.norm(x))

