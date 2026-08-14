"""
Models for Darcy Flow surrogate modeling.
1. Baseline MLP (flattened input -> flattened output)
2. Fourier Neural Operator (FNO) - spectral convolution
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBaseline(nn.Module):
    """
    Baseline MLP: flatten input (coeff + coords) -> hidden layers -> flatten output
    """
    def __init__(self, input_channels=3, output_channels=1, height=64, width=64, 
                 hidden_dims=[1024, 1024, 1024], activation=nn.GELU):
        super().__init__()
        self.height = height
        self.width = width
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        input_dim = input_channels * height * width
        output_dim = output_channels * height * width
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        # x: (B, H, W, C) -> (B, H*W*C)
        B = x.shape[0]
        x = x.view(B, -1)
        x = self.net(x)
        x = x.view(B, self.height, self.width, self.output_channels)
        return x


class SpectralConv2d(nn.Module):
    """
    2D Spectral Convolution: FFT -> multiply -> IFFT
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
    
    def compl_mul2d(self, input, weights):
        # (batch, in_channels, x, y), (in_channels, out_channels, x, y) -> (batch, out_channels, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    def forward(self, x):
        B, H, W, C = x.shape
        # Convert to (B, C, H, W) for FFT
        x = x.permute(0, 3, 1, 2)
        
        # FFT
        x_ft = torch.fft.rfft2(x)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        
        # Low frequencies
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )
        
        # IFFT
        x = torch.fft.irfft2(out_ft, s=(H, W))
        
        # Convert back to (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        return x


class FNO2d(nn.Module):
    """
    Fourier Neural Operator for 2D Darcy Flow.
    Architecture: Lift -> Spectral Conv blocks -> Project -> Output
    """
    def __init__(self, input_channels=3, output_channels=1, width=64, modes=12, 
                 n_layers=4, padding=0):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.width = width
        self.modes = modes
        self.n_layers = n_layers
        self.padding = padding
        
        # Lifting: input_channels -> width
        self.fc0 = nn.Linear(input_channels, width)
        
        # Spectral convolution blocks
        self.spectral_convs = nn.ModuleList([
            SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)
        ])
        self.ws = nn.ModuleList([
            nn.Conv2d(width, width, 1) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm([width]) for _ in range(n_layers)
        ])
        
        # Projection: width -> output_channels
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, output_channels)
        
    def forward(self, x):
        # x: (B, H, W, C_in)
        B, H, W, _ = x.shape
        
        # Lift
        x = self.fc0(x)  # (B, H, W, width)
        
        # Spectral conv blocks
        for i in range(self.n_layers):
            x1 = self.spectral_convs[i](x)
            x2 = self.ws[i](x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            x = x1 + x2
            x = self.norms[i](x)
            x = F.gelu(x)
        
        # Project
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)  # (B, H, W, C_out)
        
        return x


def get_model(model_type: str, **kwargs):
    """Factory function to get model by type."""
    if model_type == "mlp":
        return MLPBaseline(**kwargs)
    elif model_type == "fno":
        return FNO2d(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, H, W = 4, 64, 64
    
    # MLP
    mlp = MLPBaseline(input_channels=3, output_channels=1, height=H, width=W).to(device)
    x = torch.randn(B, H, W, 3).to(device)
    y = mlp(x)
    print(f"MLP: {count_parameters(mlp):,} params, input {x.shape} -> output {y.shape}")
    
    # FNO
    fno = FNO2d(input_channels=3, output_channels=1, width=64, modes=12, n_layers=4).to(device)
    y = fno(x)
    print(f"FNO: {count_parameters(fno):,} params, input {x.shape} -> output {y.shape}")