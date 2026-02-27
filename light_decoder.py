import torch
from torch import nn


class PredictionHead(nn.Module):
    """
    Prediction head that takes tokens (B, N, D) and outputs a specific prediction.
    """

    def __init__(self, num_tokens: int, input_dim: int, output_dim: int = 3, hidden_dims: list = [512], flatten_tokens: bool = False):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.flatten_tokens = flatten_tokens

        # Multiple linear layers for SH prediction
        layers = []
        prev_dim = num_tokens * input_dim

        # If no hidden dims, we directly go from flattened tokens to SH coefficients
        if hidden_dims:
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim)
                ])
                prev_dim = hidden_dim

            self.mlp = nn.Sequential(*layers)

        # Final layer to output
        self.out = nn.Linear(prev_dim, output_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, N, D)
        hidden = tokens  # Shape: (B, N, D)
        if self.flatten_tokens:
            # Reduce N tokens to 1
            hidden = hidden.reshape(hidden.shape[0], -1)  # Shape: (B, N * D)

        # Pass through MLP
        if self.hidden_dims:
            hidden = self.mlp(hidden)  # Shape: (B, N, H) or (B, H)
        out = self.out(hidden)  # Shape: (B, N, output_dim) or (B, output_dim)

        return out


class SHPredictionHead(PredictionHead):
    """
    Spherical Harmonics prediction head that takes tokens (B, N, D) and outputs SH parameters (B, N_sh).
    First reduce N tokens to 1, followed by multiple linear layers.
    """

    def __init__(self, num_tokens: int, input_dim: int, sh_order: int = 3, hidden_dims: list = [1024, 1024]):
        n_sh = (sh_order + 1) ** 2
        super().__init__(num_tokens=num_tokens, input_dim=input_dim, output_dim=3 * n_sh, hidden_dims=hidden_dims, flatten_tokens=True)
        self.sh_order = sh_order
        self.n_sh = n_sh

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        sh_params = super().forward(tokens)  # Shape: (B, 3 * N_sh)
        sh_params = sh_params.reshape(sh_params.shape[0], 3, self.n_sh)  # Shape: (B, 3, N_sh)
        return sh_params