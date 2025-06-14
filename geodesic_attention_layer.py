import torch
import torch.nn as nn
import torch.nn.functional as F

class GeodesicAttentionLayer(nn.Module):
    """
    Robust Geodesic Attention Layer (Safe for Training)
    ---------------------------------------------------
    Applies temperature-scaled softmax over squared Riemannian distances,
    with added guards against NaN, Inf, or unstable values.
    """
    def __init__(self, temperature: float = 1.0, learnable: bool = False, normalize: bool = True, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.normalize = normalize

        temp = torch.tensor(temperature, dtype=torch.float32)
        if learnable:
            self.temperature = nn.Parameter(temp)
        else:
            self.register_buffer("temperature", temp)

    def forward(self, distances_squared: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distances_squared (Tensor): Shape (B, C), squared distances to class prototypes.
        Returns:
            attention (Tensor): Shape (B, C), attention weights over classes.
        """
        if distances_squared.ndim != 2:
            raise ValueError(f"Expected shape (B, C), got {distances_squared.shape}")

        # Step 1: Clamp unreasonable values
        distances_squared = torch.clamp(distances_squared, min=0.0, max=1e4)

        # Step 2: Normalize per batch if requested
        if self.normalize:
            mean = distances_squared.mean(dim=1, keepdim=True)
            std = distances_squared.std(dim=1, keepdim=True)
            std = torch.clamp(std, min=self.eps)
            distances_squared = (distances_squared - mean) / std

        # Step 3: Clamp temperature
        temperature = torch.clamp(self.temperature, min=self.eps)

        # Step 4: Scale and apply softmax with numerically stable exp
        scaled = -distances_squared / temperature
        scaled = torch.nan_to_num(scaled, nan=0.0, posinf=-1e4, neginf=1e4)

        max_val, _ = scaled.max(dim=-1, keepdim=True)
        exp_logits = torch.exp(scaled - max_val)
        sum_exp = exp_logits.sum(dim=-1, keepdim=True) + self.eps
        attention = exp_logits / sum_exp

        # Optional final check
        if not torch.allclose(attention.sum(dim=-1), torch.ones_like(attention.sum(dim=-1)), atol=1e-2):
            attention = F.softmax(scaled, dim=-1)

        return attention

    def extra_repr(self) -> str:
        return f"temperature={self.temperature.item():.4f}, learnable={self.temperature.requires_grad}, normalize={self.normalize}"
