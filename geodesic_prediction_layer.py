import torch
import torch.nn as nn

class GeodesicPredictionLayer(nn.Module):
    """
    Final Prediction Layer for Riemannian Classifier
    ------------------------------------------------
    Uses geodesic attention scores α_c to produce class probabilities.

    Input:
        attention_scores: Tensor of shape (B, C)
            Softmax-normalized attention weights over class prototypes.

    Output:
        class_probs: Tensor of shape (B, C)
            Predicted probabilities for each class (sum to 1 across classes).
    """
    def __init__(self, normalize: bool = True, eps: float = 1e-8):
        super().__init__()
        self.normalize = normalize
        self.eps = eps

    def forward(self, attention_scores: torch.Tensor) -> torch.Tensor:
        if attention_scores.dim() != 2:
            raise ValueError(f"Expected attention_scores to be 2D (B, C), got {attention_scores.shape}")
        
        if self.normalize:
            norm = attention_scores.sum(dim=-1, keepdim=True)
            norm = torch.clamp(norm, min=self.eps)
            class_probs = attention_scores / norm
        else:
            class_probs = attention_scores

        return class_probs

    def extra_repr(self) -> str:
        return f'normalize={self.normalize}, eps={self.eps}'
