import torch
import torch.nn as nn
import torch.nn.functional as F

class RPCE_Loss(nn.Module):
    """
    Riemannian Prototype Cross-Entropy Loss (RPCE)
    ------------------------------------------------
    Computes the cross-entropy loss between predicted class probabilities and
    integer class labels (not one-hot). Ensures numerical stability.
    
    Args:
        epsilon (float): Small constant to prevent log(0)
    """
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, class_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            class_probs: Tensor of shape (B, C) — predicted class probabilities
            labels: Tensor of shape (B,) — ground truth class indices

        Returns:
            Tensor: scalar loss value
        """
        B, C = class_probs.shape
        class_probs = torch.clamp(class_probs, min=self.epsilon, max=1.0)

        # One-hot encode the labels
        one_hot = F.one_hot(labels, num_classes=C).float()

        # Compute numerically-stable cross-entropy
        loss = -torch.sum(one_hot * torch.log(class_probs), dim=1).mean()
        return loss
