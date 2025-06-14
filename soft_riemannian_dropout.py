#perfect
import torch
import torch.nn as nn

class SoftRiemannianDropout(nn.Module):
    def __init__(self, p=0.3, mode="bernoulli", alpha_range=(0.5, 1.0)):
        """
        Soft Riemannian Dropout: blends each SPD matrix with identity instead of hard replacing.

        Args:
            p (float): Drop probability.
            mode (str): 'bernoulli' for binary dropout, 'uniform' for continuous alpha sampling.
            alpha_range (tuple): If mode='uniform', sample alpha from this range.
        """
        super().__init__()
        self.p = p
        self.mode = mode
        self.alpha_range = alpha_range

    def forward(self, X):
        if not self.training or self.p == 0:
            return X

        B, D, D_ = X.shape
        assert D == D_, "Each input must be square."

        device = X.device
        I = torch.eye(D, device=device).expand(B, D, D)

        if self.mode == "bernoulli":
            mask = torch.rand(B, device=device) < self.p
            alpha = mask.to(X.dtype).view(B, 1, 1)
        elif self.mode == "uniform":
            alpha = torch.rand(B, device=device) * (self.alpha_range[1] - self.alpha_range[0]) + self.alpha_range[0]
            alpha = (alpha * self.p).view(B, 1, 1)  # scale by dropout probability
        else:
            raise ValueError("Invalid mode. Choose 'bernoulli' or 'uniform'.")

        return (1 - alpha) * X + alpha * I
