#This block is working correctly
#EigenActivation
import torch
import torch.nn as nn
import torch.nn.functional as F

class EigenActivation(nn.Module):
    """
    Optimized Flexible Activation over SPD matrices via eigen decomposition.
    Supports using precomputed (eigvals, eigvecs) to save computation.

    Available activations:
        - 'ReEig'
        - 'PowerEig'
        - 'SqrtEig'
        - 'ExpEig'
        - 'LogEig'
        - 'LearnableScale'
        - 'ThresholdSoftplus'
        - 'SpectralNormScale'
        - 'Affine'

    Args:
        activation (str): Type of activation.
        power (float, optional): Power for PowerEig. Default 0.5.
        learnable (bool, optional): Whether scale/threshold parameters are learnable. Default True.
    """
    def __init__(self, activation='ReEig', power=0.5, learnable=True):
        super(EigenActivation, self).__init__()
        self.activation = activation
        self.power = power
        self.learnable = learnable

        if activation == 'LearnableScale':
            self.scale = nn.Parameter(torch.ones(1)) if learnable else 1.0

        if activation == 'ThresholdSoftplus':
            self.threshold = nn.Parameter(torch.ones(1)) if learnable else 1.0

        if activation == 'SpectralNormScale':
            self.scale = nn.Parameter(torch.ones(1)) if learnable else 1.0

        if activation == 'Affine':
            self.affine_a = nn.Parameter(torch.ones(1)) if learnable else 1.0
            self.affine_b = nn.Parameter(torch.zeros(1)) if learnable else 0.0

    def forward(self, X=None, eigvals=None, eigvecs=None):
        """
        Forward pass.

        Either provide:
            - X (Tensor): Input SPD matrix, shape (batch, dim, dim)
        OR
            - eigvals (Tensor): Precomputed eigenvalues, shape (batch, dim)
              eigvecs (Tensor): Precomputed eigenvectors, shape (batch, dim, dim)

        Returns:
            Tensor: Activated SPD matrix
        """
        if eigvals is None or eigvecs is None:
            eigvals, eigvecs = torch.linalg.eigh(X)

        eigvals = self.apply_activation(eigvals)

        return eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)

    def apply_activation(self, eigvals):
        """Apply activation to eigenvalues only."""
        if self.activation == 'ReEig':
            eigvals = F.relu(eigvals)

        elif self.activation == 'PowerEig':
            eigvals = torch.sign(eigvals) * torch.abs(eigvals) ** self.power

        elif self.activation == 'SqrtEig':
            eigvals = torch.sign(eigvals) * torch.sqrt(torch.clamp(torch.abs(eigvals), min=1e-6))

        elif self.activation == 'ExpEig':
            eigvals = torch.exp(eigvals)

        elif self.activation == 'LogEig':
            eigvals = torch.log(torch.clamp(eigvals, min=1e-6))

        elif self.activation == 'LearnableScale':
            eigvals = eigvals * self.scale

        elif self.activation == 'ThresholdSoftplus':
            eigvals = F.softplus(eigvals - self.threshold)

        elif self.activation == 'SpectralNormScale':
            norm = eigvals.norm(p=2, dim=-1, keepdim=True) + 1e-6
            eigvals = eigvals / norm * self.scale

        elif self.activation == 'Affine':
            eigvals = eigvals * self.affine_a + self.affine_b

        else:
            raise ValueError(f"Unknown activation type: {self.activation}")

        # ✅ Post-fix: ensure all eigenvalues are strictly positive (SPD guarantee)
        eigvals = torch.where(eigvals <= 0, F.softplus(eigvals) + 1e-4, eigvals)

        return eigvals
        
        """
    def apply_activation(self, eigvals):
           Apply activation to eigenvalues only.
        if self.activation == 'ReEig':
            return F.relu(eigvals)

        elif self.activation == 'PowerEig':
            return torch.sign(eigvals) * torch.abs(eigvals) ** self.power

        elif self.activation == 'SqrtEig':
            return torch.sign(eigvals) * torch.sqrt(torch.clamp(torch.abs(eigvals), min=1e-6))

        elif self.activation == 'ExpEig':
            return torch.exp(eigvals)

        elif self.activation == 'LogEig':
            return torch.log(torch.clamp(eigvals, min=1e-6))

        elif self.activation == 'LearnableScale':
            return eigvals * self.scale

        elif self.activation == 'ThresholdSoftplus':
            return F.softplus(eigvals - self.threshold)

        elif self.activation == 'SpectralNormScale':
            norm = eigvals.norm(p=2, dim=-1, keepdim=True) + 1e-6
            return eigvals / norm * self.scale

        elif self.activation == 'Affine':
            return eigvals * self.affine_a + self.affine_b

        else:
            raise ValueError(f"Unknown activation type: {self.activation}")
"""
