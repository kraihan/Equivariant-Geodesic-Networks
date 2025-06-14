# This block most Perfect 
#SPD
import torch
import torch.nn as nn

class FastSPDForm(nn.Module):
    def __init__(self, epsilon=1e-3, clamp_eigenvalues=False, min_eigval=1e-3):
        """
        Args:
            epsilon: Base factor for dynamic diagonal shift
            clamp_eigenvalues: Whether to enforce SPD by clamping eigenvalues (costly)
            min_eigval: Minimum eigenvalue after clamping
        """
        super(FastSPDForm, self).__init__()
        self.epsilon = epsilon
        self.clamp_eigenvalues = clamp_eigenvalues
        self.min_eigval = min_eigval

    def forward(self, X):
        """
        Args:
            X: Tensor of shape (..., D, D), any square matrix
        Returns:
            SPD-corrected tensor of same shape
        """
        D = X.shape[-1]
        eye = torch.eye(D, dtype=X.dtype, device=X.device)

        # Symmetrize via X @ X.T
        X_sym = torch.matmul(X, X.transpose(-1, -2))

        # Compute per-matrix trace: (..., 1, 1)
        trace = torch.diagonal(X_sym, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        dynamic_eps = self.epsilon * trace

        # Broadcast dynamic_eps to match X shape
        while dynamic_eps.dim() < X_sym.dim():
            dynamic_eps = dynamic_eps.unsqueeze(0)

        # Shift toward SPD
        X_spd = X_sym + dynamic_eps * eye

        # Optional clamping
        if self.clamp_eigenvalues:
            eigvals, eigvecs = torch.linalg.eigh(X_spd)
            eigvals = torch.clamp(eigvals, min=self.min_eigval)
            X_spd = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)

        return X_spd
