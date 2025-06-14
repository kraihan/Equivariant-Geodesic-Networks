#woerking block
import torch
import torch.nn as nn
from torch.linalg import eigh

class RiemannianMeanPool(nn.Module):
    def __init__(self, mode="approximate", max_iter=10, tol=1e-4, taylor_order=5, jitter=1e-5):
        super().__init__()
        assert mode in ["approximate", "true"], "Mode must be 'approximate' or 'true'"
        self.mode = mode
        self.max_iter = max_iter
        self.tol = tol
        self.taylor_order = taylor_order
        self.jitter = jitter

    # ---- Utility Functions ---- #

    def safe_eigh(self, A, eps=None, max_eps=1.0):
        """Eigen-decomposition with adaptive jitter."""
        if eps is None:
            eps = self.jitter
        eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
        jitter = eps
        while jitter <= max_eps:
            try:
                vals, vecs = eigh(A + jitter * eye)
                return vals, vecs
            except RuntimeError:
                jitter *= 10
        # Final fallback
        vals, vecs = eigh(A + max_eps * eye)
        return vals, vecs

    def enforce_spd(self, M, eps=None):
        """Clamp eigenvalues to enforce SPD."""
        vals, vecs = self.safe_eigh(M, eps)
        vals = torch.clamp(vals, min=eps or self.jitter)
        M_spd = vecs @ torch.diag_embed(vals) @ vecs.transpose(-1, -2)
        return (M_spd + M_spd.transpose(-1, -2)) / 2

    def matrix_sqrt(self, A):
        eigvals, eigvecs = self.safe_eigh(A)
        sqrt_vals = torch.sqrt(torch.clamp(eigvals, min=self.jitter))
        return eigvecs @ torch.diag_embed(sqrt_vals) @ eigvecs.transpose(-1, -2)

    def matrix_inv_sqrt(self, A):
        eigvals, eigvecs = self.safe_eigh(A)
        inv_sqrt_vals = 1.0 / torch.sqrt(torch.clamp(eigvals, min=self.jitter))
        return eigvecs @ torch.diag_embed(inv_sqrt_vals) @ eigvecs.transpose(-1, -2)

    def logm_taylor(self, X):
        I = torch.eye(X.shape[-1], device=X.device).expand_as(X)
        Y = X - I
        out = torch.zeros_like(X)
        Yp = I.clone()
        for i in range(1, self.taylor_order + 1):
            Yp = Yp @ Y
            out += ((-1) ** (i + 1)) * Yp / i
        return out

    def expm_taylor(self, X):
        I = torch.eye(X.shape[-1], device=X.device).expand_as(X)
        out = I.clone()
        Xp = I.clone()
        fac = 1.0
        for i in range(1, self.taylor_order + 1):
            Xp = Xp @ X
            fac *= i
            out += Xp / fac
        return out

    # ---- Forward Pass ---- #

    def forward(self, X):
        # Normalize input shape to (B, N, D, D)
        if X.dim() == 2:
            X = X.unsqueeze(0).unsqueeze(0)
        elif X.dim() == 3:
            X = X.unsqueeze(0)
        elif X.dim() != 4:
            raise ValueError("Input must be 2D, 3D, or 4D (B,N,D,D)")

        B, N, D, _ = X.shape

        if self.mode == "approximate":
            L = self.logm_taylor(X)
            M = L.mean(dim=1)
            M = self.expm_taylor(M)
            return self.enforce_spd(M)

        # True Riemannian mean
        G = X.mean(dim=1)
        for _ in range(self.max_iter):
            G_sqrt = self.matrix_sqrt(G)
            G_inv_sqrt = self.matrix_inv_sqrt(G)

            deltas = []
            for i in range(N):
                A = G_inv_sqrt @ X[:, i] @ G_inv_sqrt
                vals, vecs = self.safe_eigh(A)
                log_vals = torch.log(torch.clamp(vals, min=self.jitter))
                log_A = vecs @ torch.diag_embed(log_vals) @ vecs.transpose(-1, -2)
                deltas.append(log_A)

            Delta = torch.stack(deltas, dim=1).mean(dim=1)
            norm_Delta = Delta.norm(dim=(-2, -1)).mean()

            G = G_sqrt @ self.expm_taylor(Delta) @ G_sqrt
            G = self.enforce_spd(G)

            if norm_Delta.item() < self.tol:
                break

        return G
