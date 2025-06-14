#Best of the best 
#GeometricBiasBlock
import torch
import torch.nn as nn

class GeometricBiasBlock(nn.Module):
    def __init__(self, dim, mode='learnable',
                 spd_correction=True,
                 normalize='spectral',
                 scaling='learnable',
                 rank_ratio=0.25,
                 epsilon=1e-4):
        super(GeometricBiasBlock, self).__init__()
        self.mode = mode
        self.dim = dim
        self.spd_correction = spd_correction
        self.normalize = normalize
        self.scaling_type = scaling
        self.epsilon = epsilon

        if mode == 'learnable':
            self.D = nn.Parameter(torch.eye(dim) + 0.01 * torch.randn(dim, dim))

        elif mode == 'adaptive':
            vec_dim = dim * (dim + 1) // 2
            self.adaptor = nn.Sequential(
                nn.Linear(vec_dim, 256),
                nn.ReLU(),
                nn.Linear(256, dim * dim)
            )

        elif mode == 'lowrank_adaptive':
            vec_dim = dim * (dim + 1) // 2
            rank = max(1, int(rank_ratio * dim))
            self.U_generator = nn.Sequential(
                nn.Linear(vec_dim, 128),
                nn.ReLU(),
                nn.Linear(128, dim * rank)
            )
            self.V_generator = nn.Sequential(
                nn.Linear(vec_dim, 128),
                nn.ReLU(),
                nn.Linear(128, dim * rank)
            )
            self.rank = rank

        if scaling == 'learnable':
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer('scale', torch.ones(1))

    def forward(self, X):
        original_shape = X.shape
        if X.dim() == 2:
            X = X.unsqueeze(0)  # Add batch dimension if missing

        batch_size, dim, _ = X.shape
        device, dtype = X.device, X.dtype

        if self.mode == 'none':
            X_bias = X

        elif self.mode == 'learnable':
            D = self.correct_and_normalize(self.D.to(dtype=dtype, device=device))
            X_bias = D @ X @ D.transpose(-1, -2)

        elif self.mode == 'adaptive':
            idx = torch.triu_indices(dim, dim, device=device)
            X_vec = X[:, idx[0], idx[1]]
            self.adaptor = self.adaptor.to(dtype=X.dtype, device=device)  
            D_flat = self.adaptor(X_vec)
            D = D_flat.view(batch_size, dim, dim)
            D = self.correct_and_normalize(D)
            X_bias = D @ X @ D.transpose(-1, -2)

        elif self.mode == 'lowrank_adaptive':
            idx = torch.triu_indices(dim, dim, device=device)
            X_vec = X[:, idx[0], idx[1]]
            self.U_generator = self.U_generator.to(dtype=X.dtype, device=device)  # dtype/device match
            self.V_generator = self.V_generator.to(dtype=X.dtype, device=device)  # dtype/device match
            
            U = self.U_generator(X_vec).view(batch_size, dim, self.rank)
            V = self.V_generator(X_vec).view(batch_size, dim, self.rank)
            D = U @ V.transpose(-1, -2)
            D = self.correct_and_normalize(D)
            X_bias = D @ X @ D.transpose(-1, -2)

        X_bias = self.scale * X_bias
        if original_shape == X_bias.shape[1:]:
            X_bias = X_bias.squeeze(0)
        return X_bias

    def correct_and_normalize(self, D):
        eye = torch.eye(D.size(-1), device=D.device, dtype=D.dtype)

        if self.spd_correction:
            D = D @ D.transpose(-1, -2)
            D = (D + D.transpose(-1, -2)) / 2
            D += self.epsilon * eye.unsqueeze(0) if D.dim() == 3 else self.epsilon * eye

        if self.normalize == 'spectral':
            D = self.safe_spectral_normalize(D)
        elif self.normalize == 'frobenius':
            frob_norm = torch.norm(D, dim=(-2, -1), keepdim=True)
            D /= (frob_norm + 1e-6)
        return D

    def safe_spectral_normalizessss(self, D, eps=1e-6):
        eigvals = torch.linalg.eigvalsh(D)
        max_eig = eigvals.max(dim=-1, keepdim=True)[0].clamp(min=eps)
        if D.dim() == 3:
            max_eig = max_eig.unsqueeze(-1)
        D /= (max_eig + eps)
        return D

    def safe_spectral_normalize(self, D, eps=1e-6):
        """
        Normalize D using spectral norm approximation without explicit eigendecomposition.
        Assumes D is already SPD from prior correction.
        """
        # Approximate spectral norm (largest singular value) via torch.linalg.norm (ord=2)
        spectral_norm = torch.linalg.norm(D, ord=2, dim=(-2, -1), keepdim=True).clamp(min=eps)
        D = D / (spectral_norm + eps)
        return (D + D.transpose(-1, -2)) / 2  # Re-symmetrize for safety
