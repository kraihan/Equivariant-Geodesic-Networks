import torch
import torch.nn as nn
import geoopt


class RiemannianGeodesicDistance(nn.Module):
    """
    Geodesic Distance Classifier (Refined):
    - Supports learnable prototypes on SPD manifold
    - Computes squared affine-invariant distances
    - Robust initialization, scaling, and gradient stability
    """

    def __init__(self, n_classes, matrix_size,
                 manifold=None,
                 init_strategy="random_spd",
                 learnable_scale=True,
                 device=None,
                 dtype=torch.float32):
        super().__init__()
        self.n_classes = n_classes
        self.matrix_size = matrix_size
        self.dtype = dtype
        self.device = device or torch.device("cpu")
        self.manifold = manifold or geoopt.manifolds.SymmetricPositiveDefinite()

        # Initialize learnable prototypes
        if init_strategy == "random_spd":
            protos = [self._random_spd(matrix_size) for _ in range(n_classes)]
        elif init_strategy == "identity":
            protos = [torch.eye(matrix_size, dtype=dtype, device=self.device) for _ in range(n_classes)]
        else:
            raise ValueError("init_strategy must be 'random_spd' or 'identity'")

        self.prototypes = geoopt.ManifoldParameter(
            torch.stack(protos), manifold=self.manifold
        )

        # Optional learnable scaling of distance
        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(1.0, dtype=dtype))
        else:
            self.register_buffer("scale", torch.tensor(1.0, dtype=dtype))

    def _random_spd(self, n):
        """Generate a well-conditioned random SPD matrix (n x n)"""
        A = torch.randn(n, n, dtype=self.dtype, device=self.device)
        return A @ A.T + 1e-2 * torch.eye(n, dtype=self.dtype, device=self.device)

    def forward(self, x):
        """
        Args:
            x: SPD tensor of shape (B, n, n)
        Returns:
            Tensor of shape (B, C): squared distances to class prototypes
        """
        assert x.ndim == 3 and x.shape[1:] == (self.matrix_size, self.matrix_size), \
            f"Expected input shape (B, {self.matrix_size}, {self.matrix_size}), got {x.shape}"

        B, n, _ = x.shape
        dists = []

        for c in range(self.n_classes):
            P = self.prototypes[c]  # (n, n)
            log_X_P = self.manifold.logmap(x, P)  # (B, n, n)
            dist_sq = self.manifold.inner(P, log_X_P, log_X_P)  # (B,)
            dists.append(dist_sq)

        #dists = torch.stack(dists, dim=1)  # (B, C)
        #return self.scale * dists
        # Stack distances for all classes
        dists = torch.stack(dists, dim=1)  # (B, C)
        
        # Step 3: Gradient Lifting - Map logits back to tangent space for backprop
        #dists = self.scale * dists
        #dists.register_hook(self.gradient_lifting)
        
        return dists


    def gradient_lifting(self, grad):
        """
        Lift the gradient from logits back to the manifold space.
        Args:
            grad: Gradient tensor of shape (B, C)
        Returns:
            Lifted gradient of shape (B, n, n)
        """
        # Placeholder for lifted gradients
        lifted_grad = torch.zeros((grad.size(0), self.matrix_size, self.matrix_size), device=grad.device)

        for c in range(self.n_classes):
            P = self.prototypes[c]  # Class prototype of shape (n, n)

            # Gradient w.r.t. the distance function (B, n, n)
            grad_tangent = torch.zeros((grad.size(0), self.matrix_size, self.matrix_size), device=grad.device)

            for b in range(grad.size(0)):
                # Step 1: Logarithmic map to tangent space
                log_X_P = self.manifold.logmap(P, self.prototypes[c])  # (n, n)

                # Step 2: Scale the tangent vector by the gradient from the loss
                tangent_grad = grad[b, c] * log_X_P  # (n, n)

                # Step 3: Lift the gradient back to the manifold space using expmap
                lifted_grad[b] += self.manifold.expmap(P, tangent_grad)

        return lifted_grad
