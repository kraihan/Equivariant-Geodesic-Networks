#This block is working correctly
#EquivariantBiMap
import torch
import torch.nn as nn
import torch.nn.functional as F

class EquivariantBiMap(nn.Module):
    def __init__(self, input_dim, output_dim, 
                 bias=False,
                 orthogonal_init=True,
                 manifold_regularization=True):
        super(EquivariantBiMap, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.orthogonal_init = orthogonal_init
        self.manifold_regularization = manifold_regularization

        self.W = nn.Parameter(torch.empty(input_dim, output_dim))  # initialized later in reset_parameters()
        if self.bias:
            self.b = nn.Parameter(torch.zeros(output_dim, output_dim))
        else:
            self.register_parameter('b', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.orthogonal_init:
            nn.init.orthogonal_(self.W)
        else:
            nn.init.xavier_uniform_(self.W)

    def forward(self, X):
        # Match dtype and device
        if self.W.dtype != X.dtype or self.W.device != X.device:
            self.W.data = self.W.data.to(dtype=X.dtype, device=X.device)
        if self.b is not None and (self.b.dtype != X.dtype or self.b.device != X.device):
            self.b.data = self.b.data.to(dtype=X.dtype, device=X.device)

        Wt = self.W.transpose(0, 1)  # (output_dim, input_dim)
        Y = torch.matmul(Wt, torch.matmul(X, self.W))  # (batch, output_dim, output_dim)

        if self.b is not None:
            Y = Y + self.b

        return Y

    def orthogonal_regularizer(self):
        if not self.manifold_regularization:
            return 0.0
        I = torch.eye(self.W.shape[1], device=self.W.device, dtype=self.W.dtype)
        WtW = torch.matmul(self.W.transpose(0, 1), self.W)
        loss = F.mse_loss(WtW, I)
        return loss
