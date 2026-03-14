"""
Learnable activation functions for the quality prediction MLP.
Author: Ankit Yadav
"""
import torch
import torch.nn as nn


class ParamLeakyReLU2(nn.Module):
    """
    Learnable Leaky-ReLU / PReLU with either a single scalar or per-channel
    negative slopes.

    Args:
        dim: hidden size (required when *per_channel=True*).
        init_a: initial negative slope.
        per_channel: if True one slope per feature, else a single scalar.
    """

    def __init__(self, dim: int | None = None, init_a: float = 0.25,
                 per_channel: bool = True):
        super().__init__()
        if per_channel:
            assert dim is not None, "dim (hidden size) required for per-channel slopes"
            self.a = nn.Parameter(torch.full((dim,), init_a, dtype=torch.float32))
        else:
            self.a = nn.Parameter(torch.tensor(init_a, dtype=torch.float32))

    def forward(self, x):
        return torch.where(x >= 0, x, self.a * x)


class ParamSigmoid2(nn.Module):
    """
    sigma(alpha * x + beta) with learnable *alpha* (slope) and *beta* (bias).

    Args:
        dim: hidden size (required when *per_channel=True*).
        init_alpha: initial slope.
        init_beta: initial bias.
        per_channel: if True one pair per feature, else scalars.
        clamp: clamp pre-sigmoid logit to [-clamp, clamp].
    """

    def __init__(self, dim: int | None = None, init_alpha: float = 1.0,
                 init_beta: float = 0.0, per_channel: bool = True,
                 clamp: float = 20.0):
        super().__init__()
        if per_channel:
            assert dim is not None, "dim (hidden size) required for per-channel parameters"
            self.alpha = nn.Parameter(torch.full((dim,), init_alpha, dtype=torch.float32))
            self.beta  = nn.Parameter(torch.full((dim,), init_beta,  dtype=torch.float32))
        else:
            self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))
            self.beta  = nn.Parameter(torch.tensor(init_beta,  dtype=torch.float32))
        self.clamp = clamp

    def forward(self, x):
        z = self.alpha * x + self.beta
        if self.clamp is not None:
            z = z.clamp(-self.clamp, self.clamp)
        return torch.sigmoid(z)


class GatedBlend(nn.Module):
    """
    y = w * ParamSigmoid2(x) + (1 - w) * ParamLeakyReLU2(x)
    where w = sigmoid(g).  g is initialised to 0 so w starts at 0.5
    (balanced blend).
    """

    def __init__(self, dim: int, per_channel: bool = True,
                 init_alpha: float = 1.0, init_beta: float = 0.0,
                 init_a: float = 0.25):
        super().__init__()
        self.sig_act   = ParamSigmoid2(dim, init_alpha, init_beta, per_channel)
        self.lrelu_act = ParamLeakyReLU2(dim, init_a, per_channel)

        if per_channel:
            self.g = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        else:
            self.g = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, x):
        w = torch.sigmoid(self.g)
        return w * self.sig_act(x) + (1.0 - w) * self.lrelu_act(x)
