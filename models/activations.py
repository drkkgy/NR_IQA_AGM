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
    sigma(alpha * x + beta) [* gamma] with learnable per-channel parameters.

    When *use_gamma* is True an additional learnable scale ``gamma`` is
    multiplied after the sigmoid, matching the ``ParamSigmoid2_2`` variant
    used by some older checkpoints.

    Args:
        dim: hidden size (required when *per_channel=True*).
        init_alpha: initial slope.
        init_beta: initial bias.
        per_channel: if True one set of parameters per feature, else scalars.
        clamp: clamp pre-sigmoid logit to [-clamp, clamp].
        use_gamma: if True, register a learnable ``gamma`` scale parameter.
        init_gamma: initial value of gamma (only used when *use_gamma=True*).
    """

    def __init__(self, dim: int | None = None, init_alpha: float = 1.0,
                 init_beta: float = 0.0, per_channel: bool = True,
                 clamp: float = 20.0, use_gamma: bool = False,
                 init_gamma: float = 2.0):
        super().__init__()
        if per_channel:
            assert dim is not None, "dim (hidden size) required for per-channel parameters"
            self.alpha = nn.Parameter(torch.full((dim,), init_alpha, dtype=torch.float32))
            self.beta  = nn.Parameter(torch.full((dim,), init_beta,  dtype=torch.float32))
        else:
            self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))
            self.beta  = nn.Parameter(torch.tensor(init_beta,  dtype=torch.float32))

        self.use_gamma = use_gamma
        if use_gamma:
            if per_channel:
                self.gamma = nn.Parameter(torch.full((dim,), init_gamma, dtype=torch.float32))
            else:
                self.gamma = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32))

        self.clamp = clamp

    def forward(self, x):
        z = self.alpha * x + self.beta
        if self.clamp is not None:
            z = z.clamp(-self.clamp, self.clamp)
        out = torch.sigmoid(z)
        if self.use_gamma:
            out = self.gamma * out
        return out


class GatedBlend(nn.Module):
    """
    y = w * ParamSigmoid2(x) + (1 - w) * ParamLeakyReLU2(x)
    where w = sigmoid(g).  g is initialised to 0 so w starts at 0.5
    (balanced blend).

    Pass *use_gamma=True* to use the ``ParamSigmoid2_2``-style sigmoid with
    a learnable ``gamma`` scale (needed for some older checkpoints).
    """

    def __init__(self, dim: int, per_channel: bool = True,
                 init_alpha: float = 1.0, init_beta: float = 0.0,
                 init_a: float = 0.25, use_gamma: bool = False):
        super().__init__()
        self.sig_act   = ParamSigmoid2(dim, init_alpha, init_beta, per_channel,
                                        use_gamma=use_gamma)
        self.lrelu_act = ParamLeakyReLU2(dim, init_a, per_channel)

        if per_channel:
            self.g = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        else:
            self.g = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, x):
        w = torch.sigmoid(self.g)
        return w * self.sig_act(x) + (1.0 - w) * self.lrelu_act(x)
