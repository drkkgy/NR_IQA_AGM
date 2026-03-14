"""
Quality-prediction MLP heads.
Author: Ankit Yadav
"""
import torch.nn as nn

from .activations import ParamLeakyReLU2, GatedBlend


class MLP3_Gated(nn.Module):
    """
    Linear -> GatedBlend(ParamSig + ParamLReLU) -> Linear -> ParamLeakyReLU -> Linear -> score
    """

    def __init__(self, input_dim: int = 1024, hidden: int = 512,
                 per_channel: bool = True):
        super().__init__()
        self.fc1  = nn.Linear(input_dim, hidden)
        self.act1 = GatedBlend(hidden, per_channel)
        self.fc2  = nn.Linear(hidden, hidden)
        self.act2 = ParamLeakyReLU2(hidden, init_a=0.25, per_channel=per_channel)
        self.fc3  = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)


class mlp_3_layer(nn.Module):
    """
    Simple 3-layer ReLU MLP (baseline head).
    """

    def __init__(self, input_dim: int = 32768, hidden: int = 512):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.adapter(x)
