import torch
import torch.nn as nn
from typing import List


class QNetwork(nn.Module):
    def __init__(
        self,
        obs_size: int,
        n_actions: int,
        hidden_layers: List[int] = None,
        activation: str = "relu",
    ):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [128, 128]

        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "elu": nn.ELU,
        }
        act_fn = activations.get(activation, nn.ReLU)

        layers = []
        in_size = obs_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(act_fn())
            in_size = hidden_size
        layers.append(nn.Linear(in_size, n_actions))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)