# 核心说明：定义类：Encoder、TransitionModel；所属路径：rsl_rl-history/rsl_rl/networks
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_sizes, out_dim):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs_hist):
        # obs_hist: [batch, in_dim]
        return self.net(obs_hist)


class TransitionModel(nn.Module):
    def __init__(self, in_dim, hidden_sizes, out_dim):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z, action):
        # z: [batch, z_dim], action: [batch, action_dim]
        x = torch.cat([z, action], dim=-1)
        return self.net(x)