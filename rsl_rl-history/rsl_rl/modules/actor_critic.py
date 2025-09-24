# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# 核心说明：定义类：Encoder、TransitionModel、ActorCritic；所属路径：rsl_rl-history/rsl_rl/modules

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_sizes, out_dim):
        super().__init__()
        layers = []
        last_dim = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class TransitionModel(nn.Module):
    def __init__(self, in_dim, hidden_sizes, out_dim):
        super().__init__()
        layers = []
        last_dim = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)


        # 计算 obs_dim
        self.obs_dim = num_actor_obs
        self.obs_history_len = 10

        latent_dim = 128  # Encoder 输出维度，与表格对应

        # 编码器
        self.encoder = Encoder(
            in_dim=self.obs_dim * self.obs_history_len,
            hidden_sizes=[256, 128],
            out_dim=latent_dim,
        )

        # Transition 模型
        self.transition_model = TransitionModel(
            in_dim=latent_dim + num_actions,
            hidden_sizes=[256, 128],
            out_dim=latent_dim,
        )

        # actor 输入维度 = 当前 obs + latent z_t
        num_actor_obs += latent_dim

        # critic 输入维度 = 当前 obs + latent z_t
        num_critic_obs += latent_dim

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        
        # 策略网络
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # 价值函数网络
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # 动作噪声
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # 动作分布（在update_distribution中填充）
        self.distribution = None
        # 禁用参数验证以加速
        Normal.set_default_validate_args(False)

    @staticmethod
    # 目前未使用
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    # 我加的
    def encode(self, obs_history: torch.Tensor) -> torch.Tensor:
        #print(obs_history.size())
        return self.encoder(obs_history)

    def predict_next_z(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.transition_model(torch.cat([z, action], dim=-1))

    def get_actor_obs(self, obs, z):
        # Actor 这里 detach z，阻断梯度
        return torch.cat([obs, z], dim=-1)

    def get_critic_obs(self, obs, z):
        return torch.cat([obs, z], dim=-1)

    def update_distribution(self, observations, z):
        #print(z)
        actor_obs = self.get_actor_obs(observations, z)
        # 计算均值
        mean = self.actor(actor_obs)
        # 计算标准偏差
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # 创建分布
        self.distribution = Normal(mean, std)

    def act(self, observations, z, **kwargs):
        self.update_distribution(observations, z)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, z):
        actor_obs = self.get_actor_obs(observations, z)
        actions_mean = self.actor(actor_obs)
        return actions_mean

    def evaluate(self, critic_observations, z, **kwargs):
        critic_obs = self.get_critic_obs(critic_observations, z)
        value = self.critic(critic_obs)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """加载执行者-评论家模型的参数。

        参数:
            state_dict (dict): 模型的状态字典。
            strict (bool): 是否严格强制state_dict中的键与此
                           模块的state_dict()函数返回的键匹配。

        返回:
            bool: 此训练是否恢复之前的训练。此标志被`OnPolicyRunner`的`load()`函数使用，
                  用于确定如何加载进一步的参数（例如与蒸馏相关）。
        """

        super().load_state_dict(state_dict, strict=strict)
        return True
