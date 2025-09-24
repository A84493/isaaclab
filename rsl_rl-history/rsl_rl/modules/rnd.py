# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# 核心说明：定义类：RandomNetworkDistillation；所属路径：rsl_rl-history/rsl_rl/modules

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules.normalizer import EmpiricalDiscountedVariationNormalization, EmpiricalNormalization
from rsl_rl.utils import resolve_nn_activation


class RandomNetworkDistillation(nn.Module):
    """随机网络蒸馏 (RND) 实现 [1]

    参考文献:
        .. [1] Burda, Yuri, et al. "Exploration by random network distillation." arXiv preprint arXiv:1810.12894 (2018).
    """

    def __init__(
        self,
        num_states: int,
        num_outputs: int,
        predictor_hidden_dims: list[int],
        target_hidden_dims: list[int],
        activation: str = "elu",
        weight: float = 0.0,
        state_normalization: bool = False,
        reward_normalization: bool = False,
        device: str = "cpu",
        weight_schedule: dict | None = None,
    ):
        """初始化RND模块。

        - 如果 :attr:`state_normalization` 为 True，则使用经验标准化层对输入状态进行标准化。
        - 如果 :attr:`reward_normalization` 为 True，则使用经验折扣变分
          标准化层对内在奖励进行标准化。

        .. note::
            如果预测器和目标网络配置中的隐藏维度为-1，则使用状态数量
            作为隐藏维度。

        参数:
            num_states: 预测器和目标网络的状态/输入数量。
            num_outputs: 预测器和目标网络的输出数量（嵌入大小）。
            predictor_hidden_dims: 预测器网络隐藏维度列表。
            target_hidden_dims: 目标网络隐藏维度列表。
            activation: 激活函数。默认为 "elu"。
            weight: 内在奖励的缩放因子。默认为 0.0。
            state_normalization: 是否标准化输入状态。默认为 False。
            reward_normalization: 是否标准化内在奖励。默认为 False。
            device: 使用的设备。默认为 "cpu"。
            weight_schedule: 用于RND权重参数的调度类型。
                默认为 None，在这种情况下权重参数为常数。
                它是一个包含以下键的字典：

                - "mode": 用于RND权重参数的调度类型。
                    - "constant": 常数权重调度。
                    - "step": 阶段权重调度。
                    - "linear": 线性权重调度。

                对于 "step" 权重调度，需要以下参数：

                - "final_step": 将权重参数设置为最终值的步骤。
                - "final_value": 权重参数的最终值。

                对于 "linear" 权重调度，需要以下参数：
                - "initial_step": 将权重参数设置为初始值的步骤。
                - "final_step": 将权重参数设置为最终值的步骤。
                - "final_value": 权重参数的最终值。
        """
        # 初始化父类
        super().__init__()

        # 存储参数
        self.num_states = num_states
        self.num_outputs = num_outputs
        self.initial_weight = weight
        self.device = device
        self.state_normalization = state_normalization
        self.reward_normalization = reward_normalization

        # 输入门控的标准化
        if state_normalization:
            self.state_normalizer = EmpiricalNormalization(shape=[self.num_states], until=1.0e8).to(self.device)
        else:
            self.state_normalizer = torch.nn.Identity()
        # 内在奖励的标准化
        if reward_normalization:
            self.reward_normalizer = EmpiricalDiscountedVariationNormalization(shape=[], until=1.0e8).to(self.device)
        else:
            self.reward_normalizer = torch.nn.Identity()

        # 更新次数计数器
        self.update_counter = 0

        # 解析权重调度
        if weight_schedule is not None:
            self.weight_scheduler_params = weight_schedule
            self.weight_scheduler = getattr(self, f"_{weight_schedule['mode']}_weight_schedule")
        else:
            self.weight_scheduler = None
        # 创建网络架构
        self.predictor = self._build_mlp(num_states, predictor_hidden_dims, num_outputs, activation).to(self.device)
        self.target = self._build_mlp(num_states, target_hidden_dims, num_outputs, activation).to(self.device)

        # 使目标网络不可训练
        self.target.eval()

    def get_intrinsic_reward(self, rnd_state) -> tuple[torch.Tensor, torch.Tensor]:
        # 注意：计数器更新每个学习迭代中的环境步数
        self.update_counter += 1
        # 标准化RND状态
        rnd_state = self.state_normalizer(rnd_state)
        # 从目标和预测器网络获取RND状态的嵌入
        target_embedding = self.target(rnd_state).detach()
        predictor_embedding = self.predictor(rnd_state).detach()
        # 计算嵌入之间的距离作为内在奖励
        intrinsic_reward = torch.linalg.norm(target_embedding - predictor_embedding, dim=1)
        # 标准化内在奖励
        intrinsic_reward = self.reward_normalizer(intrinsic_reward)

        # 检查权重调度
        if self.weight_scheduler is not None:
            self.weight = self.weight_scheduler(step=self.update_counter, **self.weight_scheduler_params)
        else:
            self.weight = self.initial_weight
        # 缩放内在奖励
        intrinsic_reward *= self.weight

        return intrinsic_reward, rnd_state

    def forward(self, *args, **kwargs):
        raise RuntimeError("Forward method is not implemented. Use get_intrinsic_reward instead.")

    def train(self, mode: bool = True):
        # 设置模块为训练模式
        self.predictor.train(mode)
        if self.state_normalization:
            self.state_normalizer.train(mode)
        if self.reward_normalization:
            self.reward_normalizer.train(mode)
        return self

    def eval(self):
        return self.train(False)

    """
    私有方法
    """

    @staticmethod
    def _build_mlp(input_dims: int, hidden_dims: list[int], output_dims: int, activation_name: str = "elu"):
        """构建目标和预测器网络"""

        network_layers = []
        # 解析隐藏维度
        # 如果维度是-1，则我们使用观测数量
        hidden_dims = [input_dims if dim == -1 else dim for dim in hidden_dims]
        # 解析激活函数
        activation = resolve_nn_activation(activation_name)
        # 第一层
        network_layers.append(nn.Linear(input_dims, hidden_dims[0]))
        network_layers.append(activation)
        # 后续层
        for layer_index in range(len(hidden_dims)):
            if layer_index == len(hidden_dims) - 1:
                # 最后一层
                network_layers.append(nn.Linear(hidden_dims[layer_index], output_dims))
            else:
                # 隐藏层
                network_layers.append(nn.Linear(hidden_dims[layer_index], hidden_dims[layer_index + 1]))
                network_layers.append(activation)
        return nn.Sequential(*network_layers)

    """
    不同的权重调度。
    """

    def _constant_weight_schedule(self, step: int, **kwargs):
        return self.initial_weight

    def _step_weight_schedule(self, step: int, final_step: int, final_value: float, **kwargs):
        return self.initial_weight if step < final_step else final_value

    def _linear_weight_schedule(self, step: int, initial_step: int, final_step: int, final_value: float, **kwargs):
        if step < initial_step:
            return self.initial_weight
        elif step > final_step:
            return final_value
        else:
            return self.initial_weight + (final_value - self.initial_weight) * (step - initial_step) / (
                final_step - initial_step
            )
