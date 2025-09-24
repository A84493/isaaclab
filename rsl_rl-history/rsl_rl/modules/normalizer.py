# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# 核心说明：定义类：EmpiricalNormalization、EmpiricalDiscountedVariationNormalization、DiscountedAverage；所属路径：rsl_rl-history/rsl_rl/modules

#  Copyright (c) 2020 Preferred Networks, Inc.

from __future__ import annotations

import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """基于经验值标准化数值的均值和方差。"""

    def __init__(self, shape, eps=1e-2, until=None):
        """初始化经验标准化模块。

        参数:
            shape (int or tuple of int): 输入值的形状（不包括批次轴）。
            eps (float): 稳定性的小值。
            until (int or None): 如果指定此参数，链接学习输入值直到批次大小的总和
            超过它。
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    def forward(self, x):
        """基于经验值标准化数值的均值和方差。

        参数:
            x (ndarray or Variable): 输入值

        返回:
            ndarray or Variable: 标准化后的输出值
        """

        if self.training:
            self.update(x)
        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        """学习输入值而不计算它们的输出值"""

        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[0]
        self.count += count_x
        rate = count_x / self.count

        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._std = torch.sqrt(self._var)

    @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean


class EmpiricalDiscountedVariationNormalization(nn.Module):
    """来自Pathak关于PPO大规模研究的奖励标准化。

    奖励标准化。由于奖励函数是非平稳的，标准化奖励的范围是有用的，
    这样价值函数可以快速学习。我们通过将奖励除以折扣奖励总和的标准偏差的运行估计来实现。
    """

    def __init__(self, shape, eps=1e-2, gamma=0.99, until=None):
        super().__init__()

        self.emp_norm = EmpiricalNormalization(shape, eps, until)
        self.disc_avg = DiscountedAverage(gamma)

    def forward(self, rew):
        if self.training:
            # 更新折扣奖励
            avg = self.disc_avg.update(rew)

            # 从折扣奖励更新矩
            self.emp_norm.update(avg)

        if self.emp_norm._std > 0:
            return rew / self.emp_norm._std
        else:
            return rew


class DiscountedAverage:
    r"""奖励的折扣平均值。

    折扣平均值定义为：

    .. math::

        \bar{R}_t = \gamma \bar{R}_{t-1} + r_t

    参数:
        gamma (float): 折扣因子。
    """

    def __init__(self, gamma):
        self.avg = None
        self.gamma = gamma

    def update(self, rew: torch.Tensor) -> torch.Tensor:
        if self.avg is None:
            self.avg = rew
        else:
            self.avg = self.avg * self.gamma + rew
        return self.avg
