# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain

from rsl_rl.modules import ActorCritic
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import string_to_callable

from ..storage.replay_buffer import StepWiseLatentBuffer


class PPO:
    """近端策略优化算法 (https://arxiv.org/abs/1707.06347)。"""

    policy: ActorCritic
    """执行者-评论家模块。"""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # RND参数
        rnd_cfg: dict | None = None,
        # 对称性参数
        symmetry_cfg: dict | None = None,
        # 分布式训练参数
        multi_gpu_cfg: dict | None = None,
    ):
        # 设备相关参数
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # 多GPU参数
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND组件
        if rnd_cfg is not None:
            # 提取学习率并从原始字典中移除
            learning_rate = rnd_cfg.pop("learning_rate", 1e-3)
            # 创建RND模块
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # 创建RND优化器
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=learning_rate)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # 对称性组件
        if symmetry_cfg is not None:
            # 检查是否启用对称性
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # 打印我们没有使用对称性
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # 如果函数是字符串，则将其解析为函数
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # 检查有效配置
            if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    "Data augmentation enabled but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            # 存储对称性配置
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # PPO组件
        self.policy = policy
        self.policy.to(self.device)
        # 创建优化器
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        # 创建滚动存储
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()

        # PPO参数
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.triplet_coef = 1e-3  #1.0  # 我加的
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # 我加的
        self.latent_buffer = StepWiseLatentBuffer(
            max_steps=1000,
            batch_size=24576,
            latent_dim=128,
            device=self.device
        )

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape
    ):
        # 为RND创建内存 :)
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        else:
            rnd_state_shape = None
        # 创建滚动存储
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            rnd_state_shape,
            self.device,
        )

    def act(self, obs, critic_obs, obs_history_step_old):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()

        # 编码历史观测
        #print(self.storage.obs_history.size())
        #print(i)
        #print(obs.size())
        z_t = self.policy.encode(obs_history_step_old.reshape(obs_history_step_old.shape[0], -1))

        # 采样动作
        self.transition.actions = self.policy.act(obs, z_t.detach()).detach()
        # 计算动作和价值
        #self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(critic_obs, z_t).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # 需要在env.step()之前记录obs和critic_obs
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        return self.transition.actions, z_t

    def process_env_step(self, rewards, dones, infos, z_t, z_t_plus_1, obs_history_step):
        # 记录奖励和完成状态
        # 注意：我们在此处克隆，因为稍后我们会根据超时对奖励进行自举
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.z_t = z_t.clone()
        self.transition.z_t_plus_1 = z_t_plus_1.clone()
        self.transition.obs_history_step = obs_history_step.clone()  # 当前步的obs_h也加进去传送

        # 计算内在奖励并添加到外在奖励
        if self.rnd:
            # 从信息中获取好奇心门控/观测
            rnd_state = infos["observations"]["rnd_state"]
            # 计算内在奖励
            # 注意：rnd_state是标准化后的门控状态（如果使用标准化）
            self.intrinsic_rewards, rnd_state = self.rnd.get_intrinsic_reward(rnd_state)
            # 将内在奖励添加到外在奖励
            self.transition.rewards += self.intrinsic_rewards
            # 记录好奇心门控
            self.transition.rnd_state = rnd_state.clone()

        # 基于超时的自举
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # 记录转换
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs, z_t):
        # 计算最后一步的价值
        last_values = self.policy.evaluate(last_critic_obs, z_t.detach()).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_triplet_loss = 0
        # -- RND损失
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- 对称性损失
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # 小批量生成器
        #if self.policy.is_recurrent:
            #generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        #else:
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        #print(generator)

        # 遍历批次
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
            proprio_history_batch,
            z_t_batch, 
            z_t_plus_1_flat_batch, 
            z_t_n_flat_batch
        ) in generator:  # 这里obs都是老的，没新的，新的根本没进来！！

            # 每个样本的增强数量
            # 我们从1开始，如果使用对称增强则增加
            num_aug = 1
            # 原始批次大小
            original_batch_size = obs_batch.shape[0]
            #print(original_batch_size)

            # 检查是否应该按小批次标准化优势
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # 执行对称增强
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # 使用对称性进行增强
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # 返回形状：[batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                )
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                )
                # 计算每个样本的增强数量
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                # repeat the rest of the batch
                # -- 执行者
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                # -- 评论家
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new 
            # print(z_t.size())
            # print(z_t_new.size())
            # print(actions_batch.size())
            # print(obs_batch.size())
            # -- 执行者
            self.policy.act(obs_batch, z_t_batch.detach(), masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- 评论家
            value_batch = self.policy.evaluate(critic_obs_batch, z_t_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate
                    # Perform this adaptation only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Symmetry loss
            if self.symmetry:
                # obtain the symmetric actions
                # if we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                    )
                    # 计算每个样本的增强数量
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # compute the symmetrically augmented actions
                # 注意：我们假设第一个增强是原始增强。
                #   我们不使用之前的action_batch，因为该动作是从分布中采样的。
                #   然而，对称性损失是使用分布的均值计算的。
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
                )

                # 计算损失（我们跳过第一个增强，因为它是原始增强）
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # 将损失添加到总损失
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # z_t: 前一时刻 latent
            # z_t_new: 当前 step 的 latent
            # actions_batch: t 时刻的动作  我加的
            triplet_loss = self.compute_triplet_loss(z_t_batch, z_t_plus_1_flat_batch, z_t_n_flat_batch, actions_batch)
            loss += self.triplet_coef * triplet_loss

            # 随机网络蒸馏损失
            if self.rnd:
                # 预测嵌入和目标
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # 将损失计算为均方误差
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # 计算梯度
            # -- 对于PPO
            self.optimizer.zero_grad()
            loss.backward()
            # -- 对于RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            # 从所有GPU收集梯度
            if self.is_multi_gpu:
                self.reduce_parameters()

            # 应用梯度
            # -- 对于PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # -- 对于RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # 存储损失
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_triplet_loss += triplet_loss.item()  # 我加的
            # -- RND损失
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- 对称性损失
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # -- 对于PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_triplet_loss /= num_updates  # 我加的
        # -- 对于RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- 对于对称性
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- 清空存储
        self.storage.clear()

        # 构造损失字典
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "triplet_loss": mean_triplet_loss,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

    """
    辅助函数
    """

    def broadcast_parameters(self):
        """向所有GPU广播模型参数。"""
        # 获取当前GPU上的模型参数
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # 广播模型参数
        torch.distributed.broadcast_object_list(model_params, src=0)
        # 从源GPU加载所有GPU上的模型参数
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self):
        """从所有GPU收集梯度并平均。

        此函数在反向传播后调用，以同步所有GPU间的梯度。
        """
        # 创建张量来存储梯度
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # 在所有GPU间平均梯度
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # 获取所有参数
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # 用减少的梯度更新所有参数的梯度
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # 从共享缓冲区复制数据
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # 更新下一个参数的偏移量
                offset += numel

    def compute_triplet_loss(self, z_t, z_t_plus_1_flat_batch, z_t_n_flat_batch, actions, margin=1.0):
        # 预测下一时刻latent (positive)
        tilde_z_t1 = self.policy.predict_next_z(z_t, actions)  # [B, D]

        z_t_new = z_t_plus_1_flat_batch

        z_tn = z_t_n_flat_batch

        # 计算距离
        pos_dist = nn.functional.pairwise_distance(z_t_new, tilde_z_t1, p=2) ** 2
        neg_dist = nn.functional.pairwise_distance(z_t_new, z_tn, p=2) ** 2

        # 计算triplet loss
        triplet_loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0).mean()
        return triplet_loss
