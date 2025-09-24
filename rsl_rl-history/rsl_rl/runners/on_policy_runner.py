# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# 核心说明：模块文档首句：在线策略训练运行器模块 负责管理PPO等在线策略强化学习算法的训练和评估过程；定义类：OnPolicyRunner；所属路径：rsl_rl-history/rsl_rl/runners

"""
在线策略训练运行器模块
负责管理PPO等在线策略强化学习算法的训练和评估过程
"""

from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque

import rsl_rl
from rsl_rl.algorithms import PPO, Distillation
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    EmpiricalNormalization,
    StudentTeacher,
    StudentTeacherRecurrent,
)
from rsl_rl.utils import store_code_state

from ..storage.replay_buffer import HistoryObsBuffer


class OnPolicyRunner:
    """
    在线策略运行器，用于强化学习训练和评估。
    
    该类管理整个强化学习训练流程，包括：
    - 环境交互
    - 策略网络管理
    - 算法执行（PPO/Distillation）
    - 数据收集与存储
    - 训练监控与记录
    """

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        """
        初始化在线策略运行器。
        
        参数:
            env: 向量化环境实例
            train_cfg: 训练配置字典（包含算法和策略配置）
            log_dir: 日志目录路径
            device: 计算设备（'cpu'或'cuda'）
        """
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]  # 算法配置
        self.policy_cfg = train_cfg["policy"]  # 策略网络配置
        self.device = device
        self.env = env

        # 检查是否启用多GPU训练
        self._configure_multi_gpu()

        # 根据算法确定训练类型
        if self.alg_cfg["class_name"] == "PPO":
            self.training_type = "rl"  # 强化学习训练
        elif self.alg_cfg["class_name"] == "Distillation":
            self.training_type = "distillation"  # 策略蒸馏训练
        else:
            raise ValueError(f"未找到算法 {self.alg_cfg['class_name']} 的训练类型。")

        # 解析观测空间维度
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]

        # 解析特权观测类型
        if self.training_type == "rl":
            if "critic" in extras["observations"]:
                self.privileged_obs_type = "critic"  # 演员-评论家强化学习，如PPO
            else:
                self.privileged_obs_type = None
        if self.training_type == "distillation":
            if "teacher" in extras["observations"]:
                self.privileged_obs_type = "teacher"  # 策略蒸馏中的教师网络
            else:
                self.privileged_obs_type = None

        # 解析特权观测维度
        if self.privileged_obs_type is not None:
            num_privileged_obs = extras["observations"][self.privileged_obs_type].shape[1]
        else:
            num_privileged_obs = num_obs

        # 动态创建策略类实例
        policy_class = eval(self.policy_cfg.pop("class_name"))
        policy: ActorCritic | ActorCriticRecurrent | StudentTeacher | StudentTeacherRecurrent = policy_class(
            num_obs, num_privileged_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # 解析随机网络蒸馏（RND）门控状态维度
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            # 检查RND门控状态是否存在
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError("在infos['observations']中未找到'rnd_state'键的观测数据。")
            # 获取RND门控状态维度
            num_rnd_state = rnd_state.shape[1]
            # 将RND门控状态添加到配置中
            self.alg_cfg["rnd_cfg"]["num_states"] = num_rnd_state
            # 根据时间步缩放RND权重（类似于legged_gym环境中奖励的缩放方式）
            self.alg_cfg["rnd_cfg"]["weight"] *= env.unwrapped.step_dt

        # 如果使用对称性，则传递环境配置对象
        if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
            # 对称函数使用此对象来处理不同的观测项
            self.alg_cfg["symmetry_cfg"]["_env"] = env

        # 初始化算法实例
        alg_class = eval(self.alg_cfg.pop("class_name"))
        self.alg: PPO | Distillation = alg_class(
            policy, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )

        # 存储训练配置参数
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(
                self.device
            )
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # 不进行标准化处理
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)  # 不进行标准化处理

        # 初始化存储和模型
        self.alg.init_storage(
            self.training_type,
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_privileged_obs],
            [self.env.num_actions],
        )

        self.obs_history_buffer = HistoryObsBuffer(
            num_envs=self.env.num_envs,
            history_len=10,
            obs_dim=num_obs,
            device=self.device
        )

        # 决定是否禁用日志记录
        # 只有rank 0进程（主进程）才记录日志
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        # 日志相关设置
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]
        self._prev_robot_state: dict[str, torch.Tensor] | None = None

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        """
        主要的学习训练循环方法
        
        参数:
            num_learning_iterations: 学习迭代次数
            init_at_random_ep_len: 是否在随机回合长度初始化
        """
        # 初始化日志写入器
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # 启动Tensorboard、Neptune或Wandb等日志记录器，默认使用Tensorboard
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # 检查教师模型是否已加载（用于蒸馏训练）
        if self.training_type == "distillation" and not self.alg.policy.loaded_teacher:
            raise ValueError("教师模型参数未加载。请加载教师模型进行蒸馏训练。")

        # 随机化初始回合长度（用于探索）
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # 开始学习过程
        obs, extras = self.env.get_observations()
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
        self.train_mode()  # 切换到训练模式（例如启用dropout等）

        # 记录保存相关变量
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # 为记录内在奖励和外在奖励创建缓冲区
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # 确保所有参数同步（分布式训练）
        if self.is_distributed:
            print(f"为rank {self.gpu_global_rank}同步参数...")
            self.alg.broadcast_parameters()
            # TODO: 是否需要同步经验标准化器？
            #   目前：不需要，因为它们最终都会"渐近"收敛到相同的值。

        # 开始训练循环
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):  # 每个iter策略不变，update才变
            start = time.time()
            # 经验收集阶段（Rollout）
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # 采样动作
                    # 填充
                    if torch.all(self.obs_history_buffer.buffer == 0):
                        self.obs_history_buffer.buffer[:] = obs.unsqueeze(1).repeat(1, self.obs_history_buffer.history_len, 1)
                    
                    obs_history_step_old = self.obs_history_buffer.get()  # 这里是老的历史obs
                    actions, z_t = self.alg.act(obs, privileged_obs, obs_history_step_old)
                    obs_old = obs # 这里是老的obs，对应z_t
                    # 环境执行一步，获取新的观测
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    # 将数据移动到指定设备
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # 执行观测标准化
                    obs = self.obs_normalizer(obs)
                    if self.privileged_obs_type is not None:
                        privileged_obs = self.privileged_obs_normalizer(
                            infos["observations"][self.privileged_obs_type].to(self.device)
                        )
                    else:
                        privileged_obs = obs

                    # 存储旧的obs到缓冲区
                    self.obs_history_buffer.add(obs_old)
                    # 这里是已经把第t步的obs加进历史里了
                    obs_history_step = self.obs_history_buffer.get() # 这里是新的历史obs
                    z_t_plus_1 = self.alg.policy.encode(obs_history_step.reshape(obs_history_step.shape[0], -1))  # 直接在这算z_t+1
                    # process the step 就是你了！！！！在这里也会处理当前的obs_history
                    self.alg.process_env_step(rewards, dones, infos, z_t.detach(), z_t_plus_1, obs_history_step)  # 需要detach吗？

                    # 提取内在奖励（仅用于日志记录）
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    # 记录保存
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        # 更新奖励统计
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        # 更新回合长度
                        cur_episode_length += 1
                        # 清除已完成回合的数据
                        # -- 通用部分
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        # -- 内在奖励和外在奖励
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                        # --- 清除 done 环境的 obs 历史
                        #print(dones)
                        done_mask = dones.squeeze(-1).bool()  # [num_envs] bool tensor
                        #print(done_mask)
                        self.obs_history_buffer.reset_envs(done_mask)

                stop = time.time()
                collection_time = stop - start
                start = stop
                # 计算回报值，这里使用的是最后一个观测
                if self.training_type == "rl":
                    self.alg.compute_returns(privileged_obs, z_t)

            # 更新策略
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # 每个iter清空一次潜在空间缓冲区
            self.alg.latent_buffer.clear()

            # 记录日志信息
            if self.log_dir is not None and not self.disable_logs:
                # 记录信息
                self.log(locals())
                # 保存模型
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # 清除回合信息
            ep_infos.clear()
            # 保存代码状态
            if it == start_iter and not self.disable_logs:
                # 获取所有差异文件
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # 如果可能的话，将它们存储到wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # 训练完成后保存最终模型
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        """记录训练信息的方法"""
        # 计算收集的样本数量
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # 更新总时间步数和总时间
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- 回合信息
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # 处理标量和零维张量信息
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # 记录到日志器和终端
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # -- 损失函数
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        # -- 策略网络
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # -- 性能指标
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- 训练指标
        if len(locs["rewbuffer"]) > 0:
            # 分别记录内在奖励和外在奖励
            if self.alg.rnd:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])
            # 其他所有指标
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb不支持非整数x轴日志记录
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        # 记录机器人动力学诊断信息
        self._log_robot_diagnostics(locs["it"])

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            # -- 损失指标
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            # -- 奖励指标
            if self.alg.rnd:
                log_string += (
                    f"""{'Mean extrinsic reward:':>{pad}} {statistics.mean(locs['erewbuffer']):.2f}\n"""
                    f"""{'Mean intrinsic reward:':>{pad}} {statistics.mean(locs['irewbuffer']):.2f}\n"""
                )
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            # -- 轮次信息
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime(
                "%H:%M:%S",
                time.gmtime(
                    self.tot_time / (locs['it'] - locs['start_iter'] + 1)
                    * (locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])
                )
            )}\n"""
        )
        print(log_string)

    def _log_robot_diagnostics(self, iteration: int) -> None:
        """Log detailed kinematic and dynamic statistics to TensorBoard."""
        if self.writer is None or self.disable_logs:
            return

        base_env = getattr(self.env, "unwrapped", None)
        if base_env is None or not hasattr(base_env, "scene"):
            return

        try:
            robot = base_env.scene["robot"]
        except (AttributeError, KeyError):
            return

        data = getattr(robot, "data", None)
        if data is None:
            return

        metrics: dict[str, tuple[torch.Tensor | None, bool]] = {}

        def _get_tensor(target, *names: str) -> torch.Tensor | None:
            for name in names:
                value = getattr(target, name, None)
                if isinstance(value, torch.Tensor):
                    return value.clone()
            return None

        def _store_metric(name: str, tensor: torch.Tensor | None, inferred: bool = False) -> None:
            metrics[name] = (tensor, inferred)

        def _metric_value(name: str) -> torch.Tensor | None:
            entry = metrics.get(name)
            return entry[0] if entry else None

        _store_metric("joint_torque", _get_tensor(data, "applied_torque", "joint_torques"))
        _store_metric("joint_position", _get_tensor(data, "joint_pos"))
        _store_metric("joint_velocity", _get_tensor(data, "joint_vel"))
        joint_acc = _get_tensor(data, "joint_acc")

        _store_metric("base_position", _get_tensor(data, "root_pos_w"))
        _store_metric("base_orientation_quat", _get_tensor(data, "root_quat_w"))
        _store_metric("base_linear_velocity", _get_tensor(data, "root_lin_vel_w"))
        _store_metric("base_angular_velocity", _get_tensor(data, "root_ang_vel_w"))

        lin_acc = _get_tensor(data, "root_lin_acc_w")
        ang_acc = _get_tensor(data, "root_ang_acc_w")

        step_dt = getattr(base_env, "step_dt", None)

        lin_acc_inferred = False
        base_lin_vel = _metric_value("base_linear_velocity")
        if lin_acc is None and base_lin_vel is not None and self._prev_robot_state:
            prev = self._prev_robot_state.get("base_linear_velocity")
            if prev is not None and step_dt:
                prev = prev.to(base_lin_vel.device)
                lin_acc = (base_lin_vel - prev) / step_dt
                lin_acc_inferred = True
        _store_metric("base_linear_acceleration", lin_acc, lin_acc_inferred)

        ang_acc_inferred = False
        base_ang_vel = _metric_value("base_angular_velocity")
        if ang_acc is None and base_ang_vel is not None and self._prev_robot_state:
            prev_ang = self._prev_robot_state.get("base_angular_velocity")
            if prev_ang is not None and step_dt:
                prev_ang = prev_ang.to(base_ang_vel.device)
                ang_acc = (base_ang_vel - prev_ang) / step_dt
                ang_acc_inferred = True
        _store_metric("base_angular_acceleration", ang_acc, ang_acc_inferred)

        contact_sensor = None
        try:
            contact_sensor = base_env.scene.sensors["contact_forces"]
        except (AttributeError, KeyError, TypeError):
            contact_sensor = None

        if contact_sensor is not None and hasattr(contact_sensor, "data"):
            force_history = _get_tensor(contact_sensor.data, "net_forces_w_history", "net_forces_w")
            torque_history = _get_tensor(contact_sensor.data, "net_torques_w_history", "net_torques_w")
            if force_history is not None:
                foot_force = force_history[:, -1] if force_history.ndim >= 3 else force_history
                _store_metric("foot_force", foot_force)
            if torque_history is not None:
                foot_torque = torque_history[:, -1] if torque_history.ndim >= 3 else torque_history
                _store_metric("foot_torque", foot_torque)

        joint_vel_value = _metric_value("joint_velocity")
        joint_acc_inferred = False
        if joint_acc is None and joint_vel_value is not None and self._prev_robot_state:
            prev_joint_vel = self._prev_robot_state.get("joint_velocity")
            if prev_joint_vel is not None and step_dt:
                prev_joint_vel = prev_joint_vel.to(joint_vel_value.device)
                joint_acc = (joint_vel_value - prev_joint_vel) / step_dt
                joint_acc_inferred = True
        _store_metric("joint_acceleration", joint_acc, joint_acc_inferred)

        joint_torque_value = _metric_value("joint_torque")
        if joint_torque_value is not None and joint_vel_value is not None:
            joint_power = torch.abs(joint_torque_value * joint_vel_value)
            _store_metric("joint_power", joint_power)

        def _flatten(tensor: torch.Tensor) -> torch.Tensor:
            tensor = tensor.detach()
            if tensor.is_cuda:
                tensor = tensor.cpu()
            return tensor.reshape(-1).float()

        def _log_stats(prefix: str, tensor: torch.Tensor, inferred: bool = False) -> None:
            flat = _flatten(tensor)
            if flat.numel() == 0:
                return
            stats = {
                "mean": flat.mean().item(),
                "std": flat.std(unbiased=False).item() if flat.numel() > 1 else 0.0,
                "p25": torch.quantile(flat, 0.25).item(),
                "p50": torch.quantile(flat, 0.5).item(),
                "p75": torch.quantile(flat, 0.75).item(),
            }
            for suffix, value in stats.items():
                tag = f"Diagnostics/{prefix}{'_inferred' if inferred else ''}_{suffix}"
                self.writer.add_scalar(tag, value, iteration)

        def _log_vector_components(prefix: str, tensor: torch.Tensor, labels: tuple[str, ...], inferred: bool) -> None:
            flat = tensor.detach()
            if flat.is_cuda:
                flat = flat.cpu()
            flat = flat.reshape(-1, flat.shape[-1]).float()
            for idx, axis in enumerate(labels):
                component = flat[:, idx]
                if component.numel() == 0:
                    continue
                stats = {
                    "mean": component.mean().item(),
                    "std": component.std(unbiased=False).item() if component.numel() > 1 else 0.0,
                    "p25": torch.quantile(component, 0.25).item(),
                    "p50": torch.quantile(component, 0.5).item(),
                    "p75": torch.quantile(component, 0.75).item(),
                }
                for suffix, value in stats.items():
                    tag = f"Diagnostics/{prefix}{'_inferred' if inferred else ''}_{axis}_{suffix}"
                    self.writer.add_scalar(tag, value, iteration)

        for name, (tensor, inferred) in metrics.items():
            if tensor is None:
                continue
            _log_stats(name, tensor, inferred)
            if tensor.ndim >= 2 and tensor.shape[-1] in (3, 4):
                axis_labels = ("x", "y", "z", "w")[: tensor.shape[-1]]
                _log_vector_components(name, tensor, axis_labels, inferred)

        base_orientation_quat = _metric_value("base_orientation_quat")
        if base_orientation_quat is not None:
            quat = base_orientation_quat.detach()
            if quat.is_cuda:
                quat = quat.cpu()
            if quat.ndim == 1:
                quat = quat.unsqueeze(0)
            w, x, y, z = quat.unbind(dim=-1)
            t0 = 2.0 * (w * x + y * z)
            t1 = 1.0 - 2.0 * (x * x + y * y)
            roll = torch.atan2(t0, t1)

            t2 = 2.0 * (w * y - z * x)
            t2 = torch.clamp(t2, -1.0, 1.0)
            pitch = torch.asin(t2)

            t3 = 2.0 * (w * z + x * y)
            t4 = 1.0 - 2.0 * (y * y + z * z)
            yaw = torch.atan2(t3, t4)

            euler = torch.stack((roll, pitch, yaw), dim=-1)
            _log_stats("base_orientation_euler", euler, True)
            _log_vector_components("base_orientation_euler", euler, ("roll", "pitch", "yaw"), True)

        state_cache: dict[str, torch.Tensor] = {}
        if joint_vel_value is not None:
            state_cache["joint_velocity"] = joint_vel_value.detach().cpu()
        if base_lin_vel is not None:
            state_cache["base_linear_velocity"] = base_lin_vel.detach().cpu()
        if base_ang_vel is not None:
            state_cache["base_angular_velocity"] = base_ang_vel.detach().cpu()
        self._prev_robot_state = state_cache

    def save(self, path: str, infos=None):
        # -- 保存模型
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # -- 保存RND模型（如果使用）
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        # -- 保存观测标准化器（如果使用）
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()

        # 保存模型
        torch.save(saved_dict, path)

        # 上传模型到外部日志服务
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False)
        # -- 加载模型
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        # -- 加载RND模型（如果使用）
        if self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        # -- 加载观测标准化器（如果使用）
        if self.empirical_normalization:
            if resumed_training:
                # 如果恢复之前的训练，为执行者/学生加载执行者/学生标准化器
                # 为评论家/教师加载评论家/教师标准化器
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
            else:
                # 如果训练未恢复但加载了模型，此次运行必须是强化学习训练后的蒸馏训练
                # 因此为教师模型加载执行者标准化器。学生的标准化器
                # 不被加载，因为观测空间可能与之前的强化学习训练不同。
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        # -- 加载优化器（如果使用）
        if load_optimizer and resumed_training:
            # -- 算法优化器
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # -- RND优化器（如果使用）
            if self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        # -- 加载当前学习迭代
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # 切换到评估模式（例如关闭dropout）
        if device is not None:
            self.alg.policy.to(device)
        policy = self.alg.policy.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.policy.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def train_mode(self):
        # -- PPO算法
        self.alg.policy.train()
        # -- RND模型
        if self.alg.rnd:
            self.alg.rnd.train()
        # -- 标准化
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.privileged_obs_normalizer.train()

    def eval_mode(self):
        # -- PPO算法
        self.alg.policy.eval()
        # -- RND模型
        if self.alg.rnd:
            self.alg.rnd.eval()
        # -- 标准化
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    """
    辅助函数。
    """

    def _configure_multi_gpu(self):
        """配置多GPU训练。"""
        # 检查是否启用分布式训练
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # 如果不是分布式训练，设置本地和全局排名为0并返回
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # 获取排名和世界大小
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # 创建配置字典
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # 主进程的排名
            "local_rank": self.gpu_local_rank,  # 当前进程的排名
            "world_size": self.gpu_world_size,  # 进程总数
        }

        # 检查用户是否为本地排名指定了设备
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # 验证多GPU配置
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # 初始化torch分布式
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # 将设备设置为本地排名
        torch.cuda.set_device(self.gpu_local_rank)
