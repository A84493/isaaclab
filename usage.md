# 使用指南

## 1. 安装

`rsl_rl-history`在基础的rsl_rl上加载了历史obs并结合三元损失进行训练，基于`robot_lab`框架进行训练


### 1.1 安装isaac_lab-4.5.0
https://docs.robotsfan.com/isaaclab/source/setup/installation/pip_installation.html

### 1.2 安装robot-mars
使用安装了 Isaac Lab 的 python 解释器，安装库
```bash
cd robot_mars

python -m pip install -e source/robot_lab
```
通过运行以下命令来打印扩展中的所有可用环境，以验证扩展是否已正确安装：
```bash
python scripts/tools/list_envs.py
```

### 1.3 安装rsl_rl
```bash
cd rsl_rl-history

# 安装依赖
pip install -e .
```
## 2. 训练及推理
```bash
cd robot_mars/
```
粗糙地形-训练：
```bash
python scripts/rsl_rl/base/train.py --task RobotLab-Isaac-Velocity-Roughcomplex-Unitree-Go2-v0 --num_envs 16382 --headless --max_iterations 20000
```
粗糙地形-推理：
```bash
python scripts/rsl_rl/base/play_c1.py --task RobotLab-Isaac-Velocity-Roughcomplex-Unitree-Go2-v0 --checkpoint <pt路径>
python scripts/rsl_rl/base/play_c1.py --task RobotLab-Isaac-Velocity-Roughcomplex-Unitree-Go2-v0 --checkpoint logs/rsl_rl/unitree_go2_rough_complex/2025-09-12_11-34-01/model_8200.pt
```
平地-训练：
```bash
python scripts/rsl_rl/base/train.py --task RobotLab-Isaac-Velocity-Flatcomplex-Unitree-Go2-v0 --num_envs 4096 --headless --max_iterations 3000
```
平地-推理：
```bash
python scripts/rsl_rl/base/play_c1.py --task RobotLab-Isaac-Velocity-Flatcomplex-Unitree-Go2-v0 --checkpoint <pt路径>
```
附上两个已训练好的pt文件，位于 ```robot_mars/logs/```

## 3.查看训练进度
'''bash
tensorboard --logdir=logs/rsl_rl/unitree_go2_rough_complex
'''

## 4.rsl_rl和skrl的差异
1. 算法框架架构：
    - rsl_rl: 专门针对PPO算法优化，使用 OnPolicyRunner (rsl_rl/train.py:160)
    - skrl: 支持多种算法 (PPO/AMP/IPPO/MAPPO)，使用统一的 Runner 接口
  (skrl/train.py:198)
2. 配置管理系统：
- rsl_rl: 强类型配置 RslRlOnPolicyRunnerCfg，Hydra配置系统 (rsl_rl/train.py:83)
- skrl: 字典式配置 agent_cfg: dict，灵活性更高 (skrl/train.py:115)
3. 机器学习框架支持：
- rsl_rl: 仅支持PyTorch，有专门的CUDA优化配置 (rsl_rl/train.py:81-84)
- skrl: 支持PyTorch/JAX/JAX-Numpy多框架 (skrl/train.py:82-85)
4. 训练循环实现：
- rsl_rl: 显式调用 runner.learn() 进行训练 (rsl_rl/train.py:176)
- skrl: 简化的 runner.run() 接口 (skrl/train.py:206)
5. 分布式训练支持：
- rsl_rl: 内置复杂的多GPU分布式配置 (rsl_rl/train.py:111-119)
- skrl: 相对简化的分布式设置 (skrl/train.py:127-128)
6. 版本追踪与日志：
- rsl_rl: 自动Git版本追踪 runner.add_git_repo_to_log() (rsl_rl/train.py:162)
- skrl: 无内置版本追踪功能

设计哲学差异：
- rsl_rl: 专门化、高性能，针对特定算法深度优化
- skrl: 通用化、框架无关，支持多种算法和ML框架的统一接口

- rsl_rl三个play对比
  | 特性     | play_ori.py | play.py | play_c1.py            |
  |--------|-------------|---------|-----------------------|
  | 推理复杂度  | 低           | 低       | 高                     |
  | 内存使用   | 低           | 低       | 高（历史缓冲区）              |
  | 支持策略类型 | MLP         | MLP     | MLP + RNN/Transformer |
  | 观测处理   | 简单          | 简单      | 复杂TensorDict          |