import torch


class HistoryObsBuffer:
    def __init__(self, num_envs: int, history_len: int, obs_dim: int, device: torch.device):
        self.num_envs = num_envs
        self.history_len = history_len
        self.obs_dim = obs_dim
        self.device = device

        # 初始化 [num_envs, history_len, obs_dim] 的历史窗口
        self.buffer = torch.zeros((num_envs, history_len, obs_dim), device=device)

    def add(self, obs: torch.Tensor):
        """
        添加新观察并更新滑动窗口。
        Args:
            obs: [num_envs, obs_dim]
        """
        assert obs.shape == (self.num_envs, self.obs_dim)
        self.buffer[:, :-1] = self.buffer[:, 1:]  # 向左滑动
        self.buffer[:, -1] = obs                 # 插入最新观测

    def get(self) -> torch.Tensor:
        """
        获取当前所有环境的历史观察。
        Returns:
            Tensor of shape [num_envs, history_len, obs_dim]
        """
        return self.buffer

    def clear(self):
        """
        清空所有历史。
        """
        self.buffer.zero_()

    def reset_envs(self, done_mask: torch.Tensor):
        """
        对于 done=True 的环境，将其历史置零。
        Args:
            done_mask: Bool tensor of shape [num_envs]
        """
        assert done_mask.shape[0] == self.num_envs
        #print(self.buffer[done_mask].size())
        self.buffer[done_mask] = 0


class StepWiseLatentBuffer:
    def __init__(self, max_steps: int, batch_size: int, latent_dim: int, device: torch.device):
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.device = device

        self.buffer = torch.zeros((max_steps, batch_size, latent_dim), device=device)
        self.step_ptr = 0
        self.full = False

    def add(self, z_t_new: torch.Tensor):
        """
        z_t_new: [batch_size, latent_dim] — shape must match
        """
        assert z_t_new.shape == (self.batch_size, self.latent_dim), "Shape mismatch when adding to latent buffer"
        self.buffer[self.step_ptr] = z_t_new
        self.step_ptr = (self.step_ptr + 1) % self.max_steps
        #print(self.step_ptr)
        if self.step_ptr == 0:
            self.full = True

    def sample_step(self) -> torch.Tensor:
        """
        Randomly sample one full time step from buffer: returns [batch_size, latent_dim]
        Equivalent to selecting z_{t+n}
        """
        max_steps = self.max_steps if self.full else self.step_ptr
        step_idx = torch.randint(0, max_steps - 1, (1,), device=self.device).item()
        return self.buffer[step_idx]  # shape: [batch_size, latent_dim]
    
    def clear(self):
        """
        Clear the latent buffer and reset step pointer.
        """
        self.buffer.zero_()
        self.step_ptr = 0
        self.full = False
