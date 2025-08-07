import argparse
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import minari
from collections import deque
import random
import tqdm
import os
import time
import math


# 设置随机种子确保可复现
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ================== 核心网络架构 ==================
class FourierFeatures(nn.Module):
    """傅里叶特征编码（高效时间嵌入）"""

    def __init__(self, input_dim, output_dim, scale=10.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(input_dim, output_dim // 2) * scale, requires_grad=False)

    def forward(self, t):
        t_proj = 2 * np.pi * t @ self.B
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)


class VectorizedFlowField(nn.Module):
    """支持批量多样本生成的流场网络"""

    def __init__(self, obs_dim, action_dim, horizon_length,
                 fourier_dim=64, hidden_dims=[256, 256]):
        super().__init__()
        self.time_encoder = FourierFeatures(1, fourier_dim)
        self.input_dim = obs_dim + action_dim * horizon_length + fourier_dim
        self.output_dim = action_dim * horizon_length

        # 修复1: 更稳定的网络初始化，避免BatchNorm在评估时的问题
        layers = []
        prev_dim = self.input_dim
        for hidden_dim in hidden_dims:
            linear = nn.Linear(prev_dim, hidden_dim)
            # Xavier初始化，更稳定
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.ReLU())
            # 修复：使用LayerNorm替代BatchNorm，避免评估时的维度问题
            layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim

        # 输出层使用小的初始化
        output_layer = nn.Linear(prev_dim, self.output_dim)
        nn.init.xavier_uniform_(output_layer.weight, gain=0.01)  # 小增益
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)

        self.net = nn.Sequential(*layers)

    def forward(self, obs, actions, t):
        t_feat = self.time_encoder(t)
        # 向量化处理：支持任意批次维度
        x = torch.cat([obs, actions, t_feat], dim=-1)
        return self.net(x)


class ParallelCritic(nn.Module):
    """支持批量多样本评估的Critic网络"""

    def __init__(self, obs_dim, action_dim, horizon_length,
                 hidden_dims=[256, 256]):
        super().__init__()
        self.horizon_length = horizon_length
        self.action_dim = action_dim
        input_dim = obs_dim + action_dim * horizon_length

        # 修复2: 更稳定的Critic网络设计
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            linear = nn.Linear(prev_dim, hidden_dim)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))  # LayerNorm更稳定
            prev_dim = hidden_dim

        # 输出层特殊初始化，避免Q值爆炸
        output_layer = nn.Linear(prev_dim, 1)
        nn.init.xavier_uniform_(output_layer.weight, gain=0.1)  # 更小的增益
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)

        self.net = nn.Sequential(*layers)

    def forward(self, obs, actions):
        # 确保动作块的维度正确
        batch_size = obs.shape[0]
        if actions.dim() > 2:
            actions = actions.view(batch_size, -1)  # 展平动作块
        elif actions.dim() == 2 and actions.shape[1] != self.horizon_length * self.action_dim:
            actions = actions.view(batch_size, -1)  # 展平动作块
            
        # 合并状态和动作
        x = torch.cat([obs, actions], dim=-1)
        result = self.net(x).squeeze(-1)  # [batch_size]
        return result


# ================== 高效回放���冲区 ==================
class VectorizedReplayBuffer:
    """支持批量存储和检索的优化缓冲区"""

    def __init__(self, horizon_length=5, capacity=1000000):
        self.buffer = deque(maxlen=capacity)
        self.horizon_length = horizon_length
        self._prealloc_buffers = {}

    def _preallocate_buffers(self, sample_shape):
        """预分配内存加速批量操作"""
        for key, shape in sample_shape.items():
            self._prealloc_buffers[key] = np.empty(
                (self.buffer.maxlen, *shape),
                dtype=np.float32
            )
        self._index = 0
        self._full = False

    def add_trajectory(self, observations, actions, rewards, terminations):
        """添加Minari格式的轨迹"""
        T = len(observations)
        for i in range(T - self.horizon_length):
            # 提取轨迹块
            obs = observations[i]
            next_obs = observations[i + self.horizon_length]
            chunk_actions = actions[i:i + self.horizon_length].flatten()
            chunk_rewards = rewards[i:i + self.horizon_length]
            done = any(terminations[i:i + self.horizon_length])

            # 填充不足部分
            if len(chunk_rewards) < self.horizon_length:
                chunk_rewards = np.pad(
                    chunk_rewards,
                    (0, self.horizon_length - len(chunk_rewards)),
                    'constant'
                )

            # 存储到缓冲区
            item = (obs, chunk_actions, chunk_rewards, next_obs, done)

            if self._prealloc_buffers:
                if self._index >= self.buffer.maxlen:
                    self._full = True
                    self._index = 0

                self._prealloc_buffers['observations'][self._index] = obs
                self._prealloc_buffers['actions_chunk'][self._index] = chunk_actions
                self._prealloc_buffers['rewards_chunk'][self._index] = chunk_rewards
                self._prealloc_buffers['next_observations'][self._index] = next_obs
                self._prealloc_buffers['terminations'][self._index] = done
                self._index += 1
            else:
                self.buffer.append(item)

    def sample(self, batch_size):
        """高效批量采样"""
        if self._prealloc_buffers:
            indices = np.random.choice(
                len(self) if not self._full else self.buffer.maxlen,
                batch_size,
                replace=False
            )
            return {
                'observations': torch.from_numpy(self._prealloc_buffers['observations'][indices]),
                'actions_chunk': torch.from_numpy(self._prealloc_buffers['actions_chunk'][indices]),
                'rewards_chunk': torch.from_numpy(self._prealloc_buffers['rewards_chunk'][indices]),
                'next_observations': torch.from_numpy(self._prealloc_buffers['next_observations'][indices]),
                'terminations': torch.from_numpy(self._prealloc_buffers['terminations'][indices])
            }
        else:
            batch = random.sample(self.buffer, batch_size)
            obs, actions, rewards, next_obs, dones = zip(*batch)
            return {
                'observations': torch.FloatTensor(np.array(obs)),
                'actions_chunk': torch.FloatTensor(np.array(actions)),
                'rewards_chunk': torch.FloatTensor(np.array(rewards)),
                'next_observations': torch.FloatTensor(np.array(next_obs)),
                'terminations': torch.FloatTensor(np.array(dones))
            }

    def __len__(self):
        return self._index if self._prealloc_buffers else len(self.buffer)


# ================== 智能体核心 ==================
class QC_FQLAgent(nn.Module):
    """优化后的QC-FQL智能体（支持高效多样化本生成）"""

    def __init__(self, obs_dim, action_dim, horizon_length=5,
                 lr=3e-4, gamma=0.99, tau=0.005, alpha=100.0,
                 actor_type="best-of-n", num_samples=32,
                 flow_steps=10, device=None, action_space=None):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon_length = horizon_length
        self.gamma = gamma
        self.tau = tau
        # 修复3: 降低alpha参数，避免蒸馏损失过大
        self.alpha = min(alpha, 10.0)  # 限制alpha最大值
        self.actor_type = actor_type
        self.num_samples = num_samples
        self.flow_steps = flow_steps
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 动作空间归一化
        self.action_low = torch.tensor(action_space.low, device=self.device)
        self.action_high = torch.tensor(action_space.high, device=self.device)
        self.action_scale = (self.action_high - self.action_low) / 2
        self.action_bias = (self.action_high + self.action_low) / 2

        # 网络架构
        self.flow_net = VectorizedFlowField(obs_dim, action_dim, horizon_length).to(self.device)
        self.critic = ParallelCritic(obs_dim, action_dim, horizon_length).to(self.device)
        self.target_critic = ParallelCritic(obs_dim, action_dim, horizon_length).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        # 蒸馏策略
        if actor_type == "distill":
            self.actor_net = VectorizedFlowField(obs_dim, action_dim, horizon_length).to(self.device)
            # 修复4: 使用更小的学习率避免训练不稳定
            self.actor_optim = optim.Adam(self.actor_net.parameters(), lr=lr * 0.5)

        # 修复5: 使用更小的初始学习率，特别是对critic
        self.flow_optimizer = optim.Adam(self.flow_net.parameters(), lr=lr * 0.5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr * 0.3)

        # 自动混合精度
        self.scaler = torch.amp.GradScaler(self.device) if self.device in ['cuda', 'cpu'] else None

        # 训练状态
        self.step_count = 0

        # 修复6: 添加Q值范围跟踪，用于稳定性监控
        self.q_running_mean = 0.0
        self.q_running_std = 1.0

    @torch.no_grad()
    def vectorized_flow_actions_from_noise(self, obs, noises):
        """从给定噪声生成动作（用于distill loss）"""
        batch_size = obs.shape[0]
        actions = noises.clone()

        # 修复：正确的流匹配ODE积分 (论文Algorithm 3)
        dt = 1.0 / self.flow_steps
        for step in range(self.flow_steps):
            t_val = step * dt
            t = torch.full((actions.shape[0], 1), t_val, device=self.device)
            velocity = self.flow_net(obs, actions, t)
            actions = actions + velocity * dt  # 修复：正确的欧拉积分

        # 动作归一化
        actions = torch.tanh(actions)

        # 修复：正确的动作反归一化 - 处理动作块维度
        # actions shape: [batch_size, action_dim * horizon_length]
        # 需要重塑为 [batch_size, action_dim, horizon_length] 来应用归一化
        actions_reshaped = actions.view(batch_size, self.action_dim, self.horizon_length)

        # 应用动作空间的反归一化
        actions_denormalized = actions_reshaped * self.action_scale.unsqueeze(-1) + self.action_bias.unsqueeze(-1)

        # 重新展平为动作块格式
        actions_denormalized = actions_denormalized.view(batch_size, -1)

        return actions_denormalized  # [batch, action_dim*h]

    @torch.no_grad()
    def vectorized_flow_actions(self, obs, num_samples=None):
        """向量化生成多个动作样本（修复流匹配积分）"""
        num_samples = num_samples or self.num_samples
        batch_size = obs.shape[0]

        # 扩展观测
        obs_expanded = obs.repeat_interleave(num_samples, dim=0)

        # 初始噪声（批量生成）
        actions = torch.randn(
            num_samples * batch_size,
            self.action_dim * self.horizon_length,
            device=self.device
        )

        # 修复：正确的流匹配ODE积分 (论文Algorithm 3)
        dt = 1.0 / self.flow_steps
        for step in range(self.flow_steps):
            t_val = step * dt
            t = torch.full((actions.shape[0], 1), t_val, device=self.device)
            velocity = self.flow_net(obs_expanded, actions, t)
            actions = actions + velocity * dt  # 修复：正确的欧拉积分

        # 动作归一化
        actions = torch.tanh(actions)

        # 修复：正确的动作反归一化 - 处理动作块维度
        # actions shape: [num_samples * batch_size, action_dim * horizon_length]
        # 需要重塑为 [num_samples * batch_size, action_dim, horizon_length] 来应用归一化
        actions_reshaped = actions.view(num_samples * batch_size, self.action_dim, self.horizon_length)

        # 应用动作空间的反归一化
        actions_denormalized = actions_reshaped * self.action_scale.unsqueeze(-1) + self.action_bias.unsqueeze(-1)

        # 重新展平为动作块格式
        actions_denormalized = actions_denormalized.view(num_samples * batch_size, -1)

        return actions_denormalized.view(batch_size, num_samples, -1)  # [batch, num_samples, action_dim*h]

    def sample_actions(self, obs, strategy=None, num_samples=None):
        """高效动作采样（支持多种策略）"""
        strategy = strategy or self.actor_type
        num_samples = num_samples or self.num_samples
        obs = obs.to(self.device)

        if strategy == "best-of-n":
            # 批量生成候选动作
            candidate_actions = self.vectorized_flow_actions(obs, num_samples)  # [batch, num_samples, action_dim*h]

            # 评估Q值
            batch_size = obs.shape[0]
            obs_expanded = obs.unsqueeze(1).repeat(1, num_samples, 1).view(-1, obs.shape[-1])
            q_values = self.critic(obs_expanded, candidate_actions.view(-1, candidate_actions.shape[-1]))
            q_values = q_values.view(batch_size, num_samples)

            # 选择最佳动作
            idx = q_values.argmax(dim=1)
            selected_actions = candidate_actions[torch.arange(batch_size), idx]
            # 确保输出维度正确
            return selected_actions.view(batch_size, -1)

        elif strategy == "distill" and hasattr(self, 'actor_net') and self.actor_net is not None:
            # 蒸馏策��直接生成
            t_zero = torch.zeros(obs.shape[0], 1, device=self.device)
            # 修复：创建正确维度的噪声张量
            noise_input = torch.randn(obs.shape[0], self.action_dim * self.horizon_length, device=self.device)
            raw_actions = self.actor_net(obs, noise_input, t_zero)
            # 确保动作维度正确
            raw_actions = raw_actions.view(-1, self.action_dim, self.horizon_length)
            raw_actions = torch.tanh(raw_actions)
            raw_actions = raw_actions * self.action_scale.unsqueeze(-1) + self.action_bias.unsqueeze(-1)
            return raw_actions.view(obs.shape[0], self.action_dim * self.horizon_length)
        else:
            # 默认流采样
            return self.vectorized_flow_actions(obs, num_samples=1).squeeze(1)

    def train_step(self, batch, offline_mode=False):
        """优化的训练步骤（修复关键训练问题）"""
        # 数据转移到设备
        obs = batch['observations'].to(self.device)
        actions_chunk = batch['actions_chunk'].to(self.device)
        rewards_chunk = batch['rewards_chunk'].to(self.device)
        next_obs = batch['next_observations'].to(self.device)
        dones = batch['terminations'].to(self.device).float()
        
        # 修复：确保dones是一维张量
        if dones.dim() > 1:
            dones = dones.squeeze(-1)
            
        B = obs.shape[0]

        # 添加维度安全检查（论文Section 4.3要求）
        assert actions_chunk.shape == (B, self.action_dim * self.horizon_length), \
            f"动作块维度错误：应为 {(self.action_dim * self.horizon_length)}，实际 {actions_chunk.shape[1]}"

        # ===== 1. BC流匹配训练（修复动作归一化问题）=====
        # 修复：动作归一化 - 正确处理动作块维度
        actions_reshaped = actions_chunk.view(B, self.action_dim, self.horizon_length)

        # 对每个时间步的动作进行归一化
        actions_normalized = (actions_reshaped - self.action_bias.unsqueeze(-1)) / self.action_scale.unsqueeze(-1)
        actions_normalized = torch.clamp(actions_normalized, -1.0, 1.0)

        # 重新展平为 [batch_size, action_dim * horizon_length]
        actions_normalized = actions_normalized.view(B, -1)

        # BC Flow Loss - 流匹配损失
        t = torch.rand(B, 1, device=self.device)
        noise = torch.randn_like(actions_normalized)
        x_t = (1 - t) * noise + t * actions_normalized
        target_velocity = actions_normalized - noise
        pred_velocity = self.flow_net(obs, x_t, t)

        # 如果使用动作块，需要按有效性加权
        if hasattr(batch, 'valid') and 'valid' in batch:
            valid_mask = batch['valid'].to(self.device)
            # 重塑为动作块格式
            valid_reshaped = valid_mask.unsqueeze(-1).expand(-1, -1, self.action_dim).view(B, -1)
            bc_flow_loss = F.mse_loss(pred_velocity * valid_reshaped, target_velocity * valid_reshaped)
        else:
            bc_flow_loss = F.mse_loss(pred_velocity, target_velocity)

        # 初始化其他损失
        distill_loss = torch.tensor(0.0, device=self.device)
        q_loss = torch.tensor(0.0, device=self.device)

        # ===== 2. Distill Loss（仅在distill模式下）=====
        if self.actor_type == "distill" and hasattr(self, 'actor_net') and self.actor_net is not None:
            # 生成目标动作（来自BC流网络）
            with torch.no_grad():
                # 使用相同的噪声生成目标动作
                target_noises = torch.randn(B, self.action_dim * self.horizon_length, device=self.device)
                target_actions = self.vectorized_flow_actions_from_noise(obs, target_noises)
                # 归一化目标动作
                target_actions_reshaped = target_actions.view(B, self.action_dim, self.horizon_length)
                target_actions_normalized = (target_actions_reshaped - self.action_bias.unsqueeze(-1)) / self.action_scale.unsqueeze(-1)
                target_actions_normalized = torch.clamp(target_actions_normalized, -1.0, 1.0)
                target_actions_normalized = target_actions_normalized.view(B, -1)

            # Actor网络生成动作（一步生成）
            actor_noises = torch.randn(B, self.action_dim * self.horizon_length, device=self.device)
            t_zero = torch.zeros(B, 1, device=self.device)
            actor_actions = self.actor_net(obs, actor_noises, t_zero)
            actor_actions = torch.clamp(actor_actions, -1.0, 1.0)

            # Distill Loss
            distill_loss = F.mse_loss(actor_actions, target_actions_normalized)

            # ===== 3. Q Loss（Actor的Q值优化）=====
            # 使用actor生成的动作计算Q值
            qs = self.critic(obs, actor_actions)
            if qs.dim() > 1:
                q = qs.mean(dim=0) if qs.shape[0] > 1 else qs.squeeze(0)
            else:
                q = qs
            q_loss = -q.mean()  # 最大化Q值

        # ===== 4. 总Actor Loss =====
        actor_loss = bc_flow_loss + self.alpha * distill_loss + q_loss

        # 更新流网络（BC部分）
        self.flow_optimizer.zero_grad()
        if offline_mode:
            # 离线模式只训练BC流损失
            bc_flow_loss.backward()
        else:
            # 在线模式训练完整的actor loss
            actor_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.flow_net.parameters(), 1.0)
        self.flow_optimizer.step()

        # 如果有蒸馏网络，单独更新
        if self.actor_type == "distill" and hasattr(self, 'actor_net') and not offline_mode:
            self.actor_optim.zero_grad()
            (distill_loss + q_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), 1.0)
            self.actor_optim.step()

        # 如果是离线预训练模式，跳过Critic训练
        if offline_mode:
            return {
                'flow_loss': bc_flow_loss.item(),
                'bc_flow_loss': bc_flow_loss.item(),
                'critic_loss': 0.0,
                'distill_loss': distill_loss.item(),
                'q_loss': q_loss.item(),
                'actor_loss': actor_loss.item(),
                'q_mean': 0.0,
                'q_min': 0.0,
                'q_max': 0.0,
                'target_q_mean': 0.0,
                'action_norm_mean': actions_normalized.abs().mean().item(),
            }

        # ===== 5. Critic训练（修复目标Q值计算）=====
        with torch.no_grad():
            # 生成下一状态的动作块
            next_actions = self.sample_actions(next_obs)

            # 正确处理下一状态动作的归一化
            next_actions_reshaped = next_actions.view(B, self.action_dim, self.horizon_length)
            next_actions_normalized = (next_actions_reshaped - self.action_bias.unsqueeze(-1)) / self.action_scale.unsqueeze(-1)
            next_actions_normalized = torch.clamp(next_actions_normalized, -1.0, 1.0)
            next_actions_normalized = next_actions_normalized.view(B, self.action_dim * self.horizon_length)

            # 计算未来价值
            next_q = self.target_critic(next_obs, next_actions_normalized)
            next_q = next_q.squeeze(-1) if next_q.dim() > 1 else next_q

            # 修复2: 正确的h步价值备份机制（论文Eq.7）
            # 创建时间步索引 [0, 1, ..., h-1]
            time_steps = torch.arange(self.horizon_length, device=self.device, dtype=torch.float32)
            # 计算折扣因子 [γ^0, γ^1, ..., γ^(h-1)]
            discounts = torch.pow(self.gamma, time_steps)  # [horizon_length]

            # 处理提前终止：创建有效性掩码
            # dones shape: [B], 需要扩展为 [B, horizon_length]
            done_mask = dones.unsqueeze(1).expand(-1, self.horizon_length)  # [B, horizon_length]

            # 如果episode在某步终止，后续奖励应置零
            # 这里简化处理：假设终止发生在chunk的最后
            valid_rewards = rewards_chunk * (1 - done_mask)  # [B, horizon_length]

            # 验证维度
            assert valid_rewards.shape == (B, self.horizon_length), \
                f"rewards shape mismatch: {valid_rewards.shape} vs ({B}, {self.horizon_length})"
            assert discounts.shape == (self.horizon_length,), \
                f"discounts shape mismatch: {discounts.shape} vs ({self.horizon_length},)"

            # 计算h步折扣奖励累积 (论文Eq.7的累积项)
            discounted_rewards = torch.sum(valid_rewards * discounts.unsqueeze(0), dim=1)  # [B]

            # 计算目标Q值：h步奖励 + γ^h * 未来价值
            # 如果episode已终止，未来价值为0
            future_value_mask = (1 - dones)  # [B] - 如果终止则掩盖未来价值
            target_q = discounted_rewards + (self.gamma ** self.horizon_length) * future_value_mask * next_q

            # 最终维度检查
            assert target_q.shape == (B,), f"target_q shape: {target_q.shape}, expected: ({B},)"

        # 当前Q估计
        current_q = self.critic(obs, actions_normalized)
        current_q = current_q.squeeze(-1) if current_q.dim() > 1 else current_q

        # 最终验证：确保维度匹配
        assert current_q.shape == target_q.shape == (B,), \
            f"Shape mismatch - current_q: {current_q.shape}, target_q: {target_q.shape}, expected: ({B},)"

        critic_loss = F.mse_loss(current_q, target_q)

        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # 目标网络软更新
        self.soft_update_target()
        self.step_count += 1

        # 返回增强的监控指标
        return {
            'flow_loss': bc_flow_loss.item(),
            'bc_flow_loss': bc_flow_loss.item(),
            'critic_loss': critic_loss.item(),
            'distill_loss': distill_loss.item(),
            'q_loss': q_loss.item(),
            'actor_loss': actor_loss.item(),
            'q_mean': current_q.mean().item(),
            'q_min': current_q.min().item(),
            'q_max': current_q.max().item(),
            'target_q_mean': target_q.mean().item(),
            'action_norm_mean': actions_normalized.abs().mean().item(),
            'discounted_rewards_mean': discounted_rewards.mean().item(),  # 新增监控
            'future_value_mean': (future_value_mask * next_q).mean().item(),  # 新增监控
        }

    def soft_update_target(self):
        """目标网络软更新"""
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    @torch.no_grad()
    def get_action(self, obs, execute_length=None):
        """实时动作生成，支持单个或批量观测输入"""
        # 确保输入是张量并添加批次维度
        if not isinstance(obs, torch.Tensor):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        else:
            obs_tensor = obs.unsqueeze(0) if obs.dim() == 1 else obs
            
        # 生成动作块
        action_chunk = self.sample_actions(obs_tensor).cpu().numpy().flatten()
        
        # 确保动作维度正确
        assert len(action_chunk) == self.action_dim * self.horizon_length, \
            f"动作块维度错误: {len(action_chunk)} != {self.action_dim} * {self.horizon_length}"
        
        # 返回指定长度的动作
        if execute_length is not None:
            # 确保执行长度不会超出动作块长度
            execute_length = min(execute_length, self.horizon_length)
            return action_chunk[:execute_length * self.action_dim]
        return action_chunk


# ================== 训练和评估框架 ==================
def load_minari_dataset(dataset_name, num_episodes=None):
    """加载Minari数据集并提取轨迹"""
    dataset = minari.load_dataset(dataset_name,download= True)
    episodes = []

    # 按需加载部分数据集
    for i in range(min(num_episodes or len(dataset), len(dataset))):
        episode = dataset[i]
        episodes.append({
            'observations': episode.observations,
            'actions': episode.actions,
            'rewards': episode.rewards,
            'terminations': episode.terminations
        })

    return episodes


def fill_buffer_from_episodes(buffer, episodes):
    """用数据集填充缓冲区"""
    print(f"Filling buffer with {len(episodes)} episodes...")
    start_time = time.time()

    for ep in episodes:
        buffer.add_trajectory(
            ep['observations'],
            ep['actions'],
            ep['rewards'],
            ep['terminations']
        )

    print(f"Buffer filled with {len(buffer)} chunks in {time.time() - start_time:.1f}s")


def run_episode(agent, env, horizon_length, max_steps=1000, render=False, deterministic=False):
    """执行一个episode（修复：动作块原子性执行）"""
    obs, _ = env.reset()
    
    # 观测维度适配
    if len(obs) != agent.obs_dim:
        if len(obs) > agent.obs_dim:
            obs = obs[:agent.obs_dim]
        else:
            padded_obs = np.zeros(agent.obs_dim)
            padded_obs[:len(obs)] = obs
            obs = padded_obs
    
    total_reward = 0
    step_count = 0

    while step_count < max_steps:
        # 核心修复：原子性执行动作块（论文Section 4.1要求）
        action_chunk = agent.get_action(obs)  # 生成完整动作块

        # 将动作块重塑为[horizon_length, action_dim]以按步执行
        actions_to_execute = action_chunk.reshape(horizon_length, agent.action_dim)

        # 原子性执行整个动作块（论文Fig.1左侧要求）
        chunk_reward = 0
        chunk_obs = obs  # 记录动作块开始时的观测

        for step_in_chunk in range(horizon_length):
            if step_count >= max_steps:
                break

            action = actions_to_execute[step_in_chunk]

            # 环境交互
            next_obs, reward, terminated, truncated, _ = env.step(action)

            # 观测维度适配
            if len(next_obs) != agent.obs_dim:
                if len(next_obs) > agent.obs_dim:
                    next_obs = next_obs[:agent.obs_dim]
                else:
                    padded_obs = np.zeros(agent.obs_dim)
                    padded_obs[:len(next_obs)] = next_obs
                    next_obs = padded_obs

            if render:
                env.render()
                time.sleep(0.01)

            chunk_reward += reward
            step_count += 1
            obs = next_obs  # 更新观测用于下一个动作块

            if terminated or truncated:
                break

        total_reward += chunk_reward

        if terminated or truncated:
            break

    return total_reward, step_count


def evaluate_agent(agent, env, horizon_length, num_episodes=5, render=False, deterministic=False):
    """评估智能体性能"""
    total_rewards = []
    total_steps = []

    for ep in range(num_episodes):
        try:
            reward, steps = run_episode(
                agent, env, horizon_length,
                render=render, deterministic=deterministic
            )
            total_rewards.append(reward)
            total_steps.append(steps)
            print(f"Episode {ep + 1}: Steps={steps}, Reward={reward:.1f}")
        except Exception as e:
            print(f"Episode {ep + 1}: Error during evaluation: {str(e)}")
            continue

    if len(total_rewards) > 0:
        avg_reward = np.mean(total_rewards)
        avg_steps = np.mean(total_steps)
        print(f"Evaluation over {len(total_rewards)} episodes: Avg Steps={avg_steps:.1f}, Avg Reward={avg_reward:.2f}")
        return avg_reward
    else:
        print("No successful episodes in evaluation")
        return float('-inf')


# ================== 主训练流程 ==================
def train_qc_fql_agent():
    """QC-FQL训练主流程"""
    parser = argparse.ArgumentParser(description="高效QC-FQL实现")
    parser.add_argument('--env_id', type=str, default='Ant-v5', help='Gym环境ID')
    parser.add_argument('--dataset_name', type=str, default='mujoco/ant/expert-v0', help='Minari数据集')
    parser.add_argument('--horizon', type=int, default=5, help='动作块长度')
    parser.add_argument('--num_samples', type=int, default=32, help='Best-of-N采样数')
    parser.add_argument('--flow_steps', type=int, default=10, help='流匹配步数')
    parser.add_argument('--actor_type', type=str, default='distill', choices=['best-of-n', 'distill'])
    parser.add_argument('--batch_size', type=int, default=256, help='训练批大小')
    parser.add_argument('--num_updates', type=int, default=10000, help='训练更新次数')
    parser.add_argument('--eval_freq', type=int, default=2000, help='评估频率')
    parser.add_argument('--num_eval_episodes', type=int, default=3, help='评估episode数')
    parser.add_argument('--num_init_episodes', type=int, default=100, help='初始化数据集episode数')
    parser.add_argument('--render_eval', action='store_true',default=True, help='评估时渲染')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='auto', help='计算设备')
    args = parser.parse_args()

    # 加载数据集
    print(f"加载数据集: {args.dataset_name}")
    episodes = load_minari_dataset(args.dataset_name, args.num_init_episodes)
    
    # 获取数据集中实际的观测维度和动作维度
    sample_ep = episodes[0]
    dataset_obs_shape = sample_ep['observations'].shape
    dataset_obs_dim = dataset_obs_shape[1] if len(dataset_obs_shape) > 1 else dataset_obs_shape[0]
    
    dataset_action_shape = sample_ep['actions'].shape
    dataset_action_dim = dataset_action_shape[1] if len(dataset_action_shape) > 1 else dataset_action_shape[0]
    
    print(f"数据集中观测维度: {dataset_obs_dim} (shape: {dataset_obs_shape})")
    print(f"数据集中动作维度: {dataset_action_dim} (shape: {dataset_action_shape})")

    # 初始化环境
    set_seed(args.seed)
    device = args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')

    # 从数据集恢复环境而不是使用gym.make
    print(f"从数据集恢复环境: {args.dataset_name}")
    dataset = minari.load_dataset(args.dataset_name)

    # 创建训练环境（不带渲染）
    env = dataset.recover_environment()

    # 创建评估环境（明确关闭渲染模式）
    if args.render_eval:
        try:
            eval_env = dataset.recover_environment(eval_env=True, render_mode='human')
            print("评估环境已设置为渲染模式")
        except Exception as e:
            print(f"渲染模式设置失败，使用无渲染模式: {e}")
            eval_env = dataset.recover_environment(eval_env=True, render_mode=None)
    else:
        eval_env = dataset.recover_environment(eval_env=True, render_mode=None)
        print("评估环境已设置为无渲染模式")

    env_obs_dim = env.observation_space.shape[0]
    env_action_dim = env.action_space.shape[0]
    print(f"环境观测维度: {env_obs_dim}, 动作维度: {env_action_dim}")
    
    # ============ 简���的动作空间处理 ============
    print("\n=== 动作空间适配检��� ===")

    # 直接��gym创建环境获取标准����作空间（如果可能）
    try:
        gym_env = gym.make(args.env_id)
        gym_action_space = gym_env.action_space
        print(f"Gym标准环境动作空间: {gym_action_space}")
        gym_env.close()
    except Exception as e:
        print(f"无法创建标准gym环境: {e}")
        gym_action_space = None

    # 从数据集获取实际动作信息
    dataset_action_low = np.min([ep['actions'].min() for ep in episodes])
    dataset_action_high = np.max([ep['actions'].max() for ep in episodes])
    print(f"数据集动作范围: [{dataset_action_low:.3f}, {dataset_action_high:.3f}]")
    print(f"数据集动作维度: {dataset_action_dim}")
    print(f"环境观测维度: {env_obs_dim}, 动作维度: {env_action_dim}")

    # 智能选择动作空间配置
    if gym_action_space is not None and gym_action_space.shape[0] == dataset_action_dim:
        # 如果gym环境存在且维度匹配，使用gym的动作空间
        print("✅ 使用Gym标准动作空间")
        action_space_for_training = gym_action_space
    else:
        # 否则基于数据集创建动作空间
        print("⚠️  使用数据集推断的动作空间")
        action_space_for_training = gym.spaces.Box(
            low=dataset_action_low,
            high=dataset_action_high,
            shape=(dataset_action_dim,),
            dtype=np.float32
        )

    print(f"训练用动作空间: {action_space_for_training}")

    # 检查是否需要动作转换
    need_action_conversion = (env_action_dim != dataset_action_dim)
    if need_action_conversion:
        print(f"🔄 评估时需要动作���度转换: {dataset_action_dim} -> {env_action_dim}")
    else:
        print("✅ 动作维度匹配���无需转换")

    # 使用数���集维度进行�����练
    obs_dim = dataset_obs_dim
    action_dim = dataset_action_dim
    

    # 初始化缓冲区
    print(f"初始化缓冲区 (horizon={args.horizon}, capacity=1000000)")
    buffer = VectorizedReplayBuffer(horizon_length=args.horizon)

    # 预分配内存
    buffer._preallocate_buffers({
        'observations': (obs_dim,),
        'actions_chunk': (action_dim * args.horizon,),
        'rewards_chunk': (args.horizon,),
        'next_observations': (obs_dim,),
        'terminations': (1,)
    })

    # 填充缓冲区
    fill_buffer_from_episodes(buffer, episodes)

    # 初始化智能体
    print(f"初始化QC-FQL智能体 (actor_type={args.actor_type})")
    agent = QC_FQLAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        horizon_length=args.horizon,
        flow_steps=args.flow_steps,
        num_samples=args.num_samples,
        actor_type=args.actor_type,
        action_space=action_space_for_training,  # 使用智能选择的动作空间
        device=device
    )

    # 修复4: 添加离线预训练阶段（论文Section 5.2）
    offline_pretrain_steps = 10000  # 根据数据集大小调整
    print(f"\n===== 离线预训练阶段 =====")
    print(f"预训练流网络 {offline_pretrain_steps} 步...")

    pretrain_progress = tqdm.tqdm(range(offline_pretrain_steps), desc="离线预训练")
    pretrain_losses = []

    for pretrain_step in pretrain_progress:
        if len(buffer) >= args.batch_size:
            batch = buffer.sample(args.batch_size)
            metrics = agent.train_step(batch, offline_mode=True)  # 仅训练流网络
            pretrain_losses.append(metrics['flow_loss'])

            # 更新进度条
            pretrain_progress.set_postfix({
                'FlowLoss': f"{metrics['flow_loss']:.4f}",
                'ActionNorm': f"{metrics['action_norm_mean']:.3f}",
            })

            # 定期检查预训练收敛
            if pretrain_step % 1000 == 0 and pretrain_step > 0:
                recent_losses = pretrain_losses[-500:] if len(pretrain_losses) >= 500 else pretrain_losses
                avg_loss = np.mean(recent_losses)
                loss_std = np.std(recent_losses)
                print(f"\n  预训练进度 {pretrain_step}/{offline_pretrain_steps}: 流损失均值={avg_loss:.6f}, 标准差={loss_std:.6f}")

                # 早停检查：如果损失已经很稳定，可以提前结束
                if len(recent_losses) >= 500 and loss_std < 0.001:
                    print(f"  ✅ 流损失已收敛，提前结束预训练")
                    break

    print(f"离线预训练完成！最终流损失: {pretrain_losses[-1]:.6f}")

    # 预训练后的验证：检查流网络是否学会模仿
    print("\n[预训练验证] 测试流网络模仿能力...")
    agent.eval()
    with torch.no_grad():
        # 随机采样一些观测，测试动作生成质量
        test_batch = buffer.sample(min(32, len(buffer)))
        test_obs = test_batch['observations'][:5].to(device)  # 取5个样本

        # 生成动作并检查合理性
        generated_actions = agent.sample_actions(test_obs)
        print(f"  生成动作范围: [{generated_actions.min().item():.3f}, {generated_actions.max().item():.3f}]")
        print(f"  生成动作均值: {generated_actions.mean().item():.3f}")
        print(f"  动作块维度: {generated_actions.shape}")
    agent.train()

    # 训练循环
    print(f"\n===== 在线强化学习阶段 =====")
    print(f"开始训练: {args.num_updates}次更新 (batch_size={args.batch_size})")
    best_reward = -np.inf
    progress = tqdm.tqdm(range(args.num_updates), desc="训练")
    metrics_history = []

    # 添加学习率调度器
    flow_scheduler = optim.lr_scheduler.CosineAnnealingLR(agent.flow_optimizer, T_max=args.num_updates)
    critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(agent.critic_optimizer, T_max=args.num_updates)
    if hasattr(agent, 'actor_optim'):
        actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(agent.actor_optim, T_max=args.num_updates)

    for step in progress:
        # 采样批次并训练
        if len(buffer) >= args.batch_size:
            batch = buffer.sample(args.batch_size)
            metrics = agent.train_step(batch)
            metrics_history.append(metrics)

            # 更新学习率
            flow_scheduler.step()
            critic_scheduler.step()
            if hasattr(agent, 'actor_optim'):
                actor_scheduler.step()

            # 更新进度条
            progress.set_postfix({
                'FlowLoss': f"{metrics['flow_loss']:.4f}",
                'CriticLoss': f"{metrics['critic_loss']:.4f}",
                'QMean': f"{metrics['q_mean']:.2f}",
                'QRange': f"[{metrics['q_min']:.1f},{metrics['q_max']:.1f}]",
                'ActNorm': f"{metrics['action_norm_mean']:.3f}",
                'FlowLR': f"{flow_scheduler.get_last_lr()[0]:.1e}"
            })

            # 增��收敛性诊断
            if step % 500 == 0 and step > 0:
                print(f"\n[收敛诊断] Step {step}: 详细分析")
                recent_metrics = metrics_history[-100:] if len(metrics_history) >= 100 else metrics_history

                # Flow Loss趋势分析
                recent_flow_loss = [m['flow_loss'] for m in recent_metrics]
                recent_critic_loss = [m['critic_loss'] for m in recent_metrics]
                recent_q_mean = [m['q_mean'] for m in recent_metrics]

                print(f"  Flow Loss: 当前={metrics['flow_loss']:.6f}, 平均={np.mean(recent_flow_loss):.6f}, 趋势={'下降' if len(recent_flow_loss) > 10 and recent_flow_loss[-5:] < recent_flow_loss[:5] else '稳定/上升'}")
                print(f"  Critic Loss: 当前={metrics['critic_loss']:.6f}, 平均={np.mean(recent_critic_loss):.6f}")
                print(f"  Q值范围: [{metrics['q_min']:.2f}, {metrics['q_max']:.2f}], ��标Q均值={metrics['target_q_mean']:.2f}")
                print(f"  动作归一化均���: {metrics['action_norm_mean']:.4f} (应接近0.5)")

                # 检查是否存在爆炸或消失
                if metrics['q_mean'] > 1e4 or metrics['q_mean'] < -1e4:
                    print(f"  ⚠️  Q值可���爆炸! ���前Q均值: {metrics['q_mean']:.2f}")
                    # 紧急修复：降低学习率
                    for param_group in agent.flow_optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    for param_group in agent.critic_optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    print(f"  🔧 已紧急降低学习率到 {agent.flow_optimizer.param_groups[0]['lr']:.2e}")

                # 检查梯度范数
                flow_grad_norm = 0
                critic_grad_norm = 0
                for p in agent.flow_net.parameters():
                    if p.grad is not None:
                        flow_grad_norm += p.grad.data.norm(2).item() ** 2
                for p in agent.critic.parameters():
                    if p.grad is not None:
                        critic_grad_norm += p.grad.data.norm(2).item() ** 2

                flow_grad_norm = flow_grad_norm ** 0.5
                critic_grad_norm = critic_grad_norm ** 0.5
                print(f"  梯度范数: Flow={flow_grad_norm:.4f}, Critic={critic_grad_norm:.4f}")

                # 动作生成测���
                with torch.no_grad():
                    test_obs = torch.randn(1, obs_dim, device=device)
                    test_actions = agent.sample_actions(test_obs)
                    print(f"  测试动作: 范围=[{test_actions.min().item():.3f}, {test_actions.max().item():.3f}], 均值={test_actions.mean().item():.3f}")

                # 收敛性建议
                if len(recent_flow_loss) >= 50:
                    flow_stability = np.std(recent_flow_loss[-25:]) / (np.mean(recent_flow_loss[-25:]) + 1e-8)
                    if flow_stability < 0.1:
                        print(f"  ✅ Flow Loss已趋于稳定 (变异系数: {flow_stability:.3f})")
                    elif flow_stability > 0.5:
                        print(f"  ⚠️  Flow Loss震荡较大 (变异系数: {flow_stability:.3f})")

                print(f"  ��习率: Flow={flow_scheduler.get_last_lr()[0]:.2e}, Critic={critic_scheduler.get_last_lr()[0]:.2e}")

        # 定期评估
        if step % args.eval_freq == 0 and step > 0:
            print(f"\n评估 @ step {step}/{args.num_updates}")
            avg_reward = evaluate_agent(
                agent, eval_env, args.horizon,  # 使用评估环境
                num_episodes=args.num_eval_episodes,
                render=args.render_eval
            )

            # 保存最佳模型
            if avg_reward > best_reward:
                best_reward = avg_reward
                model_path = f"qc_fql_{args.dataset_name.replace('/', '_')}_best.pt"
                torch.save(agent.state_dict(), model_path)
                print(f"保存最佳模型到 {model_path} (奖励={best_reward:.2f})")

        # 早期评估检查 - 在训练初期就做一次评估看看baseline
        if step == 100:
            print(f"\n[早期评估] @ step {step} - 检查初始性能")
            early_reward = evaluate_agent(
                agent, eval_env, args.horizon,
                num_episodes=1,
                render=False
            )
            print(f"早期评估奖励: {early_reward:.2f}")

            # 纯���仿学习基线测试
            print("\n[纯模仿测试] 测试流网络是否学会了基本的模仿...")
            agent.eval()
            try:
                imitation_reward = evaluate_agent(
                    agent, eval_env, args.horizon,
                    num_episodes=1,
                    render=False,
                    deterministic=True
                )
                print(f"纯模仿奖励: {imitation_reward:.2f}")
            except Exception as e:
                print(f"纯模仿测试失��: {e}")
            agent.train()

    # 最终评估和保存
    print("\n===== 最终评估 =====")
    final_reward = evaluate_agent(
        agent, eval_env, args.horizon,  # 使用评估环境
        num_episodes=10,
        render=args.render_eval
    )
    print(f"最终奖励: {final_reward:.2f} | 最佳奖励: {best_reward:.2f}")

    final_model_path = f"qc_fql_{args.dataset_name.replace('/', '_')}_final.pt"
    torch.save(agent.state_dict(), final_model_path)
    print(f"保存最终模型到 {final_model_path}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    train_qc_fql_agent()