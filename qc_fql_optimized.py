"""
优化后的QC-FQL实现
主要改进：
1. 清理冗余代码和注释
2. 修复算法实现中的问题
3. 改善代码结构和可读性
4. 优化性能和内存使用
"""

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
from tqdm import tqdm
import os
import time
from typing import Tuple, Dict, Optional, List
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """配置类，统一管理超参数"""
    def __init__(self):
        # 网络参数
        self.hidden_dims = [256, 256]
        self.fourier_dim = 64
        self.fourier_scale = 10.0

        # 训练参数
        self.lr = 3e-4
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 5.0  # 降低蒸馏损失权重
        self.batch_size = 256
        self.buffer_size = 1000000

        # 流匹配参数
        self.flow_steps = 10
        self.num_samples = 32

        # 训练设置
        self.offline_steps = 5000  # 减少预训练步数
        self.eval_freq = 2000
        self.max_grad_norm = 1.0

def set_seed(seed: int = 42):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SinusoidalPosEmb(nn.Module):
    """简化的正弦位置编码（替代傅里叶特征）"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # 修复：确保t是2维的
        if t.dim() == 2:
            embeddings = t * embeddings[None, :]
        else:
            embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class FlowNetwork(nn.Module):
    """流匹配网络（简化版）"""
    def __init__(self, obs_dim: int, action_dim: int, horizon_length: int, config: Config):
        super().__init__()
        self.action_dim = action_dim
        self.horizon_length = horizon_length

        # 时间编码
        self.time_emb = SinusoidalPosEmb(config.fourier_dim)

        # 网络架构
        input_dim = obs_dim + action_dim * horizon_length + config.fourier_dim
        output_dim = action_dim * horizon_length

        layers = []
        prev_dim = input_dim
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim

        # 输出层（小权重初始化）
        final_layer = nn.Linear(prev_dim, output_dim)
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)

        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)
        x = torch.cat([obs, actions, t_emb], dim=-1)
        return self.net(x)

class Critic(nn.Module):
    """Q网络"""
    def __init__(self, obs_dim: int, action_dim: int, horizon_length: int, config: Config):
        super().__init__()
        input_dim = obs_dim + action_dim * horizon_length

        layers = []
        prev_dim = input_dim
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], dim=-1)
        return self.net(x).squeeze(-1)

class ReplayBuffer:
    """经验回放缓冲区（简化版）"""
    def __init__(self, capacity: int, horizon_length: int):
        self.capacity = capacity
        self.horizon_length = horizon_length
        self.buffer = deque(maxlen=capacity)

    def add_trajectory(self, obs: np.ndarray, actions: np.ndarray,
                      rewards: np.ndarray, terminals: np.ndarray):
        """添加轨迹数据"""
        T = len(obs)
        for i in range(T - self.horizon_length):
            chunk = {
                'obs': obs[i],
                'next_obs': obs[i + self.horizon_length],
                'actions': actions[i:i + self.horizon_length].flatten(),
                'rewards': rewards[i:i + self.horizon_length],
                'done': terminals[i:i + self.horizon_length].any()
            }
            self.buffer.append(chunk)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """采样批次数据"""
        batch = random.sample(self.buffer, batch_size)
        
        # 修复：使用numpy数组避免警告
        obs_batch = np.array([x['obs'] for x in batch], dtype=np.float32)
        next_obs_batch = np.array([x['next_obs'] for x in batch], dtype=np.float32)
        actions_batch = np.array([x['actions'] for x in batch], dtype=np.float32)
        rewards_batch = np.array([x['rewards'] for x in batch], dtype=np.float32)
        dones_batch = np.array([float(x['done']) for x in batch], dtype=np.float32)  # 修复布尔值问题
        
        return {
            'obs': torch.from_numpy(obs_batch),
            'next_obs': torch.from_numpy(next_obs_batch),
            'actions': torch.from_numpy(actions_batch),
            'rewards': torch.from_numpy(rewards_batch),
            'dones': torch.from_numpy(dones_batch)
        }

    def __len__(self):
        return len(self.buffer)

class QC_FQL:
    """QC-FQL智能体（优化版）"""
    def __init__(self, obs_dim: int, action_dim: int, horizon_length: int,
                 action_space: gym.Space, config: Config, device: str = 'cuda'):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon_length = horizon_length
        self.device = device
        self.config = config

        # 动作空间归一化
        self.action_low = torch.tensor(action_space.low, device=device)
        self.action_high = torch.tensor(action_space.high, device=device)
        self.action_scale = (self.action_high - self.action_low) / 2
        self.action_bias = (self.action_high + self.action_low) / 2

        # 网络
        self.flow_net = FlowNetwork(obs_dim, action_dim, horizon_length, config).to(device)
        self.critic = Critic(obs_dim, action_dim, horizon_length, config).to(device)
        self.target_critic = Critic(obs_dim, action_dim, horizon_length, config).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        # 优化器
        self.flow_optimizer = optim.Adam(self.flow_net.parameters(), lr=config.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.lr)

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """动作归一化"""
        B = actions.shape[0]
        actions = actions.view(B, self.action_dim, self.horizon_length)
        normalized = (actions - self.action_bias.unsqueeze(-1)) / self.action_scale.unsqueeze(-1)
        return torch.clamp(normalized.view(B, -1), -1.0, 1.0)

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """动作反归一化"""
        B = actions.shape[0]
        actions = torch.tanh(actions)  # 确保在[-1,1]范围
        actions = actions.view(B, self.action_dim, self.horizon_length)
        denormalized = actions * self.action_scale.unsqueeze(-1) + self.action_bias.unsqueeze(-1)
        return denormalized.view(B, -1)

    @torch.no_grad()
    def sample_actions(self, obs: torch.Tensor, num_samples: Optional[int] = None) -> torch.Tensor:
        """生成动作（流匹配采样）"""
        num_samples = num_samples or self.config.num_samples
        batch_size = obs.shape[0]

        # 扩展观测
        obs_expanded = obs.repeat_interleave(num_samples, dim=0)

        # 初始噪声
        actions = torch.randn(
            batch_size * num_samples,
            self.action_dim * self.horizon_length,
            device=self.device
        )

        # 流匹配ODE求解
        dt = 1.0 / self.config.flow_steps
        for step in range(self.config.flow_steps):
            t = torch.full((actions.shape[0], 1), step * dt, device=self.device)
            velocity = self.flow_net(obs_expanded, actions, t)
            actions = actions + velocity * dt

        # 反归一化
        actions = self.denormalize_actions(actions)
        actions = actions.view(batch_size, num_samples, -1)

        # Best-of-N选择
        obs_for_critic = obs.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, obs.shape[-1])
        q_values = self.critic(obs_for_critic, self.normalize_actions(actions.view(-1, actions.shape[-1])))
        q_values = q_values.view(batch_size, num_samples)

        best_indices = q_values.argmax(dim=1)
        return actions[torch.arange(batch_size), best_indices]

    def train_step(self, batch: Dict[str, torch.Tensor], offline_mode: bool = False) -> Dict[str, float]:
        """训练步骤"""
        # 移动到设备
        for key in batch:
            batch[key] = batch[key].to(self.device)

        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_obs']
        dones = batch['dones']

        B = obs.shape[0]

        # 1. 流匹配训练（BC Loss）
        actions_norm = self.normalize_actions(actions)

        # 流匹配损失
        t = torch.rand(B, 1, device=self.device)
        noise = torch.randn_like(actions_norm)
        x_t = (1 - t) * noise + t * actions_norm
        target_velocity = actions_norm - noise
        pred_velocity = self.flow_net(obs, x_t, t)

        flow_loss = F.mse_loss(pred_velocity, target_velocity)

        # 更新流网络
        self.flow_optimizer.zero_grad()
        flow_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.flow_net.parameters(), self.config.max_grad_norm)
        self.flow_optimizer.step()

        if offline_mode:
            return {'flow_loss': flow_loss.item()}

        # 2. Critic训练
        with torch.no_grad():
            next_actions = self.sample_actions(next_obs, num_samples=1).squeeze(1)
            next_actions_norm = self.normalize_actions(next_actions)
            next_q = self.target_critic(next_obs, next_actions_norm)

            # 计算目标Q值（简化的h-step returns）
            discounted_rewards = torch.sum(rewards * (self.config.gamma ** torch.arange(self.horizon_length, device=self.device)), dim=1)
            target_q = discounted_rewards + (self.config.gamma ** self.horizon_length) * next_q * (1 - dones)

        current_q = self.critic(obs, actions_norm)
        critic_loss = F.mse_loss(current_q, target_q)

        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_optimizer.step()

        # 软更新目标网络
        self.soft_update_target()

        return {
            'flow_loss': flow_loss.item(),
            'critic_loss': critic_loss.item(),
            'q_mean': current_q.mean().item(),
            'target_q_mean': target_q.mean().item()
        }

    def soft_update_target(self):
        """软更新目标网络"""
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """获取单个动作"""
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        else:
            obs = obs.unsqueeze(0) if obs.dim() == 1 else obs

        action_chunk = self.sample_actions(obs, num_samples=1).cpu().numpy().flatten()
        return action_chunk

def load_dataset(dataset_name: str, num_episodes: Optional[int] = None) -> List[Dict]:
    """加载Minari数据集"""
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = minari.load_dataset(dataset_name, download=True)

    episodes = []
    max_episodes = min(num_episodes or len(dataset), len(dataset))

    for i in range(max_episodes):
        episode = dataset[i]
        episodes.append({
            'observations': episode.observations,
            'actions': episode.actions,
            'rewards': episode.rewards,
            'terminations': episode.terminations
        })

    logger.info(f"Loaded {len(episodes)} episodes")
    return episodes

def evaluate_agent(agent: QC_FQL, env: gym.Env, horizon_length: int,
                  num_episodes: int = 5) -> float:
    """评估智能体"""
    total_rewards = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        step_count = 0

        while step_count < 1000:  # 最大步数限制
            action_chunk = agent.get_action(obs)
            actions = action_chunk.reshape(horizon_length, agent.action_dim)

            for step in range(horizon_length):
                if step_count >= 1000:
                    break

                next_obs, reward, terminated, truncated, _ = env.step(actions[step])
                total_reward += reward
                step_count += 1
                obs = next_obs

                if terminated or truncated:
                    break

            if terminated or truncated:
                break

        total_rewards.append(total_reward)
        logger.info(f"Episode {ep + 1}: {total_reward:.1f}")

    avg_reward = np.mean(total_rewards)
    logger.info(f"Average reward: {avg_reward:.2f}")
    return avg_reward

def main():
    """主训练函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='mujoco/ant/expert-v0')
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--num_updates', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # 设置
    set_seed(args.seed)
    config = Config()
    device = args.device if torch.cuda.is_available() else 'cpu'

    # 加载数据
    episodes = load_dataset(args.dataset_name, args.num_episodes)

    # 获取维度信息
    sample_obs = episodes[0]['observations'][0]
    sample_action = episodes[0]['actions'][0]
    obs_dim = len(sample_obs)
    action_dim = len(sample_action)

    logger.info(f"Obs dim: {obs_dim}, Action dim: {action_dim}")

    # 创建环境和智能体
    dataset = minari.load_dataset(args.dataset_name)
    env = dataset.recover_environment()
    eval_env = dataset.recover_environment()

    # 创建动作空间
    action_low = np.min([ep['actions'].min() for ep in episodes])
    action_high = np.max([ep['actions'].max() for ep in episodes])
    action_space = gym.spaces.Box(low=action_low, high=action_high, shape=(action_dim,))

    # 初始化智能体
    agent = QC_FQL(obs_dim, action_dim, args.horizon, action_space, config, device)

    # 准备数据
    buffer = ReplayBuffer(config.buffer_size, args.horizon)
    for ep in episodes:
        buffer.add_trajectory(ep['observations'], ep['actions'], ep['rewards'], ep['terminations'])

    logger.info(f"Buffer size: {len(buffer)}")

    # 离线预训练
    logger.info("Starting offline pretraining...")
    for step in tqdm(range(config.offline_steps), desc="Pretraining"):
        if len(buffer) >= config.batch_size:
            batch = buffer.sample(config.batch_size)
            agent.train_step(batch, offline_mode=True)

    # 在线训练
    logger.info("Starting online training...")
    best_reward = -np.inf

    for step in tqdm(range(args.num_updates), desc="Training"):
        if len(buffer) >= config.batch_size:
            batch = buffer.sample(config.batch_size)
            metrics = agent.train_step(batch)

            # 定期评估
            if step % config.eval_freq == 0 and step > 0:
                avg_reward = evaluate_agent(agent, eval_env, args.horizon, 3)
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    torch.save(agent.flow_net.state_dict(), 'best_model.pt')
                    logger.info(f"New best model saved: {best_reward:.2f}")

    env.close()
    eval_env.close()
    logger.info(f"Training completed. Best reward: {best_reward:.2f}")

if __name__ == "__main__":
    main()
