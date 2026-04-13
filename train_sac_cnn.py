import os
import torch
import torch.nn as nn
import numpy as np
from tianshou.env import DummyVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import DiscreteSACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.discrete import Actor, Critic

from gym_wrapper import PensieveGymEnv

# ==========================================
# 1. 核心特征提取器 (1D-CNN) - 极简全 CPU 版
# ==========================================
class PensieveCNN(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        self.conv_throughput = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=4)
        self.conv_delay = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=4)
        
        self.fc = nn.Sequential(
            nn.Linear(1289, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.ReLU()
        )
        # 必须告诉 Tianshou 你的输出维度
        self.output_dim = output_dim 

    def forward(self, obs, state=None, info={}):
        # 因为全在 CPU 上，直接把输入变成最普通的 PyTorch Tensor 即可
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        
        # 兼容性修复：单步测试数据如果是 (6, 8)，增加 batch 维度变成 (1, 6, 8)
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(0)

        # 解析 6x8
        last_quality = obs[:, 0, -1].unsqueeze(1)    
        buffer_size = obs[:, 1, -1].unsqueeze(1)     
        throughput = obs[:, 2, :].unsqueeze(1)       
        delay = obs[:, 3, :].unsqueeze(1)            
        next_chunks = obs[:, 4, :6]                  
        remain_chunks = obs[:, 5, -1].unsqueeze(1)   
        
        # 1D-CNN 时序卷积
        conv_t = torch.relu(self.conv_throughput(throughput)) 
        conv_d = torch.relu(self.conv_delay(delay))           
        
        # 展平与拼接
        flat_t = conv_t.view(conv_t.size(0), -1)              
        flat_d = conv_d.view(conv_d.size(0), -1)              
        concat = torch.cat([last_quality, buffer_size, flat_t, flat_d, next_chunks, remain_chunks], dim=1) 
        
        logits = self.fc(concat)
        return logits, state

# ==========================================
# 2. 训练主逻辑
# ==========================================
def train_discrete_sac():
    # 开 4 个环境并行跑数据，加快 CPU 训练速度
    train_envs = DummyVectorEnv([lambda: PensieveGymEnv(random_seed=i) for i in range(4)])
    test_envs = DummyVectorEnv([lambda: PensieveGymEnv(random_seed=i+100) for i in range(1)])

    action_shape = 6 
    print("========== 正在使用全 CPU 模式训练 Discrete SAC ==========")

    # 独立的大脑：Actor 和 两个 Critic 各自拥有一个 CNN
    net_actor = PensieveCNN(output_dim=128)
    net_c1 = PensieveCNN(output_dim=128)
    net_c2 = PensieveCNN(output_dim=128)

    # 初始化网络，默认都在 CPU
    actor = Actor(net_actor, action_shape, softmax_output=False)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-4)

    critic1 = Critic(net_c1, last_size=action_shape)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=1e-4)
    
    critic2 = Critic(net_c2, last_size=action_shape)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=1e-4)

    # 配置 SAC 策略
    policy = DiscreteSACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic1=critic1,
        critic1_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=0.005,      
        gamma=0.99,     
        alpha=0.2,      
        estimation_step=1
    )

    buffer = VectorReplayBuffer(100000, len(train_envs))
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)

    print("========== 正在收集初始随机数据 (Warm-up) ... ==========")
    train_collector.collect(n_step=2000, random=True)

    print("========== 开始正式训练 ==========")
    def save_best_fn(policy):
        os.makedirs("./models", exist_ok=True)
        torch.save(policy.state_dict(), "./models/sac_cnn_pensieve.pth")

    result = offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=100, step_per_epoch=2000, step_per_collect=10,
        update_per_step=0.1, episode_per_test=10, batch_size=64,
        save_best_fn=save_best_fn
    )

    print(f"\n========== 训练完成！最佳模型已保存 ==========")
    print(result)

if __name__ == "__main__":
    train_discrete_sac()