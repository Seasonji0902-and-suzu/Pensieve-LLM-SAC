import os
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

from gym_wrapper import PensieveGymEnv

# ==========================================
# 🌟 核心突破：高度还原 Pensieve 架构的 1D-CNN 特征提取器
# ==========================================
class PensieveFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        # 必须调用父类初始化
        super().__init__(observation_space, features_dim)
        
        # 针对历史网速 (Throughput) 的一维卷积: 输入通道 1, 输出通道 128, 卷积核 4
        self.conv_throughput = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=4)
        # 针对历史延迟 (Delay) 的一维卷积
        self.conv_delay = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=4)
        
        # 卷积后的维度计算:
        # 输入长度是 8，kernel_size 是 4，输出长度 = 8 - 4 + 1 = 5
        # 展平后每个特征的维度 = 128 * 5 = 640
        
        # 拼接后的总维度计算:
        # 1 (上个码率) + 1 (当前缓冲) + 640 (网速特征) + 640 (延迟特征) + 6 (下个块大小) + 1 (剩余块数) = 1289
        
        self.linear = nn.Sequential(
            nn.Linear(1289, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations 传进来时的 shape 是 (batch_size, 6, 8)
        
        # 1. 拆解状态矩阵 (参考 env.py 中的赋值逻辑)
        last_quality = observations[:, 0, -1].unsqueeze(1)    # (batch, 1)
        buffer_size = observations[:, 1, -1].unsqueeze(1)     # (batch, 1)
        throughput = observations[:, 2, :].unsqueeze(1)       # (batch, 1, 8) 增加通道维度
        delay = observations[:, 3, :].unsqueeze(1)            # (batch, 1, 8)
        next_chunks = observations[:, 4, :6]                  # (batch, 6)
        remain_chunks = observations[:, 5, -1].unsqueeze(1)   # (batch, 1)
        
        # 2. 时序特征卷积提取
        conv_t = torch.relu(self.conv_throughput(throughput)) # (batch, 128, 5)
        conv_d = torch.relu(self.conv_delay(delay))           # (batch, 128, 5)
        
        # 3. 展平卷积结果
        flat_t = conv_t.view(conv_t.size(0), -1)              # (batch, 640)
        flat_d = conv_d.view(conv_d.size(0), -1)              # (batch, 640)
        
        # 4. 拼接所有特征
        concat = torch.cat([last_quality, buffer_size, flat_t, flat_d, next_chunks, remain_chunks], dim=1) # (batch, 1289)
        
        # 5. 降维输出给 DQN 的后续全连接层
        return self.linear(concat)

# ==========================================
# 训练主逻辑
# ==========================================
def train_dqn_cnn(total_timesteps=300000):
    log_dir = "./logs/CNN_DQN/"
    os.makedirs(log_dir, exist_ok=True)

    env = PensieveGymEnv(random_seed=42)
    env = Monitor(env, log_dir)

    print("========== 正在初始化带有 1D-CNN 提取器的 DQN 模型 ==========")
    
    # 将我们手写的网络告诉 SB3
    policy_kwargs = dict(
        features_extractor_class=PensieveFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = DQN(
        policy="MlpPolicy",   # 依然叫 MlpPolicy，但底层提取器已被替换
        env=env, 
        policy_kwargs=policy_kwargs, # 注入自定义网络
        verbose=1, 
        learning_rate=1e-3, 
        buffer_size=100000,   
        learning_starts=5000, 
        batch_size=64,               
        gamma=0.99,                  
        target_update_interval=1000, 
        exploration_fraction=0.15,   
        exploration_final_eps=0.05
    )

    print(f"========== 开始训练，目标步数: {total_timesteps} ==========")
    model.learn(total_timesteps=total_timesteps)

    model_path = "./models/cnn_dqn_pensieve"
    os.makedirs("./models/", exist_ok=True)
    model.save(model_path)
    print(f"\n========== 训练完成！模型已保存至 {model_path}.zip ==========")

if __name__ == "__main__":
    train_dqn_cnn(total_timesteps=300000)