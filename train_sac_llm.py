import os
import torch
import torch.nn as nn
import numpy as np
from tianshou.policy import BasePolicy
from tianshou.env import DummyVectorEnv
from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.policy import DiscreteSACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.discrete import Actor, Critic

from gym_wrapper import PensieveGymEnv

# ==========================================
# 1. 核心特征提取器 (1D-CNN) - 全 CPU 版
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
        self.output_dim = output_dim 

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(0)

        last_quality = obs[:, 0, -1].unsqueeze(1)    
        buffer_size = obs[:, 1, -1].unsqueeze(1)     
        throughput = obs[:, 2, :].unsqueeze(1)       
        delay = obs[:, 3, :].unsqueeze(1)            
        next_chunks = obs[:, 4, :6]                  
        remain_chunks = obs[:, 5, -1].unsqueeze(1)   
        
        conv_t = torch.relu(self.conv_throughput(throughput)) 
        conv_d = torch.relu(self.conv_delay(delay))           
        
        flat_t = conv_t.view(conv_t.size(0), -1)              
        flat_d = conv_d.view(conv_d.size(0), -1)              
        concat = torch.cat([last_quality, buffer_size, flat_t, flat_d, next_chunks, remain_chunks], dim=1) 
        
        logits = self.fc(concat)
        return logits, state


# ==========================================
# 2. 🌟 核心创新：模拟 LLM 专家的常识决策
# ==========================================
def simulated_llm_expert(obs_matrix):
    """
    用极简代码模拟大模型的常识推理逻辑：
    输入 6x8 状态矩阵，输出一个绝对安全的画质动作 (0-5)
    """
    # 提取物理特征
    buffer_size_sec = obs_matrix[1, -1] * 10.0 # 真实缓冲区剩余秒数
    throughputs = obs_matrix[2, :] * 1000.0 * 8 # 近似换算为网速 Mbps
    
    # 计算有效平均网速 (排除掉为0的初始填充项)
    valid_throughputs = throughputs[throughputs > 0]
    avg_throughput = np.mean(valid_throughputs) if len(valid_throughputs) > 0 else 1.0
    
    next_chunks_mb = obs_matrix[4, :6] # 下一个切片各档位大小 MB
    
    # 🔴 LLM 常识法则 1：濒临卡顿，绝对保命！
    # 如果水桶里只剩不到 4 秒的视频了，无脑选最低画质 (0档)
    if buffer_size_sec < 4.0:
        return 0 
        
    # 🟢 LLM 常识法则 2：安全情况下，贪心选择不超过带宽的最高画质
    # 从最高画质 (5档) 往下试，找到第一个能在缓冲区耗尽前下载完的画质
    for quality in range(5, -1, -1):
        # 预估下载耗时 = 文件大小(MB) * 8 / 网速(Mbps)
        estimated_download_time = (next_chunks_mb[quality] * 8.0) / avg_throughput
        
        # 如果下载耗时比缓冲区剩余时间少 1.5 秒 (留足安全余量)，就选它！
        if estimated_download_time < (buffer_size_sec - 1.5):
            return quality
            
    # 兜底：如果网速实在太差，全部都不满足，选最低画质
    return 0 

# ==========================================
# 3. 训练主逻辑 (经验注入 + 微调)
# ==========================================
def train_llm_augmented_sac():
    train_envs = DummyVectorEnv([lambda: PensieveGymEnv(random_seed=i) for i in range(4)])
    test_envs = DummyVectorEnv([lambda: PensieveGymEnv(random_seed=i+100) for i in range(1)])
    action_shape = 6 

    print("========== 正在初始化全 CPU 网络 ==========")
    net_actor = PensieveCNN(output_dim=128)
    net_c1 = PensieveCNN(output_dim=128)
    net_c2 = PensieveCNN(output_dim=128)

    actor = Actor(net_actor, action_shape, softmax_output=False)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic1 = Critic(net_c1, last_size=action_shape)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=1e-4)
    critic2 = Critic(net_c2, last_size=action_shape)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=1e-4)

    policy = DiscreteSACPolicy(
        actor=actor, actor_optim=actor_optim,
        critic1=critic1, critic1_optim=critic1_optim,
        critic2=critic2, critic2_optim=critic2_optim,
        tau=0.005, gamma=0.99, alpha=0.2, estimation_step=1
    )

    buffer = VectorReplayBuffer(100000, len(train_envs))
    
    # =======================================================
    # 🌟 专家经验注入阶段 (Offline Demonstration Injection)
    # 将 LLM 常识推理封装为原生 Policy，彻底免疫一切底层维度 Bug！
    # =======================================================
    print("========== 正在通过 LLM 原生策略高并发注入高质量专家经验... ==========")
    
    # 动态定义一个只做 LLM 推理的临时策略
    class LLMExpertPolicy(BasePolicy):
        def __init__(self):
            super().__init__()
            
        def forward(self, batch, state=None, **kwargs):
            # 批量解析 4 个环境的状态并调用 LLM 逻辑
            actions = [simulated_llm_expert(obs) for obs in batch.obs]
            # 👇 修复：Tianshou 要求所有返回数据必须打包在同一个 Batch 对象里！
            return Batch(act=np.array(actions), state=state)
            
        def learn(self, batch, **kwargs):
            return {}
    # 把我们的 LLM 专家实例化
    llm_policy = LLMExpertPolicy()
    
    # 用 Tianshou 官方的高性能 Collector 自动挂载运行，收集 3000 步！
    llm_collector = Collector(llm_policy, train_envs, buffer)
    llm_collector.collect(n_step=3000)
    
    print("========== 专家经验注入完毕！开始基于 SAC 的极速微调 ==========")
    
    # 无缝切换！使用注入了黄金经验的 Buffer 继续训练真实的 SAC
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)

    def save_best_fn(policy):
        os.makedirs("./models", exist_ok=True)
        torch.save(policy.state_dict(), "./models/llm_sac_pensieve.pth")

    # 关键点：因为有专家带着飞，我们只需要跑 20 个 epoch 就能极速收敛！
    result = offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=20, step_per_epoch=2000, step_per_collect=10,
        update_per_step=0.1, episode_per_test=10, batch_size=64,
        save_best_fn=save_best_fn
    )

    print(f"\n========== LLM 增强训练完成！模型已保存至 ./models/llm_sac_pensieve.pth ==========")

if __name__ == "__main__":
    train_llm_augmented_sac()