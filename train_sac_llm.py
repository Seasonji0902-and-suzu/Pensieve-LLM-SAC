import os
import torch
import torch.nn as nn
import numpy as np
from tianshou.env import DummyVectorEnv
from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.policy import DiscreteSACPolicy, BasePolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.discrete import Actor, Critic

# 导入环境和底层 trace 加载模块
import load_trace
from gym_wrapper import PensieveGymEnv

# ========================================================
# 🌟 关键点 1：指向正确的训练集！
# ========================================================
load_trace.COOKED_TRACE_FOLDER = './train/' 

# ==========================================
# 🌟 关键点 2：核心特征提取器 (1D-CNN) - 全 CPU 极简版
# ==========================================
class PensieveCNN(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        self.conv_throughput = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=4)
        self.conv_delay = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=4)
        self.fc = nn.Sequential(nn.Linear(1289, 128), nn.ReLU(), nn.Linear(128, output_dim), nn.ReLU())
        self.output_dim = output_dim 

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor): obs = torch.tensor(obs, dtype=torch.float32)
        if len(obs.shape) == 2: obs = obs.unsqueeze(0)

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
        return self.fc(concat), state

# ========================================================
# 🌟 关键点 3：悲观的 LLM 专家 (Pessimistic Expert)
# 彻底抛弃均值陷阱，采用 20% 极差网速分位数作为常识底线
# ========================================================
def simulated_pessimistic_llm(obs_matrix):
    buffer_size_sec = obs_matrix[1, -1] * 10.0
    throughputs = obs_matrix[2, :] * 1000.0 * 8 
    valid_throughputs = throughputs[throughputs > 0]
    
    # 核心：取历史网速的 20% 分位数，并乘以 0.8 的安全系数！极度保守！
    if len(valid_throughputs) > 0:
        safe_throughput = np.percentile(valid_throughputs, 30) * 0.95 
    else:
        safe_throughput = 1.0
        
    next_chunks_mb = obs_matrix[4, :6]
    
    # 水桶少于4秒，绝对保命
    if buffer_size_sec < 4.0: return 0 
        
    # 基于极其悲观的 safe_throughput 预估下载时间
    for quality in range(5, -1, -1):
        estimated_time = (next_chunks_mb[quality] * 8.0) / safe_throughput
        if estimated_time < (buffer_size_sec - 1.5):
            return quality
    return 0 

# 原生策略包装器 (彻底解决维度报错)
class LLMExpertPolicy(BasePolicy):
    def __init__(self): super().__init__()
    def forward(self, batch, state=None, **kwargs):
        actions = [simulated_pessimistic_llm(obs) for obs in batch.obs]
        return Batch(act=np.array(actions), state=state)
    def learn(self, batch, **kwargs): return {}

# ==========================================
# 🌟 关键点 4：训练主逻辑 (高并发经验注入 + 极速微调)
# ==========================================
def train_llm_augmented_sac():
    # 此时环境底层已经是真随机了！
    train_envs = DummyVectorEnv([lambda: PensieveGymEnv(random_seed=i) for i in range(4)])
    test_envs = DummyVectorEnv([lambda: PensieveGymEnv(random_seed=i+100) for i in range(1)])
    action_shape = 6 

    print("========== 正在初始化鲁棒版 LLM-SAC 网络 ==========")
    actor = Actor(PensieveCNN(128), action_shape, softmax_output=False)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic1 = Critic(PensieveCNN(128), last_size=action_shape)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=1e-4)
    critic2 = Critic(PensieveCNN(128), last_size=action_shape)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=1e-4)

    policy = DiscreteSACPolicy(
        actor=actor, actor_optim=actor_optim, critic1=critic1, critic1_optim=critic1_optim,
        critic2=critic2, critic2_optim=critic2_optim, tau=0.005, gamma=0.99, alpha=0.2, estimation_step=1
    )

    buffer = VectorReplayBuffer(100000, len(train_envs))
    
    print("========== 正在高并发注入【悲观专家】的高质量保命经验... ==========")
    llm_policy = LLMExpertPolicy()
    llm_collector = Collector(llm_policy, train_envs, buffer)
    llm_collector.collect(n_step=3000)
    
    print("========== 专家经验注入完毕！开始基于真实随机环境的 SAC 微调 ==========")
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)

    def save_best_fn(policy):
        os.makedirs("./models", exist_ok=True)
        torch.save(policy.state_dict(), "./models/llm_sac_pensieve.pth")

    # 跑 20 个 Epoch 极速见效
    result = offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=40, step_per_epoch=4000, step_per_collect=10,
        update_per_step=0.1, episode_per_test=10, batch_size=64,
        save_best_fn=save_best_fn
    )

    print(f"\n========== 终极训练完成！模型已保存，准备迎接胜利！ ==========")

if __name__ == "__main__":
    train_llm_augmented_sac()