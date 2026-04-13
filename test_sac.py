import os
import torch
import numpy as np
from tianshou.data import Batch, to_torch
from tianshou.policy import DiscreteSACPolicy
from tianshou.utils.net.discrete import Actor, Critic

# 导入你的 CNN 和 环境
from train_sac_cnn import PensieveCNN
from gym_wrapper import PensieveGymEnv
import load_trace

# 设置测试集路径
TEST_TRACE_DIR = './test/'
load_trace.COOKED_TRACE_FOLDER = TEST_TRACE_DIR

def evaluate_sac(model_path="./models/llm_sac_pensieve.pth", log_dir="./test_results/"):
    os.makedirs(log_dir, exist_ok=True)
    
    # 1. 构造完全相同的网络结构
    action_shape = 6
    net_actor = PensieveCNN(output_dim=128)
    actor = Actor(net_actor, action_shape, softmax_output=False)
    
    # 这里的 Critic 只是为了初始化 Policy 对象，不参与测试推理
    net_c1 = PensieveCNN(output_dim=128)
    critic1 = Critic(net_c1, last_size=action_shape)
    net_c2 = PensieveCNN(output_dim=128)
    critic2 = Critic(net_c2, last_size=action_shape)

    policy = DiscreteSACPolicy(
        actor=actor, actor_optim=None, # 测试不需要优化器
        critic1=critic1, critic1_optim=None,
        critic2=critic2, critic2_optim=None
    )

    # 2. 加载权重 (全 CPU 模式)
    print(f"正在加载 SAC 模型权重: {model_path}")
    policy.load_state_dict(torch.load(model_path, map_location="cpu"))
    policy.eval()

    # 3. 初始化环境
    env = PensieveGymEnv(random_seed=42)
    num_test_traces = len(env.abr_env.net_env.all_cooked_time)

    print(f"========== 开始 SAC 模型测试，共 {num_test_traces} 个轨迹 ==========")

    for trace_idx in range(num_test_traces):
        obs, info = env.reset()
        done = False
        
        log_file_path = os.path.join(log_dir, f'llm_sac_trace_{trace_idx}.txt')
        
        with open(log_file_path, 'w') as f:
            while not done:
                # 使用 policy 直接推理，不加噪声
                batch = Batch(obs=[obs], info={})
                result = policy(batch)
                # 选择概率最大的动作
                action = result.act[0]
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # 记录符合 plot.py 格式的日志
                time_stamp = env.abr_env.time_stamp / 1000.0
                bitrate_kbps = info['bitrate']
                buffer_size = env.abr_env.buffer_size / 1000.0
                rebuffer_sec = info['rebuffer']
                
                log_line = f"{time_stamp:.3f}\t{bitrate_kbps:.1f}\t{buffer_size:.3f}\t{rebuffer_sec:.3f}\t0\t0\t{reward:.4f}\n"
                f.write(log_line)
        
        print(f"进度: {trace_idx + 1}/{num_test_traces}")

    print("========== 测试完成！请立即运行 plot.py 生成对比图 ==========")

if __name__ == "__main__":
    evaluate_sac()