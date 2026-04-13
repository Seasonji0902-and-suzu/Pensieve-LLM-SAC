import os
import numpy as np
from stable_baselines3 import DQN
import load_trace

# ==========================================
# 🌟 关键：将数据源切换为测试集文件夹
# 假设你的测试文件放在 ./test/ 目录下
# ==========================================
TEST_TRACE_DIR = './test/' 
# 动态修改 load_trace 的默认路径，欺骗底层环境读取测试集
load_trace.COOKED_TRACE_FOLDER = TEST_TRACE_DIR 

from gym_wrapper import PensieveGymEnv

def evaluate_model(model_path="./models/pure_dqn_pensieve", log_dir="./test_results/"):
    os.makedirs(log_dir, exist_ok=True)
    
    # 1. 初始化测试环境
    env = PensieveGymEnv(random_seed=42)
    # 获取测试集中 trace 文件的总数
    num_test_traces = len(env.abr_env.net_env.all_cooked_time)
    
    # 2. 加载训练好的模型
    print(f"正在加载模型: {model_path}.zip")
    model = DQN.load(model_path)
    
    print(f"========== 开始测试，共需跑 {num_test_traces} 个测试文件 ==========")
    
    # 3. 遍历每一个测试网络文件
    for trace_idx in range(num_test_traces):
        # 重置环境，env 会按顺序或根据你的修改加载下一个 trace
        obs, info = env.reset()
        done = False
        
        # 为了让 plot.py 能识别，文件名前缀必须包含 'rl'（对应 Pensieve 的默认标识）
        log_file_path = os.path.join(log_dir, f'rl_test_trace_{trace_idx}.txt')
        
        with open(log_file_path, 'w') as f:
            while not done:
                # 🌟 核心：deterministic=True 意味着关闭 epsilon-greedy
                # 让模型完全凭借学到的 Q 值网络做最优决策，绝不盲目试错
                action, _states = model.predict(obs, deterministic=True)
                
                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # 提取底层物理数据，拼装成 plot.py 需要的格式
                # plot.py 关键索引：[0]时间, [1]码率(kbps), [3]卡顿(s), [-1]奖励
                time_stamp = env.abr_env.time_stamp / 1000.0  # 毫秒转秒
                bitrate_kbps = info['bitrate']
                buffer_size = env.abr_env.buffer_size / 1000.0 # 毫秒转秒
                rebuffer_sec = info['rebuffer']
                
                # 格式化写入日志 (为了兼容官方格式，中间补上占位的 chunk_size 和 delay)
                # 格式: 时间 码率 缓冲 卡顿 空位 空位 奖励
                log_line = f"{time_stamp:.3f}\t{bitrate_kbps:.1f}\t{buffer_size:.3f}\t{rebuffer_sec:.3f}\t0\t0\t{reward:.4f}\n"
                f.write(log_line)
                
        print(f"完成测试进度: {trace_idx + 1}/{num_test_traces}")

    print(f"========== 所有测试完成！日志已保存至 {log_dir} ==========")
    print("现在你可以运行 plot.py 来生成对比图了。")

if __name__ == "__main__":
    evaluate_model()