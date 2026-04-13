import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve_zoomed(log_dir, window_size=50, y_lower_limit=-500, y_upper_limit=100):
    """
    绘制平滑曲线，并截断极端负向 Reward 以放大收敛细节
    """
    log_file = None
    for file in os.listdir(log_dir):
        if file.endswith("monitor.csv"):
            log_file = os.path.join(log_dir, file)
            break
            
    if log_file is None:
        raise FileNotFoundError(f"在 {log_dir} 目录下没有找到 monitor.csv")

    df = pd.read_csv(log_file, skiprows=1)
    
    # 我们只取前 2500 个 Episode 来画图（因为 10万步大概对应 2000 多个视频，后面都是平台期了）
    # 这样 X 轴也不会被拉得太长
    max_episodes = 2500
    rewards = df['r'].values[:max_episodes]
    episodes = np.arange(len(rewards))
    
    # 计算滑动平均
    smoothed_rewards = pd.Series(rewards).rolling(window=window_size, min_periods=1).mean()

    plt.figure(figsize=(10, 6))
    
    # 画底部的半透明原始数据
    plt.plot(episodes, rewards, alpha=0.15, color='tab:blue', label='Raw Reward')
    
    # 画上层的平滑主干曲线
    plt.plot(episodes, smoothed_rewards, linewidth=2.5, color='#E15759', label=f'Smoothed (Window={window_size})')

    plt.title('Pure DQN Baseline', fontsize=14, fontweight='bold')
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Total Reward (QoE Score)', fontsize=12)
    
    # ==========================================
    # 🌟 核心改进：限制 Y 轴范围，切掉 -10000 的毛刺
    # ==========================================
    plt.ylim(y_lower_limit, y_upper_limit)
    
    # 画一条 y=0 的参考线
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.6, linewidth=1.5, label='Zero QoE Baseline')

    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    save_path = 'pure_dqn_zoomed_curve.png'
    plt.savefig(save_path, dpi=300)
    print(f"放大的曲线图已生成并保存为: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    log_directory = "./logs/pure_DQN/"
    # 这里将Y轴限制在 [-500, 100] 的区间内，可以根据出图效果微调这两个数字
    plot_learning_curve_zoomed(log_directory, window_size=100, y_lower_limit=-1500, y_upper_limit=100)