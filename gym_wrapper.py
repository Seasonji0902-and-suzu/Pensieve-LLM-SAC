import gymnasium as gym
from gymnasium import spaces
import numpy as np

# 确保你的文件夹里有 env.py 和 core.py (即 fixed_env.py) 以及 load_trace.py
from env import ABREnv 

class PensieveGymEnv(gym.Env):
    """
    为 Pensieve 原生环境套上的标准的 Gymnasium 外壳，
    使其能够被 Stable Baselines3 完美识别和调用。
    """
    def __init__(self, random_seed=42):
        super(PensieveGymEnv, self).__init__()
        
        # 1. 初始化底层的 Pensieve 环境
        self.abr_env = ABREnv(random_seed=random_seed)
        
        # 2. 定义动作空间 (Action Space)
        self.action_space = spaces.Discrete(6)
        
        # 3. 定义状态空间 (Observation Space)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(6, 8), 
            dtype=np.float32
        )

    # 【修复1】：增加 seed 和 options 参数，并返回 (state, info)
    def reset(self, seed=None, options=None):
        """重置环境，返回初始状态和空字典 info"""
        # 继承父类的 seed 处理
        super().reset(seed=seed)
        if seed is not None:
            self.abr_env.seed(seed)
            
        # 原版的 reset() 直接返回 state
        state = self.abr_env.reset()
        
        # Gymnasium 要求返回 (observation, info)
        return state, {}

    # 【修复2】：返回 5 元组 (state, reward, terminated, truncated, info)
    def step(self, action):
        """执行动作，返回下一步状态、奖励、是否结束以及额外信息"""
        state, reward, done, info = self.abr_env.step(action)
        
        # Gymnasium 要求将 done 拆分为 terminated (任务正常结束) 和 truncated (超时截断)
        # 视频播完属于正常的 terminated，我们没有设置外部超时时间，所以 truncated 恒为 False
        return state, reward, done, False, info

    def render(self, mode='human'):
        """可视化接口（保留为空）"""
        pass