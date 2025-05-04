import gymnasium as gym
import numpy as np
from gymnasium import spaces


class VelocityCartPoleEnv(gym.Wrapper):
    """
    CartPole environment with velocity information hidden.
    The agent only observes position but must infer velocity.
    This creates a POMDP setting requiring memory.
    """
    
    def __init__(self):
        super().__init__(gym.make("CartPole-v1"))
        
        self.is_velocity_cartpole = True
        
        # only position observations (cart position and pole angle)
        # velocities are hidden
        self.observation_space = spaces.Box(
            low=np.array([-4.8, -0.418], dtype=np.float32),
            high=np.array([4.8, 0.418], dtype=np.float32),
            dtype=np.float32
        )
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # return only position components (hide velocities)
        partial_obs = np.array([obs[0], obs[2]], dtype=np.float32)
        return partial_obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # return only position components (hide velocities)
        partial_obs = np.array([obs[0], obs[2]], dtype=np.float32)
        return partial_obs, reward, terminated, truncated, info 