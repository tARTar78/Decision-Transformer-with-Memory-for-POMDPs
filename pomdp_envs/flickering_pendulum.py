import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FlickeringPendulumEnv(gym.Wrapper):
    """
    Pendulum environment where observations randomly flicker (become unobservable).
    This creates a POMDP that requires memory to solve effectively.
    """
    
    def __init__(self, flicker_probability=0.3):
        super().__init__(gym.make("Pendulum-v1"))
        self.flicker_probability = flicker_probability
        
        self.is_flickering_pendulum = True
        
        # keep the same observation space as the pendulum
        # but we'll occasionally return zeros instead
        
        # add a binary flag to indicate if observation is visible
        self.observation_space = spaces.Dict({
            'observation': self.env.observation_space,
            'mask': spaces.Discrete(2),  # 0: hidden, 1: visible
        })
        
        self.last_obs = None
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs  # store the observation
        
        # always return the first observation (no flicker on reset)
        return {'observation': obs, 'mask': 1}, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # check if observation should flicker
        is_visible = np.random.random() > self.flicker_probability
        
        if is_visible:
            self.last_obs = obs  # update last observation if visible
            return {'observation': obs, 'mask': 1}, reward, terminated, truncated, info
        else:
            # return zeros for the observation with mask=0
            return {'observation': np.zeros_like(obs), 'mask': 0}, reward, terminated, truncated, info 