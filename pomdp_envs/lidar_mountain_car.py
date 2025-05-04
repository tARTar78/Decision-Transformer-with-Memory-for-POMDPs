import gymnasium as gym
import numpy as np
from gymnasium import spaces


class LiDARMountainCarEnv(gym.Wrapper):
    """
    MountainCar environment where the agent only observes simulated LiDAR readings
    instead of exact position and velocity. This creates a POMDP that requires
    memory to solve effectively.
    """
    
    def __init__(self, num_sensors=8):
        super().__init__(gym.make("MountainCar-v0"))
        self.num_sensors = num_sensors
        
        self.is_lidar_mountain_car = True
        
        # sensor angles (in radians, distributed around a circle)
        self.sensor_angles = np.linspace(0, 2*np.pi, num_sensors, endpoint=False)
        
        # LiDAR readings as observations
        self.observation_space = spaces.Box(
            low=0.0,
            high=2.0,  # maximum possible distance in the environment
            shape=(num_sensors,),
            dtype=np.float32
        )
    
    def _get_lidar_readings(self, state):
        """
        Convert the MountainCar state (position, velocity) into simulated LiDAR readings.
        """
        position, velocity = state
        
        # calculate distance to walls/mountains in each direction
        readings = np.zeros(self.num_sensors, dtype=np.float32)
        
        # mountain shape is approximately a cosine function
        # we'll use a simple model to calculate distances
        for i, angle in enumerate(self.sensor_angles):
            # direction vector
            dx = np.cos(angle)
            dy = np.sin(angle)
            
            # simple ray-casting calculation (approximate)
            # this is a simplified model of distance to the mountain surface
            if dy > 0:  # looking up
                # distance to the mountain surface (approximated as a sine wave)
                x = position
                while -1.2 <= x <= 0.6:
                    x += 0.01 * dx
                    y_surface = np.sin(3 * x) * 0.45 + 0.55
                    distance = np.sqrt((x - position)**2 + y_surface**2)
                    if distance > 0.01:
                        readings[i] = min(distance, 2.0)
                        break
            else:  # looking down or horizontally
                # distance to ground
                if abs(dx) < 1e-6:  # nearly vertical
                    readings[i] = 0.1  # close to ground
                else:
                    # simplified calculation
                    distance = min(2.0, 0.5 / abs(dy) if dy != 0 else 2.0)
                    readings[i] = distance
        
        return readings
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        lidar_readings = self._get_lidar_readings(obs)
        return lidar_readings, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        lidar_readings = self._get_lidar_readings(obs)
        
        # store the actual position in info for analysis
        info['actual_position'] = obs[0]
        info['actual_velocity'] = obs[1]
        
        return lidar_readings, reward, terminated, truncated, info 