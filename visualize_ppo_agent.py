import os
import argparse
import torch
import numpy as np
import gymnasium as gym
import time
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from recurrent_ppo import RecurrentActorCritic, RecurrentPPO
from pomdp_envs.velocity_cartpole import VelocityCartPoleEnv
from pomdp_envs.flickering_pendulum import FlickeringPendulumEnv
from pomdp_envs.lidar_mountain_car import LiDARMountainCarEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='velocity_cartpole',
                        choices=['velocity_cartpole', 'flickering_pendulum', 'lidar_mountain_car'],
                        help='POMDP environment to use')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained PPO model')
    parser.add_argument('--num_episodes', type=int, default=5,
                        help='Number of episodes to visualize')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension used in the model')
    parser.add_argument('--rnn_type', type=str, default='gru', choices=['gru', 'lstm','attention'],
                        help='Type of RNN used in the model')
    parser.add_argument('--delay', type=float, default=0.05,
                        help='Delay between steps for better visualization')
    parser.add_argument('--record', action='store_true',
                        help='Record video of the agent performance')
    return parser.parse_args()


def create_env(env_name, render_mode='human'):
    """Create environment with the specified render mode."""
    if env_name == 'velocity_cartpole':
        # Base env (it's MDP, we then make it POMDP via wrappers from pomdp_envs/)
        base_env = gym.make("CartPole-v1", render_mode=render_mode)

        # Wrap it with our VelocityCartPoleEnv but keep the render functionality
        env = VelocityCartPoleEnv.__new__(VelocityCartPoleEnv)
        VelocityCartPoleEnv.__init__(env)
        env.env = base_env
        return env
    elif env_name == 'flickering_pendulum':
        # Base env (it's MDP, we then make it POMDP via wrappers from pomdp_envs/)
        base_env = gym.make("Pendulum-v1", render_mode=render_mode)

        # Wrap it with our FlickeringPendulumEnv
        env = FlickeringPendulumEnv.__new__(FlickeringPendulumEnv)
        FlickeringPendulumEnv.__init__(env, flicker_probability=0.3)
        env.env = base_env
        return env
    elif env_name == 'lidar_mountain_car':
        # Base env (it's MDP, we then make it POMDP via wrappers from pomdp_envs/)
        base_env = gym.make("MountainCar-v0", render_mode=render_mode)

        # Wrap it with our LiDARMountainCarEnv
        env = LiDARMountainCarEnv.__new__(LiDARMountainCarEnv)
        LiDARMountainCarEnv.__init__(env, num_sensors=8)
        env.env = base_env
        return env
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def load_model(model_path, env, hidden_dim, rnn_type):
    """Load a trained PPO policy model."""
    # observation dimension
    if isinstance(env.observation_space, gym.spaces.Box):
        obs_dim = env.observation_space.shape[0]
    elif isinstance(env.observation_space, gym.spaces.Dict):
        # flatten all the values in the dictionary (for FlickeringPendulum)
        if 'observation' in env.observation_space.spaces:
            obs_dim = env.observation_space.spaces['observation'].shape[0]
            if 'mask' in env.observation_space.spaces:
                obs_dim += 1
        else:
            # just sum all dimensions
            obs_dim = sum(
                space.shape[0] if hasattr(space, 'shape') else 1 
                         for space in env.observation_space.spaces.values()
                )
    else:
        raise ValueError(f"Unsupported observation space: {env.observation_space}")
    
    # action dimension
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
    else:
        raise ValueError("Only discrete action spaces supported for now. But if you want to use continuous action spaces, you can do it by yourself =)")
    
    # load model to determine hidden_dim
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    
    # determine hidden_dim from the dimension of the model weights
    feature_weight_shape = state_dict['feature_extractor.0.weight'].shape
    actual_hidden_dim = feature_weight_shape[0]
    
    if actual_hidden_dim != hidden_dim:
        print(f"WARNING: Determined hidden_dim={actual_hidden_dim} from model, overriding argument hidden_dim={hidden_dim}")
        hidden_dim = actual_hidden_dim
    
    # policy network
    policy = RecurrentActorCritic(obs_dim, hidden_dim, action_dim, rnn_type)
    
    # load saved checkpoint
    policy.load_state_dict(state_dict)
    
    return policy


def evaluate_agent(env, policy, num_episodes, delay=0.05):
    """Evaluate and visualize the agent."""
    device = torch.device("cpu") # viz on cpu
    policy.to(device)
    policy.eval()
    
    # observation preprocessing function, similar to that in RecurrentPPO
    def preprocess_obs(obs):
        if isinstance(obs, dict):
            if 'observation' in obs and 'mask' in obs:
                obs_tensor = np.concatenate([
                    obs['observation'].flatten(),
                    np.array([obs['mask']], dtype=np.float32)
                ])
                
                is_visible = obs['mask'] == 1
                if not is_visible:
                    print("Observation HIDDEN (flickering)")
            else:
                obs_tensor = np.concatenate(
                    [o.flatten() if hasattr(o, 'flatten') else np.array([o])
                        for o in obs.values()]
                )
        else:
            obs_tensor = obs
        
            if len(obs_tensor) == 4:
                obs_tensor[0] = obs_tensor[0] / 3.0
                obs_tensor[1] = np.clip(obs_tensor[1] / 5.0, -1.0, 1.0)
                angle = obs_tensor[2]
                obs_tensor[3] = np.clip(obs_tensor[3] / 5.0, -1.0, 1.0)
                sin_theta = np.sin(angle)
                cos_theta = np.cos(angle)

                obs_tensor = np.array([
                    obs_tensor[0],    # normalized position
                    obs_tensor[1],    # normalized velocity
                    sin_theta,        # sin of angle
                    cos_theta,        # cos of angle
                    obs_tensor[3]     # normalized angular velocity
                ], dtype=np.float32)
        
        return obs_tensor
    
    total_rewards = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        hidden_state = policy.get_init_hidden()
        hidden_state = tuple(h.to(device) for h in hidden_state) if isinstance(hidden_state, tuple) else hidden_state.to(device)
        
        done = False
        episode_reward = 0
        timestep = 0
        
        print(f"\nEpisode {episode+1}/{num_episodes}")
        print("=" * 40)
        
        while not done:
            obs_tensor = preprocess_obs(obs)
            
            state_tensor = torch.FloatTensor(obs_tensor).to(device)
            
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            # sample action from policy
            with torch.no_grad():
                action_probs, value, hidden_state = policy(state_tensor, hidden_state)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
            
            print(f"Step {timestep+1}: Action={action.item()}, Value={value.item():.3f}")
            
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            obs = next_obs
            episode_reward += reward
            timestep += 1
            
            time.sleep(delay)
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode+1} finished with reward: {episode_reward}")
    
    print("\nEvaluation complete!")
    print(f"Average reward over {num_episodes} episodes: {np.mean(total_rewards):.2f}")
    
    return total_rewards


def main():
    args = parse_args()
    
    env = create_env(args.env, render_mode='human')
    
    policy = load_model(args.model_path, env, args.hidden_dim, args.rnn_type)
    
    print(f"Visualization of {args.env} from {args.model_path}")
    print(f"RNN type: {args.rnn_type}, Hidden dim: {policy.hidden_dim}")
    
    evaluate_agent(env, policy, args.num_episodes, args.delay)
    
    env.close()


if __name__ == "__main__":
    main() 