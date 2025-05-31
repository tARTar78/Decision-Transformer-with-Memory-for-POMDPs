import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import gymnasium as gym

from pomdp_envs.velocity_cartpole import VelocityCartPoleEnv
from pomdp_envs.flickering_pendulum import FlickeringPendulumEnv
from pomdp_envs.lidar_mountain_car import LiDARMountainCarEnv
from memory_dt import train_memory_dt, evaluate_memory_dt, MemoryDecisionTransformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='velocity_cartpole',
                        choices=['velocity_cartpole', 'flickering_pendulum', 'lidar_mountain_car'],
                        help='POMDP environment to use')
    parser.add_argument('--data_dir', type=str, default='pomdp_datasets',
                        help='Directory containing collected trajectories')
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--context_length', type=int, default=20,
                        help='Context length for the transformer')
    parser.add_argument('--n_embed', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--n_layer', type=int, default=3,
                        help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--memory_type', type=str, default='gru',
                        choices=['gru', 'lstm','attention', 'none'],
                        help='Type of memory to use (gru, lstm, or none)')
    parser.add_argument('--memory_dim', type=int, default=64,
                        help='Dimension of memory state')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--eval_episodes', type=int, default=10,
                        help='Number of episodes to evaluate')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during evaluation')
    parser.add_argument('--target_return', type=float, default=None,
                        help='Target return for cartpole (adjust for each env)')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to pre-trained model to load (skip training)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')
    return parser.parse_args()


def create_env(env_name):
    if env_name == 'velocity_cartpole':
        return VelocityCartPoleEnv()
    elif env_name == 'flickering_pendulum':
        return FlickeringPendulumEnv(flicker_probability=0.3)
    elif env_name == 'lidar_mountain_car':
        return LiDARMountainCarEnv(num_sensors=8)
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def load_model(model_path, env, args):
    """Load a pre-trained model and create the model instance."""
    if isinstance(env.observation_space, gym.spaces.Box):
        state_dim = env.observation_space.shape[0]
    elif isinstance(env.observation_space, gym.spaces.Dict):
        if 'observation' in env.observation_space.spaces:
            state_dim = env.observation_space.spaces['observation'].shape[0]
            if 'mask' in env.observation_space.spaces:
                state_dim += 1
        else:
            state_dim = sum(space.shape[0] if hasattr(space, 'shape') else 1
                          for space in env.observation_space.spaces.values())
    else:
        raise ValueError(f"Unsupported observation space: {env.observation_space}")
    
    n_actions = env.action_space.n
    
    model = MemoryDecisionTransformer(
        state_dim=state_dim,
        n_actions=n_actions,
        n_embed=args.n_embed,
        n_layer=args.n_layer,
        n_head=args.n_head,
        context_length=args.context_length,
        memory_type=args.memory_type if args.memory_type != 'none' else None,
        memory_dim=args.memory_dim,
        debug=args.debug
    )
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    return model


def main():
    args = parse_args()
    
    if args.memory_type == 'none':
        args.memory_type = None
    
    dataset_path = os.path.join(args.data_dir, args.env)
    
    env = create_env(args.env)
    
    if args.env == 'velocity_cartpole':
        target_return = 500.0
    elif args.env == 'flickering_pendulum':
        target_return = -200.0  # the goal is to minimize loss in pendulum
    elif args.env == 'lidar_mountain_car':
        target_return = -100.0  # the goal is to reach the flag with minimum steps
    else:
        target_return = args.target_return if args.target_return is not None else 500.0
    
    if args.load_model:
        print(f"Loading pre-trained model from {args.load_model}")
        model = load_model(args.load_model, env, args)
        train_losses = []
    else:
        print(f"Training Memory Decision Transformer on {args.env}...")
        model, train_losses, val_returns = train_memory_dt(
            env_name=args.env,
            dataset_path=dataset_path,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            context_length=args.context_length,
            n_embed=args.n_embed,
            n_layer=args.n_layer,
            n_head=args.n_head,
            memory_type=args.memory_type,
            memory_dim=args.memory_dim,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            debug=args.debug
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Evaluating model on {args.env}...")
    mean_return, returns, success_rate = evaluate_memory_dt(
        model=model,
        env=env,
        num_episodes=args.eval_episodes,
        render=args.render,
        target_return=target_return,
        context_length=args.context_length,
        debug=args.debug
    )
    
    print(f"Evaluation complete. Mean return: {mean_return:.2f}, Success rate: {success_rate:.2%}")


if __name__ == "__main__":
    main() 