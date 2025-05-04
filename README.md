# Advanced Reinforcement Learning Course @ AI Masters
## Homework 6: Decision Transformer with Memory

### Introduction

Decision Transformer (DT) is a powerful algorithm for Offline Reinforcement Learning with significant flexibility for modifications. One common approach is adding memory mechanisms to enhance its capabilities (https://arxiv.org/abs/2306.09459, https://arxiv.org/abs/2305.16338, https://arxiv.org/abs/2410.07071, https://openreview.net/forum?id=NHMuM84tRT).

While DT can inherently handle Partially Observable Markov Decision Processes (POMDPs) through its attention mechanism, incorporating explicit memory structures can help process dependencies beyond the transformer's context window and improve overall performance metrics.

This assignment explores the implementation and evaluation of memory-enhanced Decision Transformers for POMDP control tasks.
> **Note**: Solution for GRU and LSTM will be added after the deadline.

### Setup

Before starting, install the required dependencies:

```bash
pip install torch numpy matplotlib gymnasium tqdm
```

### 1. POMDP Environments

We use classic MDP tasks from Gymnasium (CartPole, Pendulum, MountainCar) transformed into POMDPs using specialized wrappers that introduce partial observability:

- `velocity_cartpole.py`: CartPole with hidden velocity information
- `flickering_pendulum.py`: Pendulum with randomly missing observations
- `lidar_mountain_car.py`: MountainCar with only LiDAR-like distance readings

**Important Note**: The codebase is thoroughly tested only with `VelocityCartPoleEnv`. Experiments with other environments will require additional modifications:
- `FlickeringPendulumEnv`: You'll need to add support for continuous action spaces
- `LiDARMountainCarEnv`: You'll need to implement trajectory collection methods

This assignment focuses on `VelocityCartPoleEnv`. While a Decision Transformer with context length > 1 should be capable of solving this task, our goal is to explore how memory mechanisms can enhance performance.

### 2. Offline RL Dataset Collection

Since we're working in the Offline RL domain, we first need to collect training data:

```bash
python train_and_collect_data.py --env velocity_cartpole --train_timesteps 300000 --num_trajectories 100 --reward_threshold 475
```

This trains a PPO-GRU agent that will then be used to generate trajectories for our dataset.

After collecting the dataset, examine the performance of the PPO-GRU agent:

```bash
python utils/visualize_ppo_agent.py --env velocity_cartpole --model_path pomdp_datasets/velocity_cartpole/recurrent_ppo_velocity_cartpole.pt --rnn_type gru
```

And review dataset statistics:

```bash
python memory_dt.py --dataset pomdp_datasets/velocity_cartpole --stats_only
```

You should see output similar to:

```
Dataset statistics:
Total episodes: 100
Total steps: 49924
Mean reward per episode: 499.24
Median reward per episode: 500.00
Min/Max reward: 476.00/500.00
Reward std: 3.62
Mean episode length: 499.24
Reward percentiles: 
  10%: 500.00
  25%: 500.00
  50%: 500.00
  75%: 500.00
  90%: 500.00
  95%: 500.00
  99%: 500.00
```

The dataset shows high-quality trajectories with mean episode rewards close to the maximum possible for CartPole (500).

### 3. Decision Transformer with Memory

First, train and validate a standard Decision Transformer as a baseline:

```bash
# Train basic DT
python run_memory_dt.py --env velocity_cartpole --memory_type none --n_epochs 7 --eval_episodes 20

# Validate basic DT
python utils/visualize_dt_agent.py --env velocity_cartpole --model_path models/memory_dt_velocity_cartpole_None_best.pt --memory_type none 
```

Record these results for later comparison with your memory-enhanced implementations.

### Tasks

#### Task 1: Implementing GRU and LSTM Memory (0.2 points)

Recurrent Neural Networks like Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU) are natural choices for implementing memory in neural networks. While the Decision Transformer operates only on information visible within its context window, adding RNN layers enables maintaining internal hidden states that "remember" past observations.

Complete the code in `memory_dt.py` to implement GRU and LSTM memory:

```python
# file: memory_dt.py
self.pos_encoder = PositionalEncoding(n_embed)

# Memory
if memory_type == 'gru':
   # TODO: Implement GRU memory
   self.memory_proj = nn.Linear(memory_dim, n_embed)
elif memory_type == 'lstm':
   # TODO: Implement LSTM memory
   self.memory_proj = nn.Linear(memory_dim, n_embed)
else:
   self.memory = None

# Transformer
```

```python
# file: memory_dt.py
# add memory
if self.memory is not None:
   if self.memory_type == 'gru':
         if self.hidden_state is None:
            # TODO: Implement GRU memory
         
         memory_out, self.hidden_state = self.memory(state_embeddings, self.hidden_state)
   elif self.memory_type == 'lstm':
         if self.hidden_state is None:
            # TODO: Implement LSTM memory
         
         memory_out, self.hidden_state = self.memory(state_embeddings, self.hidden_state)
   
   # project memory to embedding dimension
   memory_embedding = self.memory_proj(memory_out)
```

After implementing the memory modules, train your enhanced models:

```bash
# Train DT+GRU (0.1 points)
python run_memory_dt.py --env velocity_cartpole --memory_type gru --n_epochs 7 --eval_episodes 20

# Train DT+LSTM (0.1 points)
python run_memory_dt.py --env velocity_cartpole --memory_type lstm --n_epochs 7 --eval_episodes 20
```

Then validate their performance:

```bash
# Validate DT+GRU
python utils/visualize_dt_agent.py --env velocity_cartpole --model_path models/memory_dt_velocity_cartpole_gru_best.pt --memory_type gru 

# Validate DT+LSTM
python utils/visualize_dt_agent.py --env velocity_cartpole --model_path models/memory_dt_velocity_cartpole_lstm_best.pt --memory_type lstm 
```

#### Task 2: Comparative Analysis (0.3 points)

Compare the performance of standard DT versus DT+GRU and DT+LSTM implementations. Analyze:

- Training efficiency (convergence speed)
- Final performance metrics
- Data requirements for convergence
- Overall impact of adding memory modules

Conduct experiments with different model parameters (context length, training set size, data quality, etc.) and compare results. What happens when return-to-go (RTG) changes?

Include training graphs and performance metrics to support your analysis.

#### Task 3: Custom Memory Mechanism (0.3 points)

Design and implement your own memory mechanism for the Decision Transformer. There is no single correct approach – be creative within reasonable bounds.

Document your approach:
- What information will your mechanism process?
- How will it be processed?
- Which model architecture will you use?
- What training approach will you take?

Conduct experiments and compare your custom memory implementation with the previous approaches. What conclusions can you draw?

### Submission Guidelines (0.2 points)

1. Fork or clone this repository and make your changes
2. Add your report with experimental results and analysis in any convenient format (.ipynb, .pdf, etc.) to the github repository 
3. Submit a file containing the link to your GitHub repository to the submission telegram bot

A well-structured, professionally formatted report will earn 0.2 points.

### Repository Structure

- `pomdp_envs/velocity_cartpole.py`: Implementation of the velocity-hidden CartPole environment
- `pomdp_envs/flickering_pendulum.py`: Implementation of the flickering Pendulum environment
- `pomdp_envs/lidar_mountain_car.py`: Implementation of the LiDAR MountainCar environment
- `recurrent_ppo.py`: Recurrent PPO implementation for collecting training data
- `train_and_collect_data.py`: Script to train the recurrent PPO agent and collect trajectories
- `memory_dt.py`: Decision Transformer (with memory) implementation
- `run_memory_dt.py`: Script to train and evaluate the Decision Transformer (with memory) 