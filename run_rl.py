import os
import argparse
import time
import random
from v2 import UNORuleModelV2

import torch
import numpy as np
import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)
from collections import deque


class EpsGreedyDecay:
    """ Class to handle epsilon-greedy decay for exploration in DQN """
    def __init__(self, start_eps=1.0, end_eps=0.1, decay_episodes=1000):
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.decay_episodes = decay_episodes
        self.eps = start_eps

    def get_epsilon(self, episode):
        self.eps = max(self.end_eps, self.start_eps - (self.start_eps - self.end_eps) * episode / self.decay_episodes)
        return self.eps


def train(args):

    # Check whether GPU is available
    device = get_device()

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(
        args.env,
        config={
            'seed': args.seed,
        }
    )

    # Initialize epsilon-greedy decay strategy for DQN exploration
    if args.algorithm == 'dqn':
        eps_decay = EpsGreedyDecay(start_eps=args.start_eps, end_eps=args.end_eps, decay_episodes=args.decay_episodes)

    # Initialize the agent and use random agents as opponents
    if args.algorithm == 'dqn':
        from rlcard.agents import DQNAgent
        agent = DQNAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            mlp_layers=[128, 128],  # Increased layer size for better exploration capacity
            device=device,
        )
    elif args.algorithm == 'nfsp':
        from rlcard.agents import NFSPAgent
        agent = NFSPAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            hidden_layers_sizes=[128, 128],  # Increased layer size for better capacity
            q_mlp_layers=[128, 128],
            device=device,
        )

    # Load existing model if resume_training is True
    if args.resume_training and os.path.exists(args.model_path):
        agent = torch.load(args.model_path)
        print(f'Model loaded from {args.model_path}')

    agents = [agent]
    if args.train_against_self:
        for _ in range(1, env.num_players):
            agents.append(agent)
    else:
        for _ in range(1, env.num_players):
            # Train against a mix of random agents and previously trained versions of the main agent
            if random.random() < args.pretrained_agent_ratio:
                agents.append(torch.load(args.pretrained_model_path))  # Load less trained agent versions
            else:
                agents.append(UNORuleModelV2().agents[0])
                # agents.append(RandomAgent(num_actions=env.num_actions))

    # Shuffle agents to choose the starting agent randomly
    random.shuffle(agents)
    env.set_agents(agents)

    # Initialize replay memory for mini-batch training
    memory = deque(maxlen=args.memory_size)

    # Start training
    start_time = time.time()
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):

            print(" time : ", time.time() - start_time)
            if args.max_time and (time.time() - start_time) > args.max_time:
                break

            if args.algorithm == 'nfsp':
                agents[0].sample_episode_policy()

            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions into agent memory, and train the agent
            for ts in trajectories[0]:
                memory.append(ts)

            # If enough experiences, perform mini-batch updates
            if len(memory) >= args.batch_size:
                mini_batch = random.sample(memory, args.batch_size)
                for ts in mini_batch:
                    agent.feed(ts)

            # Dynamically adjust epsilon for exploration
            if args.algorithm == 'dqn':
                agent.epsilon = eps_decay.get_epsilon(episode)

            # Evaluate the performance. Play with random agents.
            if episode % args.evaluate_every == 0:
                logger.log_performance(
                    episode,
                    tournament(
                        env,
                        args.num_eval_games,
                    )[0]
                )

            # Save intermediate models every few episodes
            if episode % args.save_every == 0:
                save_path = os.path.join(args.log_dir, f'model_{episode}.pth')
                torch.save(agent, save_path)
                print(f'Model saved at episode {episode}')

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    plot_curve(csv_path, fig_path, args.algorithm)

    # Save final model
    save_path = os.path.join(args.log_dir, 'model.pth')
    torch.save(agent, save_path)
    print('Final model saved in', save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN/NFSP in RLCard")
    parser.add_argument(
        '--env',
        type=str,
        default='leduc-holdem',
        choices=[
            'blackjack',
            'leduc-holdem',
            'limit-holdem',
            'doudizhu',
            'mahjong',
            'no-limit-holdem',
            'uno',
            'gin-rummy',
            'bridge',
        ],
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        default='dqn',
        choices=[
            'dqn',
            'nfsp',
        ],
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=5000,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=2000,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/',
    )
    parser.add_argument(
        '--max_time',
        type=int,
        default=None,
        help='Maximum training time in seconds',
    )
    parser.add_argument(
        '--resume_training',
        action='store_true',
        help='Resume training from an existing model',
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='experiments/model.pth',
        help='Path to the existing model to resume training',
    )
    parser.add_argument(
        '--train_against_self',
        action='store_true',
        help='Train the agent against a copy of itself',
    )
    parser.add_argument(
        '--pretrained_agent_ratio',
        type=float,
        default=0.5,
        help='Probability of training against a pretrained agent',
    )
    parser.add_argument(
        '--pretrained_model_path',
        type=str,
        default='experiments/pretrained_model.pth',
        help='Path to the pretrained model for opponent agent',
    )
    parser.add_argument(
        '--start_eps',
        type=float,
        default=1.0,
        help='Starting epsilon value for exploration in DQN',
    )
    parser.add_argument(
        '--end_eps',
        type=float,
        default=0.1,
        help='Ending epsilon value for exploration in DQN',
    )
    parser.add_argument(
        '--decay_episodes',
        type=int,
        default=1000,
        help='Number of episodes to decay epsilon from start to end',
    )
    parser.add_argument(
        '--memory_size',
        type=int,
        default=10000,
        help='Replay memory size for experience replay',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training from replay memory',
    )
    parser.add_argument(
        '--save_every',
        type=int,
        default=500,
        help='Save model every N episodes',
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    from argparse import Namespace
    train(Namespace(env='uno',
                    algorithm='dqn',
                    cuda='',
                    seed=42,
                    num_episodes=20000,
                    num_eval_games=1000,
                    evaluate_every=100,
                    log_dir='experiments/uno_dqn2',
                    max_time=60,
                    resume_training=False,
                    model_path='experiments/uno_dqn2/model.pth',
                    train_against_self=False,
                    pretrained_agent_ratio=0.5,
                    pretrained_model_path='experiments/uno_dqn/model.pth',
                    start_eps=1.0,
                    end_eps=0.05,
                    decay_episodes=5000,
                    memory_size=50000,
                    batch_size=64,
                    save_every=1000))
