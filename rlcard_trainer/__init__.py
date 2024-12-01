import os
import random
from pathlib import Path

import torch

import rlcard
from rlcard.utils import (
    set_seed,
    tournament,
    reorganize,
    Logger,
)
from .utils import (
    load_model,
    get_device,
    plot_curve
)
from .utils.type_checker import type_check


@type_check
def train(seed: str | int | float, env: str, algorithm: str, num_episodes: int, num_eval_games: int, evaluate_every: int, log_dir:  Path | str, learning_rate: float, resume_training: bool, cuda: str, *args, **kwargs):

    # Check whether gpu is available
    device = get_device()

    # Seed numpy, torch, random
    set_seed(seed)

    # Make the environment with seed
    env = rlcard.make(
        env,
        config={
            'seed': seed,
        }
    )

    # Charger un modèle existant si possible
    if resume_training and os.path.exists(model_path := os.path.join(log_dir, 'model.pth')):
        agent = load_model(model_path=model_path, device=device, env=env)
    else:
        print(f"/!\\ No model existing at : {model_path}")
        print("Start training a new model ...")
        from time import sleep
        sleep(3)

        # Initialize the agent and use random agents as opponents
        match algorithm:
            case 'dqn':
                from rlcard.agents import DQNAgent
                agent = DQNAgent(
                    num_actions=env.num_actions,
                    state_shape=env.state_shape[0],
                    mlp_layers=[64, 64, 64, 64],
                    device=torch.device(device),
                    learning_rate=learning_rate,
                )
            case 'nfsp':
                from rlcard.agents import NFSPAgent
                agent = NFSPAgent(
                    num_actions=env.num_actions,
                    state_shape=env.state_shape[0],
                    hidden_layers_sizes=[64,64],
                    q_mlp_layers=[64,64],
                    device=torch.device(device),
                )
            case _:
                raise ValueError(f"Algorithm {algorithm} not supported")

    agents = [agent]
    if env.name == 'uno':
        from rlcard.models.uno_rule_models import UNORuleAgentV1
        from .rule_agents import UNORuleAgentV2, UNORuleAgentV4
        from rlcard.agents import RandomAgent
        ALL_MODELS = {
            'rd': RandomAgent,
            'v1': UNORuleAgentV1,
            'v2': UNORuleAgentV2,
            'v4': UNORuleAgentV4,
        }
        for _ in range(1, env.num_players):
            if (agent_model := random.choice(list(ALL_MODELS.keys()))) not in ('rd'):
                agents.append(ALL_MODELS[agent_model]())
            else:
                agents.append(RandomAgent(num_actions=env.num_actions))
    else:
        for _ in range(1, env.num_players):
            from rlcard.agents import RandomAgent
            agents.append(RandomAgent(num_actions=env.num_actions))

    # Set agents in the environment
    env.set_agents(agents)

    # Start training
    with Logger(log_dir) as logger:
        for episode in range(num_episodes):

            if algorithm == 'nfsp':
                agents[0].sample_episode_policy()

            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions into agent memory, and train the agent
            # Here, we assume that DQN always plays the first position
            # and the other players play randomly (if any)
            for ts in trajectories[0]:
                agent.feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % evaluate_every == 0:
                logger.log_performance(
                    episode,
                    tournament(
                        env,
                        num_eval_games,
                    )[0]
                )

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    plot_curve(csv_path, fig_path, algorithm, True)

    # Save model
    save_path = os.path.join(log_dir, 'model.pth')
    torch.save(agent, save_path)
    print('Model saved in', save_path)
