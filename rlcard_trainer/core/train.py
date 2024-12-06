"""
This module provides functions to train a model on a given environment.
"""

import random
from pathlib import Path
from time import time

from rlcard.envs import Env
from rlcard.utils import (
    tournament,
    reorganize,
)
from rlcard.models.uno_rule_models import UNORuleAgentV1
from rlcard.agents import RandomAgent
from rlcard import make

from ..utils import (
    Logger,
    get_device,
    plot_curve,
    seconds_to_time,
    type_check,
    reset_default_args
)
from .model import Model
from ..rule_agents import UNORuleAgentV2, UNORuleAgentV4


ALL_UNO_MODELS = {
    'rd': RandomAgent,
    'v1': UNORuleAgentV1,
    'v2': UNORuleAgentV2,
    'v4': UNORuleAgentV4,
}


@type_check
@reset_default_args
def fill_env_with_agents(env: Env, agents: list = [], logger: Logger = None) -> None:
    """
        Fill the environment with agents.

        Args:
            env (Env): The environment to fill
            agents (list): List of agents to use in the environment
            logger (Logger): Logger to log the results

        Returns:
            None
    """
    for _ in range(len(agents), env.num_players):
        if env.name == 'uno' and (agent_model := random.choice(list(ALL_UNO_MODELS.keys()))) not in ('rd'):
            agents.append(ALL_UNO_MODELS[agent_model]())
        else:
            agents.append(RandomAgent(num_actions=env.num_actions))
    env.set_agents(agents)
    if logger:
        logger.log(
            f"Agents Used: {', '.join(set(str(type(a)) for a in agents))}"
        )
    print("ENV agents set")


@type_check
def train(
        seed: str | int | float,
        env: str,
        algorithm: str,
        num_episodes: int,
        num_eval_games: int,
        evaluate_every: int,
        log_dir:  Path | str,
        resume_training: bool,
        cuda: bool | str = True,
        **kwargs
    ) -> None:
    """
        Train a model on a given environment.

        Args:
            seed (int): Random seed for reproducibility
            env (str): Name of the environment
            algorithm (str): Name of the algorithm
            num_episodes (int): Number of episodes to train the model
            num_eval_games (int): Number of games to evaluate the model
            evaluate_every (int): Evaluate the model every x episodes
            log_dir (str): Directory to save the log and model
            resume_training (bool): Whether to resume training from a previous checkpoint
            cuda (bool): Whether to use cuda for training
            **kwargs: Other arguments for the model

        Returns:
            None
    """
    # Check whether gpu is available
    if 'cuda' in (device := get_device(True)) and not cuda:
        device = 'cpu'

    # Make the environment with seed
    env: Env = make(
        env,
        config={
            'seed': seed,
        }
    )
    num_games = num_episodes // int(num_episodes**0.5)

    # Make the agent
    model = Model(env, algorithm, device=device, **kwargs)
    try:
        if resume_training:
            model.load(path=log_dir)
    except FileNotFoundError:
        print(f"No model found at {log_dir}. Starting from scratch.")

    # Start training
    start_time = time()

    with Logger(log_dir) as logger:
        # Set agents in the environment
        agents = [model.agent]
        fill_env_with_agents(env, agents, logger)
        logger.log(
            f"Training an {algorithm} model on {num_games} {env.name} games with {env.num_players} players."
        )

        # Define a function to evaluate the model and log the results
        @reset_default_args
        def evaluate_model(episode: int, tmp_time: float = time()) -> None:
            """
                Evaluate the model and log the results.

                Args:
                    episode (int): Current episode
                    tmp_time (float): Time when the evaluation started.
            """
            logger.log_performance(
                num_episodes,
                tournament(
                    env,
                    num_eval_games,
                )[0],
                time() - tmp_time
            )
            logger.log(
                f"Episode {episode + 1}/{num_episodes} - Elapsed time: {seconds_to_time(tmp_time - start_time)} - device: {device}"
            )

        for episode in range(num_episodes):
            tmp_time = time()

            if algorithm == 'nfsp':
                agents[0].sample_episode_policy()

            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions into agent memory, and train the agent
            # TODO : We assume that model always plays the first position (it must change)
            for ts in trajectories[0]:
                model.agent.feed(ts)

            # Evaluate the performance.
            if episode % evaluate_every == 0:
                evaluate_model(episode, tmp_time)

            # Change oponents
            if episode % (num_games) == 0:
                fill_env_with_agents(env, agents[0], logger)

        # TODO : Evaluate the final model against randoms agents

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    plot_curve(csv_path, fig_path, algorithm, True)

    # Save model
    model.save(path=log_dir)
