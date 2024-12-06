__version__ = "2.0.4"

import os
import random
from pathlib import Path
from time import time

import rlcard
from rlcard.envs import Env
from rlcard.utils import (
    tournament,
    reorganize,
    Logger,
)
from .utils import (
    get_device,
    plot_curve
)
from .utils.type_checker import type_check

from rlcard.models.uno_rule_models import UNORuleAgentV1
from .rule_agents import UNORuleAgentV2, UNORuleAgentV4
from rlcard.agents import RandomAgent

ALL_UNO_MODELS = {
    'rd': RandomAgent,
    'v1': UNORuleAgentV1,
    'v2': UNORuleAgentV2,
    'v4': UNORuleAgentV4,
}


@type_check
def fill_env_with_agents(env: Env, agents: list = []) -> None:
    for _ in range(len(agents), env.num_players):
        if env.name == 'uno' and(agent_model := random.choice(list(ALL_UNO_MODELS.keys()))) not in ('rd'):
            agents.append(ALL_UNO_MODELS[agent_model]())
        else:
            agents.append(RandomAgent(num_actions=env.num_actions))
    env.set_agents(agents)
    # print(f"Agents used: {', '.join(str(type(a)) for a in agents)}")
    print("ENV agents set")


@type_check
def train(seed: str | int | float, env: str, algorithm: str, num_episodes: int, num_eval_games: int, evaluate_every: int, log_dir:  Path | str, resume_training: bool, cuda: bool | str = True, *args, **kwargs):
    from debug_sys.logger import Logger as DLogger
    from debug_sys import Types as DTypes
    dlogger = DLogger(os.path.join(log_dir, 'train.log'))

    # Check whether gpu is available
    if 'cuda' in (device := get_device(True)) and not cuda: device = 'cpu'

    # Make the environment with seed
    env: Env = rlcard.make(
        env,
        config={
            'seed': seed,
        }
    )

    # Make the agent
    model = Model(env, algorithm, device=device, **kwargs)
    try:
        if resume_training: model.load(path=log_dir)
    except FileNotFoundError:
        print(f"No model found at {log_dir}. Starting from scratch.")

    # Set agents in the environment
    agents = [model.agent]
    fill_env_with_agents(env, agents)

    dlogger.log(DTypes.INFO, message := f"Training {env.num_players}-player {env.name} game with {algorithm} algorithm")
    print(message)
    dlogger.log(DTypes.INFO, message := f"Agents Used: {', '.join(set(str(type(a)) for a in agents))}")
    print(message)

    # Start training
    start_time = time()
    with Logger(log_dir) as logger:
        for episode in range(num_episodes):

            if algorithm == 'nfsp':
                agents[0].sample_episode_policy()

            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions into agent memory, and train the agent
            # TODO : Here, we assume that model always plays the first position -> it must be changed to be random
            for ts in trajectories[0]:
                model.agent.feed(ts)

            # Evaluate the performance.
            if episode % evaluate_every == 0:
                logger.log_performance(
                    episode,
                    tournament(
                        env,
                        num_eval_games,
                    )[0]
                )

                elapsed_time = time() - start_time
                h, elapsed_time = elapsed_time//3600, elapsed_time%3600
                m, s = elapsed_time//60, elapsed_time%60
                dlogger.log(DTypes.INFO, message := f"{episode / num_episodes:.2%}% - Elapsed time: {h}H {m}M {s}S - device: {device}")
                print(message)

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    plot_curve(csv_path, fig_path, algorithm, True)

    # Save model
    model.save(path=log_dir)
