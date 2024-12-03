import os
import random
from pathlib import Path
from time import time

import torch

import rlcard
from rlcard.envs import Env
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

from rlcard.models.uno_rule_models import UNORuleAgentV1
from .rule_agents import UNORuleAgentV2, UNORuleAgentV4
from rlcard.agents import RandomAgent

ALL_UNO_MODELS = {
    'rd': RandomAgent,
    'v1': UNORuleAgentV1,
    'v2': UNORuleAgentV2,
    'v4': UNORuleAgentV4,
}


class Model(object):
    @type_check
    def __init__(self, env: Env, agent: str | type, path: Path | str = r'./experiments/', name: str | None = None, device: str | torch.device = get_device(), **kwargs) -> None:
        self.env = env
        self.device = torch.device(device)

        if isinstance(agent, str):
            match agent:
                case 'dqn':
                    from rlcard.agents import DQNAgent
                    self.agent = DQNAgent(
                        num_actions=env.num_actions,
                        state_shape=env.state_shape[0],
                        mlp_layers= kwargs.pop('mlp_layers', [64, 64, 64, 64]),
                        device=self.device,
                        **kwargs
                    )
                case 'nfsp':
                    from rlcard.agents import NFSPAgent
                    self.agent = NFSPAgent(
                        num_actions=env.num_actions,
                        state_shape=env.state_shape[0],
                        hidden_layers_sizes= kwargs.pop('hidden_layers_sizes', [64, 64]),
                        q_mlp_layers= kwargs.pop('q_mlp_layers', [64, 64]),
                        device=self.device,
                        **kwargs
                    )
                case _:
                    raise ValueError(f"Algorithm {agent} not supported")
        else:
            self.agent = agent

        if name:
            self.name = name
            self.path = path if os.path.isdir(path) else os.path.dirname(path)
        elif os.path.isdir(path):
            from datetime import date
            self.name = f"{env.name}_{type(self.agent).__name__}_{date.today().strftime('%Y-%m-%d')}"
            self.path = path
        else:
            self.name = os.path.basename(path)
            self.path = os.path.dirname(path)

    @type_check
    def save(self, name: str | None = None, path: Path | str | None = None) -> None:
        save_path = os.path.join(path or self.path, name or self.name)
        torch.save(self.agent, save_path)
        print('Model saved in', save_path)

    @type_check
    def load(self, path: Path | str | None = None, name: str | None = None) -> None:
        if path:
            model_path = path if os.path.isfile(path) else os.path.join(path, name or self.name)
        else:
            model_path = os.path.join(self.path, name)

        if os.path.exists(model_path):
            self.agent = load_model(model_path=model_path, device=self.device, env=self.env)
        else:
            raise FileNotFoundError(f"/!\\ No model existing at : {model_path}")


def UNOenv(seed: str | int | float) -> Env:
    set_seed(seed)
    return rlcard.make('uno', config={'seed': seed})

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

                dlogger.log(DTypes.INFO, message := f"{episode / num_episodes * 100:.2f}% - Elapsed time: {time() - start_time:.2f}s - device: {device}")
                print(message)

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    plot_curve(csv_path, fig_path, algorithm, True)

    # Save model
    model.save(path=log_dir)
