"""
This module provides the Model class,
which is a base class for models with some useful feautures for training with RLCard
and saving models in the pytorch format.
"""

import os
from pathlib import Path
from datetime import date

import torch

from rlcard.envs import Env
from rlcard.models.model import Model as RLCardModel
from rlcard.agents import DQNAgent, NFSPAgent

from ..utils import (
    load_model,
    get_device,
    type_check,
    reset_default_args
)


class Model(RLCardModel):
    """ A base class for models with some useful feautures for training with RLCard """
    @type_check
    @reset_default_args
    def __init__(self,
                env: Env,
                agent: str | type,
                path: Path | str = r'./experiments/',
                name: str | None = None,
                device: str | torch.device = get_device(),
                **kwargs
                ) -> None:
        """
            Initialize the model.

            Args:
                env (Env): The environment to run the game.
                agent (str or type): The agent to use for training.
                path (str or Path): The path to save the model.
                name (str): The name of the model.
                device (str or torch.device): The device to run the model.
                **kwargs: Other parameters.
        """

        self.env = env
        self.device = torch.device(device)

        if isinstance(agent, str):
            match agent:
                case 'dqn':
                    self.agent = DQNAgent(
                        num_actions=env.num_actions,
                        state_shape=env.state_shape[0],
                        mlp_layers=kwargs.pop('mlp_layers', [64, 64, 64, 64]),
                        device=self.device,
                        **kwargs
                    )
                case 'nfsp':
                    self.agent = NFSPAgent(
                        num_actions=env.num_actions,
                        state_shape=env.state_shape[0],
                        hidden_layers_sizes=kwargs.pop(
                            'hidden_layers_sizes', [64, 64]),
                        q_mlp_layers=kwargs.pop('q_mlp_layers', [64, 64]),
                        device=self.device,
                        **kwargs
                    )
                case _:
                    raise ValueError(f"Algorithm {agent} not supported")
        else:
            self.agent = agent(**kwargs)

        if name:
            self.name = name
            self.path = path if os.path.isdir(path) else os.path.dirname(path)
        elif os.path.isdir(path):
            self.name = f"{env.name}_{type(self.agent).__name__}_{
                date.today():%Y-%m-%d}.pth"
            self.path = path
        else:
            self.name = os.path.basename(path)
            self.path = os.path.dirname(path)

    @type_check
    def save(self, name: str | None = None, path: Path | str | None = None) -> None:
        """
            Save the model.

            Args:
                name (str): The name of the model.
                path (str or Path): The path to save the model.
        """
        save_path = os.path.join(path or self.path, name or self.name)
        torch.save(self.agent, save_path)
        print('Model saved in', save_path)

    @type_check
    def load(self, path: Path | str | None = None, name: str | None = None) -> None:
        """
            Load the model.

            Args:
                path (str or Path): The path to load the model.
                name (str): The name of the model.
        """
        if path:
            model_path = path if os.path.isfile(
                path) else os.path.join(path, name or self.name)
        else:
            model_path = os.path.join(self.path, name)

        if os.path.exists(model_path):
            self.agent = load_model(
                model_path=model_path, device=self.device, env=self.env)
        else:
            raise FileNotFoundError(
                f"/!\\ No model existing at : {model_path}")

    @property
    def agents(self) -> list:
        ''' Get a list of agents for each position in a the game

        Returns:
            agents (list): A list of agents

        Note:   Each agent should be just like RL agent with step and eval_step
                functioning well.
        '''
        return [self.agent]
