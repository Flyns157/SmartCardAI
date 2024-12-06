import os
from pathlib import Path

import torch

from rlcard.envs import Env
from ..utils import (
    load_model,
    get_device
)
from ..utils.type_checker import type_check

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
            self.name = f"{env.name}_{type(self.agent).__name__}_{date.today():%Y-%m-%d}.pth"
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
