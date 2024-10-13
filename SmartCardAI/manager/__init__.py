from ..utils import tournament, load_model
from ..agents import *
import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import (
    get_device,
    set_seed,
    reorganize,
    Logger
)
import numpy as np

class Pool(object):
    MODES = ('loader', 'generator')
    
    def __init__(self, agents:list[str|list[int]], env_type:str='uno', mode:str = 'loader', seed:int = 42, display:bool = False, agent_type:str='dqn') -> None:
        self.set_mode(mode)
        self.seed = seed
        self.env = rlcard.make(
            env_type,
            config={
                'seed': seed,
            }
        )
        self.display = display
        device = get_device()
        self.agents = [load_model(model_path=path, env=self.env, device=device) for path in agents] if self.mode == 0 else [self.gen_agent(mlp_layers=mlp_layers, agent_type=agent_type) for mlp_layers in agents]

    def set_mode(self, mode) -> None:
        tmp = Pool.recognise_mode(mode)
        if tmp is None: raise ValueError(f'The specified mode is invalid : {mode}')
        self.mode = tmp

    def tournament(self, num_games:int = 1000):
        """
        Evalue une liste d'agents dans un tournoi en mode round-robin et retourne un classement des agents avec leur taux de victoires.

        Args:
            num_games (int): Le nombre de parties à jouer pour chaque duel d'agents (par défaut 1000).

        Returns:
            list: Liste triée des agents avec leur taux de victoires sous forme de tuples (index de l'agent, taux de victoires).
        """
        # Créer un tableau pour stocker le nombre de victoires pour chaque agent
        num_agents = len(self.agents)
        wins = np.zeros(num_agents)

        # Comparer chaque agent à tous les autres dans un tournoi round-robin
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                # Associer les deux agents au jeu
                self.env.set_agents([self.agents[i], self.agents[j]])
                
                # Jouer les parties
                _, (first_agent_wins, second_agent_wins) = tournament(env=self.env, num=num_games)

                # Mettre à jour le nombre de victoires
                wins[i] += first_agent_wins
                wins[j] += second_agent_wins

        # Calculer le taux de victoires pour chaque agent
        total_duels = num_games * (num_agents - 1)  # Chaque agent joue contre (num_agents - 1) autres agents
        win_rates = wins / total_duels

        # Créer une liste des agents avec leur taux de victoires et la trie par taux de victoires décroissant
        results = sorted([(i, win_rate) for i, win_rate in enumerate(win_rates)], key=lambda x: x[1], reverse=True)

        # Afficher le classement
        if self.display:
            print("Classement des agents :")
            for rank, (agent_idx, win_rate) in enumerate(results, start=1): print(f"Rank {rank}: Agent {agent_idx} - Taux de victoires : {win_rate:.2%}")

        return results
    
    def train(self):
        self.env.set_agents(self.agents + [RandomAgent(num_actions=self.env.num_actions)for _ in range(len(self.agents), self.env.num_players)])
        device = get_device()
        if device == 'cpu':
            
    
    @staticmethod
    def recognise_mode(mode):
        if isinstance(mode, int) and 0 <= mode < len(Pool.MODES): return mode
        if isinstance(mode, str):
            if mode.lower() in Pool.MODES: return Pool.MODES.index(mode)
            N = len(Pool.MODES)
            i = 0
            while i < N:
                if mode.lower() in Pool.MODES[i]: return i
                i += 1
    
    def gen_agent(self, mlp_layers:list[int], agent_type:str='dqn', **kwargs) -> Agent:
        # Vérifier si le GPU est disponible
        device = get_device()
        # Initialiser les graines pour numpy, torch, random
        set_seed(self.seed)

        # Initialiser l'agent et utiliser des agents aléatoires comme adversaires
        if agent_type == 'dqn':
            return DQNAgent(
                num_actions=self.env.num_actions,
                state_shape=self.env.state_shape[0],
                mlp_layers=mlp_layers,
                device=device,
                **kwargs
            )
        if agent_type == 'ddqn':
            return DuelingDQNAgent(
                action_size=self.env.num_actions,
                state_size=self.env.state_shape[0],
                mlp_layers=mlp_layers,
                device=device
            )
        elif agent_type == 'nfsp':
            return NFSPAgent(
                num_actions=self.env.num_actions,
                state_shape=self.env.state_shape[0],
                hidden_layers_sizes=mlp_layers,
                q_mlp_layers=[64, 64],
                device=device,
                **kwargs
            )
    
    def add_agent(self, agents:list[str|list[int]], agent_type:str='dqn', **kwargs) -> bool:
        device = get_device()
        if isinstance(agents[0], list):
            self.agents += [load_model(model_path=path, env=self.env, device=device) for path in agents] if self.mode == 0 else [self.gen_agent(mlp_layers=mlp_layers, agent_type=agent_type) for mlp_layers in agents]
        elif isinstance(agents, str):
            self.agents.append(load_model(model_path=agents, env=self.env, device=device))
        elif isinstance(agents[0], int):
            self.agents.append(self.gen_agent(mlp_layers=agents, agent_type=agent_type, **kwargs))
