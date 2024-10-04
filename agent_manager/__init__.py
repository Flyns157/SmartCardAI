name = "agent_manager"
__version__ = "1.2.1"

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import (
    get_device,
    set_seed,
    reorganize,
    tournament,
    Logger
)
from rlcard.agents.human_agents.uno_human_agent import _print_action

import os
import torch
import random
import time
from .utils import load_model, plot_curve

def lets_play_uno(first_agent, second_agent) -> int:
    """
    Joue deux parties de Uno entre deux agents.

    Args:
        first_agent: Le premier agent.
        second_agent: Le deuxième agent.
    """
    # Créer l'environnement pour le jeu Uno
    env = rlcard.make('uno')

    # Associer les agents à l'environnement
    env.set_agents([first_agent, second_agent])

    for i in range(2):
        print(">> Start a new game")

        trajectories, payoffs = env.run(is_training=False)
        # Si l'humain ne prend pas la dernière action, nous devons
        # imprimer l'action des autres joueurs
        final_state = trajectories[0][-1]
        action_record = final_state['action_record']
        state = final_state['raw_obs']
        _action_list = []
        for i in range(1, len(action_record) + 1):
            _action_list.insert(0, action_record[-i])
        for pair in _action_list:
            print('>> Player', pair[0], 'chooses ', end='')
            _print_action(pair[1])
            print('')

        print('===============     Result     ===============')
        if payoffs[0] > 0:
            print('Player 0 win!')
            print(type(env.agents[0]))
            return 0
        else:
            print('Player 1 win!')
            print(type(env.agents[1]))
            return 1

# A implémenter prochainement {'pretrained_agent_ratio': 0.5, 'pretrained_model_path': 'experiments/pretrained_model.pth', 'start_eps': 1.0, 'end_eps': 0.1, 'decay_episodes': 1000, 'memory_size': 10000, 'batch_size': 32, 'save_every': None}
def train(env_type: str, algorithm: str, seed: int, num_episodes: int = 5000, num_eval_games: int = 2000, evaluate_every: int = 100, dir: str = 'experiments/', max_time: int = 600, resume_training:str = None, train_against_self: bool = False, mlp_layers:list[int] = [64, 64], *args, **kwargs):
    """
    Entraîne un agent dans un environnement donné.

    Args:
        env_type (str): Le type d'environnement.
        algorithm (str): L'algorithme à utiliser pour l'entraînement.
        seed (int): La graine pour la reproductibilité.
        num_episodes (int, optional): Le nombre d'épisodes d'entraînement. Par défaut 5000.
        num_eval_games (int, optional): Le nombre de parties d'évaluation. Par défaut 2000.
        evaluate_every (int, optional): La fréquence d'évaluation. Par défaut 100.
        dir (str, optional): Le répertoire de journalisation et de sauvegarde. Par défaut 'experiments/'.
        max_time (int, optional): Le temps maximum d'entraînement en secondes. Par défaut 600.
        resume_training (str, optional): Reprendre l'entraînement à partir d'un modèle existant si le nom est spécifié. Par défaut None.
        train_against_self (bool, optional): Entraîner l'agent contre lui-même. Par défaut False.
        *args: Arguments supplémentaires.
        **kwargs: Arguments supplémentaires.

    Returns:
        None
    """
    # Vérifier si le GPU est disponible
    device = get_device()

    # Initialiser les graines pour numpy, torch, random
    set_seed(seed)

    # Créer l'environnement avec la graine
    env = rlcard.make(
        env_type,
        config={
            'seed': seed,
        }
    )

    # Initialiser l'agent et utiliser des agents aléatoires comme adversaires
    if algorithm == 'dqn':
        from rlcard.agents import DQNAgent
        agent = DQNAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            mlp_layers=mlp_layers,
            device=device,
        )
    elif algorithm == 'nfsp':
        from rlcard.agents import NFSPAgent
        agent = NFSPAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            hidden_layers_sizes=[64, 64],
            q_mlp_layers=[64, 64],
            device=device,
        )

    # Charger un modèle existant si resume_training contient quelque chose
    if resume_training is not None:
        model_path = os.path.join(dir, resume_training)
        if os.path.exists(model_path):
            agent = load_model(model_path=model_path, device=device)
        else:
            print(f"/!\\ The {resume_training} model don't exist")
            print(f"Start training a new model ...")
            time.sleep(3)

    agents = [agent]
    if train_against_self:
        for _ in range(1, env.num_players):
            agents.append(agent)
    else:
        for _ in range(1, env.num_players):
            agents.append(RandomAgent(num_actions=env.num_actions))
            # agents.append(UNORuleModelV2().agents[0])

    # Mélanger les agents pour choisir l'agent de départ au hasard
    random.shuffle(agents)
    env.set_agents(agents)

    # Commencer l'entraînement
    start_time = time.time()
    with Logger(dir) as logger:
        for episode in range(num_episodes):

            if max_time and (time.time() - start_time) > max_time:
                break

            if algorithm == 'nfsp':
                agents[0].sample_episode_policy()

            # Générer des données à partir de l'environnement
            trajectories, payoffs = env.run(is_training=True)

            # Réorganiser les données pour être état, action, récompense, état suivant, terminé
            trajectories = reorganize(trajectories, payoffs)

            # Alimenter les transitions dans la mémoire de l'agent et entraîner l'agent
            for ts in trajectories[0]:
                agent.feed(ts)

            # Évaluer la performance. Jouer avec des agents aléatoires.
            if episode % evaluate_every == 0:
                logger.log_performance(
                    episode,
                    tournament(
                        env,
                        num_eval_games,
                    )[0]
                )

        # Obtenir les chemins
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Tracer la courbe d'apprentissage
    plot_curve(csv_path, fig_path, algorithm, display_avg=True)

    # Sauvegarder le modèle
    save_path = os.path.join(dir, 'model.pth')
    torch.save(agent, save_path)
    print('Model saved in', save_path)
