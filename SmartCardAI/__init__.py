name = "SmartCardAI"
__version__ = "2.2.5"

from .utils import load_model, plot_curve, EpsGreedyDecay
from collections import deque
import time
import random
import torch
import os
from rlcard.agents.human_agents.uno_human_agent import _print_action
from rlcard.utils import (
    get_device,
    set_seed,
    reorganize,
    tournament,
    Logger
)
from rlcard.agents import RandomAgent
import rlcard


# The next code is deprecated, it will soon no longer be functional


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
        # state = final_state['raw_obs']
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


def train(env_type: str,
            algorithm: str,
            seed: int,
            num_episodes: int = 5000,
            num_eval_games: int = 2000,
            evaluate_every: int = 100,
            dir: str = 'experiments/',
            max_time: int = 600,
            resume_training: str = None,
            train_against_self: bool = False,
            mlp_layers: list[int] = [64, 64],
            pretrained_agent_ratio: float = 0.5,
            pretrained_model_path: str = 'experiments/pretrained_model.pth',
            start_eps: float = 1.0,
            end_eps: float = 0.1,
            decay_episodes: int = 1000,
            memory_size: int = 10000,
            batch_size: int = 32,
            save_every: int = None,
            *args,
            **kwargs):
    """
    Entraîne un agent en utilisant un algorithme donné dans un environnement spécifique.

    Args:
        env_type (str): Le type d'environnement (par exemple 'uno').
        algorithm (str): L'algorithme utilisé pour l'entraînement (par exemple 'dqn', 'nfsp').
        seed (int): La graine aléatoire pour garantir la reproductibilité.
        num_episodes (int, optional): Le nombre d'épisodes pour l'entraînement (par défaut 5000).
        num_eval_games (int, optional): Le nombre de jeux à jouer lors des évaluations (par défaut 2000).
        evaluate_every (int, optional): La fréquence d'évaluation en nombre d'épisodes (par défaut 100).
        dir (str, optional): Le répertoire où enregistrer les résultats et modèles (par défaut 'experiments/').
        max_time (int, optional): Le temps maximum d'entraînement en secondes (par défaut 600).
        resume_training (str, optional): Nom du modèle à partir duquel reprendre l'entraînement (par défaut None).
        train_against_self (bool, optional): Si True, entraîne l'agent contre lui-même (par défaut False).
        mlp_layers (list[int], optional): La structure des couches du MLP pour les agents DQN ou NFSP (par défaut [64, 64]).
        pretrained_agent_ratio (float, optional): Proportion d'agents pré-entrainés à utiliser comme adversaires (par défaut 0.5).
        pretrained_model_path (str, optional): Chemin vers le modèle pré-entraîné (par défaut 'experiments/pretrained_model.pth').
        start_eps (float, optional): Valeur initiale d'epsilon pour epsilon-greedy (par défaut 1.0).
        end_eps (float, optional): Valeur finale d'epsilon après le déclin (par défaut 0.1).
        decay_episodes (int, optional): Le nombre d'épisodes sur lequel epsilon décroît (par défaut 1000).
        memory_size (int, optional): Taille de la mémoire pour l'apprentissage par mini-lots (par défaut 10000).
        batch_size (int, optional): Taille des mini-lots pour l'entraînement (par défaut 32).
        save_every (int, optional): Fréquence d'enregistrement du modèle en nombre d'épisodes (par défaut None).
        *args, **kwargs: Arguments supplémentaires.

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
        from .agents import DQNAgent
        agent = DQNAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            mlp_layers=mlp_layers,
            device=device,
            **kwargs
        )
    elif algorithm == 'nfsp':
        from .agents import NFSPAgent
        agent = NFSPAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            hidden_layers_sizes=mlp_layers,
            q_mlp_layers=[64, 64],
            device=device,
            **kwargs
        )

    # Charger un modèle existant si resume_training contient quelque chose
    if resume_training is not None:
        model_path = os.path.join(dir, resume_training)
        if os.path.exists(model_path):
            agent = load_model(model_path=model_path, device=device)
        else:
            print(f"/!\\ The {resume_training} model don't exist")
            print("Start training a new model ...")
            time.sleep(3)

    agents = [agent]
    if train_against_self:
        for _ in range(1, env.num_players):
            agents.append(agent)
    else:
        for _ in range(1, env.num_players):
            # Train against a mix of random agents and previously trained versions of the main agent
            if random.random() < pretrained_agent_ratio:
                # Load less trained agent versions
                agents.append(torch.load(pretrained_model_path))
            else:
                # agents.append(UNORuleModelV2().agents[0])
                agents.append(RandomAgent(num_actions=env.num_actions))

    # Mélanger les agents pour choisir l'agent de départ au hasard
    random.shuffle(agents)
    env.set_agents(agents)

    # Initialize replay memory for mini-batch training
    memory = deque(maxlen=memory_size)

    # Start training
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
                memory.append(ts)

            # If enough experiences, perform mini-batch updates
            if len(memory) >= batch_size:
                mini_batch = random.sample(memory, batch_size)
                for ts in mini_batch:
                    agent.feed(ts)

            # Dynamically adjust epsilon for exploration
            # TODO : ajuster le code afin de pouvoir faire un équivalent
            # if algorithm == 'dqn':
            #     agent.epsilon = eps_decay.get_epsilon(episode)

            # Evaluate the performance. Play with random agents.
            if episode % evaluate_every == 0:
                logger.log_performance(
                    episode,
                    tournament(
                        env,
                        num_eval_games,
                    )[0]
                )

            # Save intermediate models every few episodes
            if save_every is not None and episode % save_every == 0:
                save_path = os.path.join(
                    dir, f'model_saves/model_{episode}.pth')
                torch.save(agent, save_path)
                print(f'Model saved at episode {episode}')

        # Obtenir les chemins
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Tracer la courbe d'apprentissage
    plot_curve(csv_path, fig_path, algorithm, display_avg=True)

    # Sauvegarder le modèle
    save_path = os.path.join(dir, 'model.pth')
    torch.save(agent, save_path)
    print('Final model saved in', save_path)
