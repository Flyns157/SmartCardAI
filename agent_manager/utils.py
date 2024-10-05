import os
from matplotlib.figure import Figure
from rlcard import models
from rlcard.agents import RandomAgent
from rlcard.envs import Env
import rlcard
import numpy as np

class EpsGreedyDecay:
    """
    Gère le déclin d'epsilon pour la stratégie epsilon-greedy utilisée dans l'exploration des agents DQN.

    Args:
        start_eps (float): La valeur initiale d'epsilon (par défaut 1.0).
        end_eps (float): La valeur finale d'epsilon après déclin (par défaut 0.1).
        decay_episodes (int): Le nombre d'épisodes sur lesquels epsilon décroît (par défaut 1000).

    Attributes:
        eps (float): La valeur actuelle d'epsilon, initialisée à `start_eps`.

    Methods:
        get_epsilon(episode: int) -> float:
            Calcule et retourne la valeur actuelle d'epsilon en fonction de l'épisode en cours.
    """

    def __init__(self, start_eps=1.0, end_eps=0.1, decay_episodes=1000):
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.decay_episodes = decay_episodes
        self.eps = start_eps

    def get_epsilon(self, episode):
        """
        Calcule la valeur actuelle d'epsilon en fonction de l'épisode.

        Args:
            episode (int): Le numéro de l'épisode actuel.

        Returns:
            float: La valeur actuelle d'epsilon, décroit progressivement de start_eps à end_eps.
        """
        self.eps = max(self.end_eps, self.start_eps - (self.start_eps - self.end_eps) * episode / self.decay_episodes)
        return self.eps

def load_model(model_path:str, env:Env = None, position:int = None, device:str = None, weights_only:bool = False):
    """
    Charge un modèle d'agent à partir d'un chemin donné.

    Args:
        model_path (str): Le chemin du modèle à charger.
        env (Environment, optional): L'environnement de jeu. Par défaut None.
        position (int, optional): La position de l'agent dans l'environnement. Par défaut None.
        device (str, optional): Le dispositif à utiliser (CPU ou GPU). Par défaut None.

    Returns:
        agent: L'agent chargé.
    """
    if os.path.isfile(model_path):  # Torch model
        import torch
        agent = torch.load(model_path, map_location=device, weights_only=weights_only)
        agent.set_device(device)
    elif os.path.isdir(model_path):  # CFR model
        from rlcard.agents import CFRAgent
        agent = CFRAgent(env, model_path)
        agent.load()
    elif model_path == 'random':  # Random model
        from rlcard.agents import RandomAgent
        agent = RandomAgent(num_actions=env.num_actions)
    else:  # A model in the model zoo
        from rlcard import models
        agent = models.load(model_path).agents[position]

    print(f'Model loaded from {model_path}')
    return agent

def tournament(env: Env, num: int, display_results: bool = False):
    ''' Evaluate the performance of the agents in the environment and display the results.

    Args:
        env (Env class): The environment to be evaluated.
        num (int): The number of games to play.
        display_results (bool): if it must display results.

    Returns:
        payoffs (list): A list of average payoffs for each player.
        wins (list): A list of the number of wins for each player.

    Functionality:
        - Runs the specified number of games (`num`) in the environment.
        - Tracks the number of wins for each agent and calculates the average payoffs.
        - Sorts the agents based on their number of wins, from most to least effective.
        - Displays a ranking of agents by number of wins, including their average payoff.
    '''
    payoffs = [0 for _ in range(env.num_players)]
    wins = [0 for _ in range(env.num_players)]
    counter = 0
    
    while counter < num:
        _, _payoffs = env.run(is_training=False)
        
        if isinstance(_payoffs, list):
            for _p in _payoffs:
                winner = np.argmax(_p)
                wins[winner] += 1
                for i, _ in enumerate(payoffs):
                    payoffs[i] += _p[i]
                counter += 1
        else:
            winner = np.argmax(_payoffs)
            wins[winner] += 1
            for i, _ in enumerate(payoffs):
                payoffs[i] += _payoffs[i]
            counter += 1
    
    for i, _ in enumerate(payoffs):
        payoffs[i] /= counter

    # Combine wins and payoffs into tuples for sorting
    results = [(i, wins[i], payoffs[i]) for i in range(env.num_players)]

    # Sort by number of wins (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Print the results in order of efficiency
    if display_results:
        for i, (agent, win_count, avg_payoff) in enumerate(results):
            print(f"[ {win_count/num:.2%} ] Rank {i + 1}: Agent {agent} - Wins: {win_count}, Avg Payoff: {avg_payoff:.2f}")

    return payoffs, wins

def rank_agents(agents, env_type='uno', num_games=1000, display_results:bool = False):
    """
    Evalue une liste d'agents dans un tournoi en mode round-robin et retourne un classement des agents avec leur taux de victoires.

    Args:
        agents (list): Liste des agents à évaluer.
        env_type (str): Le type d'environnement à utiliser (par défaut 'uno').
        num_games (int): Le nombre de parties à jouer pour chaque duel d'agents (par défaut 1000).
        display_results (bool): Si les résultats doivent être affichés.

    Returns:
        list: Liste triée des agents avec leur taux de victoires sous forme de tuples (index de l'agent, taux de victoires).
    """
    # Créer l'environnement
    env = rlcard.make(env_type)

    # Créer un tableau pour stocker le nombre de victoires pour chaque agent
    num_agents = len(agents)
    wins = np.zeros(num_agents)

    # Comparer chaque agent à tous les autres dans un tournoi round-robin
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            # Associer les deux agents au jeu
            env.set_agents([agents[i], agents[j]])
            
            # Jouer les parties
            _, (first_agent_wins, second_agent_wins) = tournament(env=env, num=num_games)

            # Mettre à jour le nombre de victoires
            wins[i] += first_agent_wins
            wins[j] += second_agent_wins

    # Calculer le taux de victoires pour chaque agent
    total_duels = num_games * (num_agents - 1)  # Chaque agent joue contre (num_agents - 1) autres agents
    win_rates = wins / total_duels

    # Créer une liste des agents avec leur taux de victoires
    results = [(i, win_rate) for i, win_rate in enumerate(win_rates)]

    # Trier par taux de victoires décroissant
    results.sort(key=lambda x: x[1], reverse=True)

    # Afficher le classement
    if display_results:
        print("Classement des agents :")
        for rank, (agent_idx, win_rate) in enumerate(results, start=1):
            print(f"Rank {rank}: Agent {agent_idx} - Taux de victoires : {win_rate:.2%}")

    return results

def agent_1v1(agent, agent_bis=None, num_games:int = 10000, env_type:str = 'uno'):
    """
    Évalue deux agents en jouant un nombre donné de parties.

    Args:
        agent: Le premier agent à évaluer.
        agent_bis (optional): Le deuxième agent à évaluer. Par défaut, un agent aléatoire.
        num_games (int, optional): Le nombre de parties à jouer. Par défaut 10000.

    Returns:
        tuple: Le nombre de victoires pour chaque agent.
    """
    # Créer l'environnement pour le jeu Uno
    env = rlcard.make(env_type)

    if agent_bis is None:
        agent_bis = RandomAgent(num_actions=env.num_actions)

    # Associer les agents à l'environnement
    env.set_agents([agent, agent_bis])

    # Lancer le tournoi
    return tournament(env=env, num=num_games, display_results=True)[1]

def avg(E: list | tuple | set) -> int | float:
    ''' Calculate the average of a list, tuple, or set of numbers.

    Args:
        E (list, tuple, or set): A collection of numeric values (int or float) to compute the average.

    Returns:
        int or float: The average of the values in the collection.
    
    Raises:
        ZeroDivisionError: If the input collection is empty.

    Example:
        avg([1, 2, 3, 4, 5])   # Returns 3.0
        avg((10, 20, 30))      # Returns 20.0
        avg({1.5, 2.5, 3.5})   # Returns 2.5
    '''
    return sum(E) / len(E)


def plot_curve(csv_path: str, save_path: str, algorithm: str, display_avg: bool = False) -> Figure:
    ''' Plot and save a reward curve from a CSV file.

    Args:
        csv_path (str): The path to the CSV file containing the episode-reward data.
        save_path (str): The path where the generated plot will be saved.
        algorithm (str): The name of the algorithm to be displayed in the plot legend.
        display_avg (bool, optional): Whether to display the average reward line (default is False).

    Returns:
        Figure: The matplotlib figure object of the generated plot.

    CSV Format:
        The CSV file should contain at least two columns: 
        - 'episode': An integer representing the episode number.
        - 'reward': A float representing the reward value for that episode.

    Functionality:
        - Reads episode-reward pairs from the CSV file and plots them on a graph.
        - If `display_avg` is set to True, an average line for the rewards is plotted.
        - The plot is labeled with the algorithm's name and saved to the specified `save_path`.
        - If the directory for `save_path` does not exist, it is created.

    Example:
        plot_curve('data/rewards.csv', 'plots/reward_curve.png', 'DQN', display_avg=True)
    '''
    import os
    import csv
    import matplotlib.pyplot as plt
    with open(csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        xs = []
        ys = []
        for row in reader:
            xs.append(int(row['episode']))
            ys.append(float(row['reward']))
        fig, ax = plt.subplots()
        ax.plot(xs, ys, label=algorithm)
        if display_avg:
            ax.plot(xs, [avg(ys)] * len(xs), label=algorithm + "_avg")
        ax.set(xlabel='episode', ylabel='reward')
        ax.legend()
        ax.grid()

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fig.savefig(save_path)

        return fig
