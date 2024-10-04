import os
from matplotlib.figure import Figure
from rlcard import models
from rlcard.agents import RandomAgent
from rlcard.envs import Env
import rlcard
import numpy as np

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
