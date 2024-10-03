import os
from matplotlib.figure import Figure
from rlcard import models
from rlcard.agents import RandomAgent
from rlcard.envs import Env
import rlcard

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












# ======================================================================================================================================================================
# ======================================================================================================================================================================










def tournament(env: Env, num: int, display_wins:bool = False):
    ''' Evaluate the performance of the agents in the environment

    Args:
        env (Env class): The environment to be evaluated.
        num (int): The number of games to play.

    Returns:
        A list of average payoffs for each player and the number of wins for each player.
    '''
    payoffs = [0 for _ in range(env.num_players)]
    wins = [0 for _ in range(env.num_players)]  # To track the number of wins
    counter = 0
    
    while counter < num:
        _, _payoffs = env.run(is_training=False)
        
        if isinstance(_payoffs, list):
            for _p in _payoffs:
                winner = _p.index(max(_p))  # Find the index of the winning player
                wins[winner] += 1           # Increment the win count for the winner
                for i, _ in enumerate(payoffs):
                    payoffs[i] += _p[i]
                counter += 1
        else:
            winner = _payoffs.index(max(_payoffs))  # Single game: Find the winner
            wins[winner] += 1                       # Increment win count
            for i, _ in enumerate(payoffs):
                payoffs[i] += _payoffs[i]
            counter += 1
    
    for i, _ in enumerate(payoffs):
        payoffs[i] /= counter
    
    # Print the number of wins for each player
    for i, win_count in enumerate(wins):
        print(f"[{win_count/num:.2%}] Agent {i} has {win_count} wins.")
    
    return payoffs, wins

def evaluate_agents(agent, agent_bis=None, num_games=10000):
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
    env = rlcard.make('uno')

    if agent_bis is None:
        agent_bis = RandomAgent(num_actions=env.num_actions)

    # Associer les agents à l'environnement
    env.set_agents([agent, agent_bis])

    # Variables pour compter les résultats
    first_agent_wins = 0
    second_agent_wins = 0

    # Lancer les parties
    for _ in range(num_games):
        # Exécuter une partie
        trajectories, payoffs = env.run(is_training=False)

        # Le payoff du premier joueur correspond à l'agent basé sur des règles
        if payoffs[0] > 0:
            first_agent_wins += 1
        else:
            second_agent_wins += 1

    # Afficher les résultats finaux
    print(f"Après {num_games} parties :")
    print(f"Agent 1 a gagné {first_agent_wins} fois")
    print(f"Agent 2 a gagné {second_agent_wins} fois")
    print(f"Taux de victoire de l'agent 1 : {first_agent_wins / num_games:.2%}")
    print(f"Taux de victoire de l'agent 2 : {second_agent_wins / num_games:.2%}")

    return first_agent_wins, second_agent_wins


# ======================================================================================================================================================================
# ======================================================================================================================================================================



def avg(E:list|tuple|set) -> int|float:
    return sum(E)/len(E)

def plot_curve(csv_path:str, save_path:str, algorithm:str, display_avg:bool = False) -> Figure:
    ''' Read data from csv file and plot the results
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
        ax.plot([avg(xs)]*len(xs), ys, label=algorithm+"_avg")
        ax.set(xlabel='episode', ylabel='reward')
        ax.legend()
        ax.grid()

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fig.savefig(save_path)

        return fig
