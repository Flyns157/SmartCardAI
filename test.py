from rlcard.agents import DQNAgent

import numpy as np

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import get_device

# Créer l'environnement
env = rlcard.make('uno')

# Check whether gpu is available
device = get_device()

# Créer l'agent DQN
agent = DQNAgent(
    num_actions=env.num_actions,
    state_shape=env.state_shape[0],
    mlp_layers=[64, 64],  # Vous pouvez ajuster les couches selon vos besoins
    device=device,
)

# Ajouter l'agent à l'environnement
env.set_agents([agent] + [RandomAgent(num_actions=env.num_actions) for _ in range(env.num_players - 1)])

import copy

class DQNGeneration:
    def __init__(self, base_agent, num_variants=5):
        self.base_agent = base_agent
        self.num_variants = num_variants
        self.agents = []

    def create_variants(self):
        for _ in range(self.num_variants):
            # Clone the base agent and possibly change hyperparameters
            variant = copy.deepcopy(self.base_agent)
            # Par exemple, tu peux changer le taux d'apprentissage ou d'autres paramètres
            # variant.learning_rate *= np.random.uniform(0.5, 1.5)  # Ajustement aléatoire du taux d'apprentissage
            self.agents.append(variant)

    def evaluate_variants(self, env, num_games=100):
        results = []
        for agent in self.agents:
            # Évaluer chaque agent
            env.set_agents([agent] + [RandomAgent(num_actions=env.num_actions) for _ in range(env.num_players - 1)])
            win_count = 0
            for _ in range(num_games):
                trajectories, payoffs = env.run(is_training=False)
                if payoffs[0] > 0:
                    win_count += 1
            results.append(win_count)
        return results

    def select_best_variants(self, results):
        # Garde les agents avec le meilleur score
        best_agents = np.argsort(results)[-2:]  # Garde les deux meilleurs
        self.agents = [self.agents[i] for i in best_agents]


num_generations = 10
num_episodes = 100
num_eval_games = 1000
for generation in range(num_generations):
    print(f"Generation {generation + 1}")
    generation_manager = DQNGeneration(agent)
    generation_manager.create_variants()
    
    # Entraînement des variantes
    for episode in range(num_episodes):
        # Entraîner chaque agent de la génération
        for agent in generation_manager.agents:
            # Exécuter le code d'entraînement ici pour chaque agent
            pass

    # Évaluer les variantes
    results = generation_manager.evaluate_variants(env, num_eval_games)
    
    # Sélectionner les meilleurs agents pour la prochaine génération
    generation_manager.select_best_variants(results)
