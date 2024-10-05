from agents import DQNAgent
from environments.rlcard_env import RLCardEnv
from experiments.experiment_manager import ExperimentManager

# Configuration de l'environnement
game = 'uno'  # Vous pouvez changer pour d'autres jeux comme 'blackjack'
env = RLCardEnv(game)

# Configuration de l'agent
state_size = env.get_state_shape()
action_size = env.get_action_shape()
agent = DQNAgent(state_size, action_size)

# Gestion de l'expérience
experiment = ExperimentManager(agent, env, episodes=500)
experiment.run()
