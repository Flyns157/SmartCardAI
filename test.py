import random
import numpy as np
import torch
import rlcard
from rlcard.agents import DQNAgent

class DQNAgentUNO(DQNAgent):
    def __init__(self, num_actions, state_shape, mlp_layers=None, learning_rate=0.001, device=None):
        super().__init__(num_actions=num_actions, state_shape=state_shape, mlp_layers=mlp_layers, learning_rate=learning_rate, device=device)
        self.fitness = 0

    def mutate(self):
        # Implémentez ici une méthode pour modifier légèrement les poids du réseau
        for param in self.q_estimator.qnet.parameters():
            noise = torch.randn_like(param) * 0.1
            param.data += noise

class GenerationalDQNTrainer:
    def __init__(self, env, population_size=10, generations=100, games_per_evaluation=100):
        self.env = env
        self.population_size = population_size
        self.generations = generations
        self.games_per_evaluation = games_per_evaluation
        self.population = []

    def initialize_population(self):
        for _ in range(self.population_size):
            agent = DQNAgentUNO(
                num_actions=self.env.num_actions,
                state_shape=self.env.state_shape[0],
                mlp_layers=[64, 64]
            )
            self.population.append(agent)

    def evaluate_fitness(self, agent):
        total_reward = 0
        for _ in range(self.games_per_evaluation):
            state, _ = self.env.reset()
            done = False
            while not done:
                action, _ = agent.eval_step(state)
                state, reward, done, _, _ = self.env.step(action)
                total_reward += reward
        return total_reward / self.games_per_evaluation

    def select_best_agents(self):
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        return self.population[:self.population_size // 2]

    def create_next_generation(self, best_agents):
        next_generation = best_agents.copy()
        while len(next_generation) < self.population_size:
            parent = random.choice(best_agents)
            child = DQNAgentUNO(
                num_actions=self.env.num_actions,
                state_shape=self.env.state_shape[0],
                mlp_layers=[64, 64]
            )
            child.q_estimator.qnet.load_state_dict(parent.q_estimator.qnet.state_dict())
            child.mutate()
            next_generation.append(child)
        return next_generation

    def train(self):
        self.initialize_population()

        for generation in range(self.generations):
            print(f"Generation {generation + 1}/{self.generations}")

            for agent in self.population:
                agent.fitness = self.evaluate_fitness(agent)

            best_agents = self.select_best_agents()
            best_fitness = best_agents[0].fitness
            avg_fitness = sum(agent.fitness for agent in self.population) / len(self.population)

            print(f"Best fitness: {best_fitness:.2f}, Average fitness: {avg_fitness:.2f}")

            self.population = self.create_next_generation(best_agents)

        return best_agents[0]

# Utilisation
env = rlcard.make('uno')
trainer = GenerationalDQNTrainer(env)
best_agent = trainer.train()

# Sauvegarde du meilleur agent
torch.save(best_agent.q_estimator.qnet.state_dict(), 'best_uno_agent.pth')
