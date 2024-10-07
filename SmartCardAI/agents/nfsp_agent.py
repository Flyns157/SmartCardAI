import random
import torch
import torch.nn as nn
import torch.optim as optim
from ..utils import ReplayMemory

class NFSPAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(10000)

        self.gamma = gamma
        self.learning_rate = learning_rate

        self.best_response_model = self._build_model()
        self.average_model = self._build_model()
        self.optimizer = optim.Adam(self.best_response_model.parameters(), lr=self.learning_rate)

    def _build_model(self):
        # Simple neural network
        return nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )

    def select_action(self, state):
        if random.random() < 0.5:
            state = torch.FloatTensor(state)
            return torch.argmax(self.best_response_model(state)).item()
        else:
            state = torch.FloatTensor(state)
            return torch.argmax(self.average_model(state)).item()

    def train(self, batch_size=32):
        # Training logic here (similar to DQN but two models)
        pass

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
