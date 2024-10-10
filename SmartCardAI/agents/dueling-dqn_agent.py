import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

# --- Double DQN with Prioritized Experience Replay and Dueling DQN ---

class DuelingDQNAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDQNAgent, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Value stream
        self.value_fc = nn.Linear(128, 1)
        
        # Advantage stream
        self.advantage_fc = nn.Linear(128, action_size)
        
        self.dropout = nn.Dropout(p=0.2)  # Dropout regularization
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(torch.relu(self.fc2(x)))  # Apply Dropout
        
        value = self.value_fc(x)
        advantage = self.advantage_fc(x)
        
        # Combining value and advantage streams
        q_value = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_value


class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, alpha=0.6):
        self.buffer_size = buffer_size
        self.alpha = alpha  # controls how much prioritization is used
        self.buffer = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.position = 0

    def add(self, experience, td_error):
        max_priority = max(self.priorities) if self.buffer else 1.0
        self.buffer.append(experience)
        self.priorities.append(max_priority)

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return [], []

        priorities = np.array(self.priorities) ** self.alpha
        sampling_probs = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=sampling_probs)
        experiences = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * sampling_probs[indices]) ** (-beta)
        weights /= weights.max()

        return experiences, weights, indices

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-5  # avoid 0 priority


class DQNAgent:
    def __init__(self, state_size, action_size, seed, gamma=0.99, lr=5e-4, buffer_size=10000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q-Network
        self.qnetwork_local = DuelingDQNAgent(state_size, action_size).to(self.device)
        self.qnetwork_target = DuelingDQNAgent(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr, weight_decay=0.0)

        # Replay memory
        self.memory = PrioritizedReplayBuffer(buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Calculate TD error for the new experience
        with torch.no_grad():
            q_target_next = self.qnetwork_target(next_state).max(1)[0].unsqueeze(1)
            q_expected = self.qnetwork_local(state).gather(1, torch.tensor([[action]]))
            td_error = abs(reward + self.gamma * q_target_next * (1 - done) - q_expected).item()

        # Save experience in replay memory with the TD error as priority
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.add(experience, td_error)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0 and len(self.memory.buffer) >= self.batch_size:
            experiences, weights, indices = self.memory.sample(self.batch_size)
            self.learn(experiences, weights, indices)

    def learn(self, experiences, weights, indices):
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Get max predicted Q values (for next states) from target model
        with torch.no_grad():
            q_targets_next = self.qnetwork_target(next_states).max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        q_expected = self.qnetwork_local(states).gather(1, actions.unsqueeze(1))
        
        # Compute loss, weighted by importance-sampling (weights)
        loss = (weights * (q_expected - q_targets) ** 2).mean()

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the priorities in replay memory
        td_errors = (q_expected - q_targets).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

    def act(self, state, epsilon=0.0):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def update_target_network(self):
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
