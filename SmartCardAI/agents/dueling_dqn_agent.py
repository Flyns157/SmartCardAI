import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from typing import List, Tuple

# --- Double DQN with Prioritized Experience Replay and Dueling DQN ---

class DuelingDQNAgent(nn.Module):
    """
    Neural network implementing the Dueling DQN architecture.
    Separates the state value and action advantage streams.
    """
    def __init__(self, state_size: int, action_size: int) -> None:
        """
        Initialize parameters and build the Dueling DQN network.
        
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        super(DuelingDQNAgent, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Value stream
        self.value_fc = nn.Linear(128, 1)
        
        # Advantage stream
        self.advantage_fc = nn.Linear(128, action_size)
        
        self.dropout = nn.Dropout(p=0.2)  # Dropout regularization
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network to compute Q-values.
        
        Args:
            x (torch.Tensor): Input state tensor
        
        Returns:
            torch.Tensor: Q-values for each action
        """
        x = torch.relu(self.fc1(x))
        x = self.dropout(torch.relu(self.fc2(x)))  # Apply Dropout
        
        value = self.value_fc(x)
        advantage = self.advantage_fc(x)
        
        # Combining value and advantage streams
        q_value = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_value


class PrioritizedReplayBuffer:
    """
    Replay buffer that samples experiences with prioritization.
    """
    def __init__(self, buffer_size: int, alpha: float = 0.6) -> None:
        """
        Initialize a prioritized replay buffer.
        
        Args:
            buffer_size (int): Maximum size of the buffer
            alpha (float): Prioritization factor (default is 0.6)
        """
        self.buffer_size = buffer_size
        self.alpha = alpha  # Controls how much prioritization is used
        self.buffer = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.position = 0

    def add(self, experience: namedtuple, td_error: float) -> None:
        """
        Add a new experience to the buffer.
        
        Args:
            experience (namedtuple): Experience tuple (state, action, reward, next_state, done)
            td_error (float): TD error to prioritize the experience
        """
        max_priority = max(self.priorities) if self.buffer else 1.0
        self.buffer.append(experience)
        self.priorities.append(max_priority)

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[namedtuple], np.ndarray, np.ndarray]:
        """
        Sample a batch of experiences based on their priorities.
        
        Args:
            batch_size (int): Size of each training batch
            beta (float): Importance-sampling parameter to correct for bias (default is 0.4)
        
        Returns:
            Tuple[List[namedtuple], np.ndarray, np.ndarray]: Batch of experiences, importance weights, and sampled indices
        """
        if len(self.buffer) == 0:
            return [], [], []

        priorities = np.array(self.priorities) ** self.alpha
        sampling_probs = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=sampling_probs)
        experiences = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * sampling_probs[indices]) ** (-beta)
        weights /= weights.max()

        return experiences, weights, indices

    def update_priorities(self, indices: List[int], td_errors: np.ndarray) -> None:
        """
        Update the priorities of sampled experiences in the buffer.
        
        Args:
            indices (List[int]): Indices of the experiences to update
            td_errors (np.ndarray): Updated TD errors for the corresponding experiences
        """
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-5  # Avoid 0 priority


class DQNAgent:
    """
    DQN agent that interacts with the environment and learns from experiences.
    Uses Double DQN, Prioritized Experience Replay, and Dueling DQN.
    """
    def __init__(self, state_size: int, action_size: int, seed: int, 
                 gamma: float = 0.99, lr: float = 5e-4, buffer_size: int = 10000, 
                 batch_size: int = 64) -> None:
        """
        Initialize a DQN agent.
        
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            gamma (float): Discount factor for future rewards (default is 0.99)
            lr (float): Learning rate (default is 5e-4)
            buffer_size (int): Size of replay buffer (default is 10000)
            batch_size (int): Size of each training batch (default is 64)
        """
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

    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Store an experience and learn if enough experiences are collected.
        
        Args:
            state (np.ndarray): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.ndarray): Next state after the action
            done (bool): Whether the episode has terminated
        """
        # Calculate TD error for the new experience
        with torch.no_grad():
            q_target_next = self.qnetwork_target(torch.FloatTensor(next_state).unsqueeze(0).to(self.device)).max(1)[0].unsqueeze(1)
            q_expected = self.qnetwork_local(torch.FloatTensor(state).unsqueeze(0).to(self.device)).gather(1, torch.tensor([[action]]).to(self.device))
            td_error = abs(reward + self.gamma * q_target_next * (1 - done) - q_expected).item()

        # Save experience in replay memory with the TD error as priority
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.add(experience, td_error)

        # Learn every few time steps if enough experiences are available
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0 and len(self.memory.buffer) >= self.batch_size:
            experiences, weights, indices = self.memory.sample(self.batch_size)
            self.learn(experiences, weights, indices)

    def learn(self, experiences: List[namedtuple], weights: np.ndarray, indices: List[int]) -> None:
        """
        Update Q-network using a batch of experiences from the replay buffer.
        
        Args:
            experiences (List[namedtuple]): Batch of experiences
            weights (np.ndarray): Importance-sampling weights for each experience
            indices (List[int]): Indices of the sampled experiences
        """
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

        # Update priorities in replay memory
        td_errors = abs(q_expected - q_targets).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Select an action for the given state based on an epsilon-greedy policy.
        
        Args:
            state (np.ndarray): Current state
            epsilon (float): Epsilon value for epsilon-greedy policy (default is 0.0)
        
        Returns:
            int: Action to be taken
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def soft_update(self, local_model: nn.Module, target_model: nn.Module, tau: float) -> None:
        """
        Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target
        
        Args:
            local_model (nn.Module): Model parameters from the local model
            target_model (nn.Module): Model parameters from the target model
            tau (float): Interpolation parameter (0 <= tau <= 1)
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def update_target_network(self, tau: float = 1.0) -> None:
        """
        Update target network by copying weights from the local network.
        
        Args:
            tau (float): Factor for soft update (default is 1.0, meaning hard update)
        """
        # self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.soft_update(self.qnetwork_local, self.qnetwork_target, tau)
