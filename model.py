import numpy as np
import tensorflow as tf
import random
from collections import namedtuple
from copy import deepcopy

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done', 'legal_actions'])

class DQNAgent:
    def __init__(self,
                 replay_memory_size=20000,
                 replay_memory_init_size=100,
                 update_target_estimator_every=1000,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_steps=20000,
                 batch_size=32,
                 num_actions=2,
                 state_shape=None,
                 train_every=1,
                 mlp_layers=None,
                 learning_rate=0.00005,
                 device=None,
                 save_path=None,
                 save_every=float('inf')):
        
        self.use_raw = False
        self.replay_memory_init_size = replay_memory_init_size
        self.update_target_estimator_every = update_target_estimator_every
        self.discount_factor = discount_factor
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.train_every = train_every

        # Device configuration (TensorFlow uses slightly different device management)
        self.device = device or ('/gpu:0' if tf.test.is_gpu_available() else '/cpu:0')
        
        # Total timesteps
        self.total_t = 0
        
        # Total training step
        self.train_t = 0
        
        # The epsilon decay scheduler
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
        
        # Create estimators
        with tf.device(str(self.device)):
            self.q_estimator = Estimator(
                num_actions=num_actions, 
                learning_rate=learning_rate, 
                state_shape=state_shape, 
                mlp_layers=mlp_layers
            )
            self.target_estimator = Estimator(
                num_actions=num_actions, 
                learning_rate=learning_rate, 
                state_shape=state_shape, 
                mlp_layers=mlp_layers
            )
        
        # Create replay memory
        self.memory = Memory(replay_memory_size, batch_size)
        
        # Checkpoint saving parameters
        self.save_path = save_path
        self.save_every = save_every

    def feed(self, ts):
        (state, action, reward, next_state, done) = tuple(ts)
        self.feed_memory(
            state['obs'], 
            action, 
            reward, 
            next_state['obs'], 
            list(next_state['legal_actions'].keys()), 
            done
        )
        self.total_t += 1
        tmp = self.total_t - self.replay_memory_init_size
        if tmp >= 0 and tmp % self.train_every == 0:
            self.train()

    def step(self, state):
        q_values = self.predict(state)
        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps-1)]
        legal_actions = list(state['legal_actions'].keys())
        
        probs = np.ones(len(legal_actions), dtype=float) * epsilon / len(legal_actions)
        best_action_idx = legal_actions.index(np.argmax(q_values))
        probs[best_action_idx] += (1.0 - epsilon)
        
        action_idx = np.random.choice(np.arange(len(probs)), p=probs)
        return legal_actions[action_idx]

    def eval_step(self, state):
        q_values = self.predict(state)
        best_action = np.argmax(q_values)

        info = {}
        info['values'] = {
            state['raw_legal_actions'][i]: float(q_values[list(state['legal_actions'].keys())[i]]) 
            for i in range(len(state['legal_actions']))
        }

        return best_action, info

    def predict(self, state):
        q_values = self.q_estimator.predict(np.expand_dims(state['obs'], 0))[0]
        masked_q_values = -np.inf * np.ones(self.num_actions, dtype=float)
        legal_actions = list(state['legal_actions'].keys())
        masked_q_values[legal_actions] = q_values[legal_actions]

        return masked_q_values

    def train(self):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, legal_actions_batch = self.memory.sample()

        # Calculate best next actions using Q-network (Double DQN)
        q_values_next = self.q_estimator.predict(next_state_batch)
        legal_actions = []
        for b in range(self.batch_size):
            legal_actions.extend([i + b * self.num_actions for i in legal_actions_batch[b]])
        
        masked_q_values = -np.inf * np.ones(self.num_actions * self.batch_size, dtype=float)
        masked_q_values[legal_actions] = q_values_next.flatten()[legal_actions]
        masked_q_values = masked_q_values.reshape((self.batch_size, self.num_actions))
        best_actions = np.argmax(masked_q_values, axis=1)

        # Evaluate best next actions using Target-network (Double DQN)
        q_values_next_target = self.target_estimator.predict(next_state_batch)
        target_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
            self.discount_factor * q_values_next_target[np.arange(self.batch_size), best_actions]

        # Perform gradient descent update
        state_batch = np.array(state_batch)
        loss = self.q_estimator.update(state_batch, action_batch, target_batch)
        print(f'\rINFO - Step {self.total_t}, rl-loss: {loss}', end='')

        # Update the target estimator
        if self.train_t % self.update_target_estimator_every == 0:
            self.target_estimator.model.set_weights(self.q_estimator.model.get_weights())
            print("\nINFO - Copied model parameters to target network.")

        self.train_t += 1

        if self.save_path and self.train_t % self.save_every == 0:
            self.save_checkpoint(self.save_path)
            print("\nINFO - Saved model checkpoint.")

    def feed_memory(self, state, action, reward, next_state, legal_actions, done):
        self.memory.save(state, action, reward, next_state, legal_actions, done)

    def set_device(self, device):
        self.device = device

    def checkpoint_attributes(self):
        return {
            'agent_type': 'DQNAgent',
            'q_estimator': self.q_estimator.checkpoint_attributes(),
            'memory': self.memory.checkpoint_attributes(),
            'total_t': self.total_t,
            'train_t': self.train_t,
            'epsilon_start': self.epsilons.min(),
            'epsilon_end': self.epsilons.max(),
            'epsilon_decay_steps': self.epsilon_decay_steps,
            'discount_factor': self.discount_factor,
            'update_target_estimator_every': self.update_target_estimator_every,
            'batch_size': self.batch_size,
            'num_actions': self.num_actions,
            'train_every': self.train_every,
            'device': self.device
        }

    @classmethod
    def from_checkpoint(cls, checkpoint):
        print("\nINFO - Restoring model from checkpoint...")
        agent_instance = cls(
            replay_memory_size=checkpoint['memory']['memory_size'],
            update_target_estimator_every=checkpoint['update_target_estimator_every'],
            discount_factor=checkpoint['discount_factor'],
            epsilon_start=checkpoint['epsilon_start'],
            epsilon_end=checkpoint['epsilon_end'],
            epsilon_decay_steps=checkpoint['epsilon_decay_steps'],
            batch_size=checkpoint['batch_size'],
            num_actions=checkpoint['num_actions'], 
            device=checkpoint['device'], 
            state_shape=checkpoint['q_estimator']['state_shape'],
            mlp_layers=checkpoint['q_estimator']['mlp_layers'],
            train_every=checkpoint['train_every']
        )
        
        agent_instance.total_t = checkpoint['total_t']
        agent_instance.train_t = checkpoint['train_t']
        
        agent_instance.q_estimator = Estimator.from_checkpoint(checkpoint['q_estimator'])
        agent_instance.target_estimator = Estimator.from_checkpoint(checkpoint['q_estimator'])
        agent_instance.memory = Memory.from_checkpoint(checkpoint['memory'])
        
        return agent_instance

    def save_checkpoint(self, path, filename='checkpoint_dqn.h5'):
        import os
        os.makedirs(path, exist_ok=True)
        checkpoint = self.checkpoint_attributes()
        self.q_estimator.model.save_weights(os.path.join(path, filename))


class Estimator:
    def __init__(self, num_actions=2, learning_rate=0.00005, state_shape=None, mlp_layers=None):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers or [64, 64]

        self.model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=self.state_shape))
        model.add(tf.keras.layers.BatchNormalization())

        for layer_size in self.mlp_layers:
            model.add(tf.keras.layers.Dense(layer_size, activation='tanh'))
        
        model.add(tf.keras.layers.Dense(self.num_actions, activation='linear'))
        
        model.compile(optimizer=self.optimizer, loss='mse')
        return model

    def predict(self, s):
        return self.model.predict(s)

    def update(self, s, a, y):
        with tf.GradientTape() as tape:
            q_values = self.model(s)
            one_hot_actions = tf.one_hot(a, depth=self.num_actions)
            q_values_for_actions = tf.reduce_sum(q_values * one_hot_actions, axis=1)
            loss = tf.keras.losses.MSE(y, q_values_for_actions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return tf.reduce_mean(loss).numpy()

    def checkpoint_attributes(self):
        return {
            'num_actions': self.num_actions,
            'learning_rate': self.learning_rate,
            'state_shape': self.state_shape,
            'mlp_layers': self.mlp_layers
        }

    @classmethod
    def from_checkpoint(cls, checkpoint):
        estimator = cls(
            num_actions=checkpoint['num_actions'],
            learning_rate=checkpoint['learning_rate'],
            state_shape=checkpoint['state_shape'],
            mlp_layers=checkpoint['mlp_layers']
        )
        return estimator


class Memory:
    def __init__(self, memory_size, batch_size):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []

    def save(self, state, action, reward, next_state, legal_actions, done):
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = Transition(state, action, reward, next_state, done, legal_actions)
        self.memory.append(transition)

    def sample(self):
        samples = random.sample(self.memory, self.batch_size)
        samples = tuple(zip(*samples))
        return tuple(map(np.array, samples[:-1])) + (samples[-1],)

    def checkpoint_attributes(self):
        return {
            'memory_size': self.memory_size,
            'batch_size': self.batch_size,
            'memory': self.memory
        }

    @classmethod
    def from_checkpoint(cls, checkpoint):
        instance = cls(checkpoint['memory_size'], checkpoint['batch_size'])
        instance.memory = checkpoint['memory']
        return instance
