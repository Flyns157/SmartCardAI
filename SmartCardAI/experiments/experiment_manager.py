class ExperimentManager:
    def __init__(self, agent, env, episodes=1000, batch_size=32):
        self.agent = agent
        self.env = env
        self.episodes = episodes
        self.batch_size = batch_size

    def run(self):
        for episode in range(self.episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
            if episode % 10 == 0:
                print(f"Episode {episode}, Total reward: {total_reward}")
            self.agent.replay(self.batch_size)
