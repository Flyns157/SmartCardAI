from SmartCardAI.agents import DQNAgent, NFSPAgent, DMCAgent

dqn_agent = DQNAgent(state_size=100, action_size=10)
nfsp_agent = NFSPAgent(state_size=100, action_size=10)
dmc_agent = DMCAgent(state_size=100, action_size=10)

# dqn_agent.remember(state, action, reward, next_state, done)
dqn_agent.train()
