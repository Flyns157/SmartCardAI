import rlcard
from rlcard import models
from rlcard.agents import RandomAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)
from rlcard.agents.human_agents.uno_human_agent import _print_action

import os
import torch
import random
import time

def load_model(model_path, env=None, position=None, device=None):
    if os.path.isfile(model_path):  # Torch model
        import torch
        agent = torch.load(model_path, map_location=device)
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

    return agent

def evaluate_agents(agent, agent_bis = None, num_games=10000):
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

def lets_play_uno(first_agent, second_agent):
    # Make environment
    env = rlcard.make('uno')
    # human_agent = HumanAgent(env.num_actions)
    # https://github.com/datamllab/rlcard/blob/master/rlcard/models/__init__.py
    # rule_agent = models.load('uno-rule-v1').agents[0]

    # load_checkpoint_path = "experiments/uno_dqn_result/model.pth"
    # dqn_agent=torch.load(load_checkpoint_path)
    #load_checkpoint_path = "experiments/uno_dqn_result/checkpoint_dqn.pt"
    #dqn_agent = DQNAgent.from_checkpoint(checkpoint=torch.load(load_checkpoint_path)) # load_checkpoint_path

    env.set_agents([
        first_agent,
        second_agent
    ])

    for i in range(2):
        print(">> Start a new game")

        trajectories, payoffs = env.run(is_training=False)
        # If the human does not take the final action, we need to
        # print other players action
        final_state = trajectories[0][-1]
        action_record = final_state['action_record']
        state = final_state['raw_obs']
        _action_list = []
        for i in range(1, len(action_record)+1):
            #if action_record[-i][0] == state['current_player']:
            #    break
            _action_list.insert(0, action_record[-i])
        for pair in _action_list:
            print('>> Player', pair[0], 'chooses ', end='')
            _print_action(pair[1])
            print('')

        print('===============     Result     ===============')
        if payoffs[0] > 0:
            print('Player 0 win!')
            print(type(env.agents[0]))
        else:
            print('Player 1 win!')
            print(type(env.agents[1]))
        print('')
        #input("Press any key to continue...")











def train(env_type:str, algorithm:str, seed:int, num_episodes:int = 5000, num_eval_games:int = 2000, evaluate_every:int = 100, log_dir:str = 'experiments/', max_time:int = 600, resume_training:bool = False, train_against_self:bool = False, *args, **kwargs):

    # Check whether gpu is available
    device = get_device()

    # Seed numpy, torch, random
    set_seed(seed)

    # Make the environment with seed
    env = rlcard.make(
        env_type,
        config={
            'seed': seed,
        }
    )

    # Initialize the agent and use random agents as opponents
    if algorithm == 'dqn':
        from rlcard.agents import DQNAgent
        agent = DQNAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            mlp_layers=[64,64],
            device=device,
        )
    elif algorithm == 'nfsp':
        from rlcard.agents import NFSPAgent
        agent = NFSPAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            hidden_layers_sizes=[64,64],
            q_mlp_layers=[64,64],
            device=device,
        )
    
    # Load existing model if resume_training is True
    model_path = os.path.join(log_dir, 'model.pth')
    if resume_training and os.path.exists(model_path):
        agent = torch.load(model_path)
        print(f'Model loaded from {model_path}')
    
    agents = [agent]
    if train_against_self:
        for _ in range(1, env.num_players):
            agents.append(agent)
    else:
        for _ in range(1, env.num_players):
            agents.append(RandomAgent(num_actions=env.num_actions))
            # agents.append(UNORuleModelV2().agents[0])
    
    # Shuffle agents to choose the starting agent randomly
    random.shuffle(agents)
    env.set_agents(agents)

    # Start training
    start_time = time.time()
    with Logger(log_dir) as logger:
        for episode in range(num_episodes):

            if max_time and (time.time() - start_time) > max_time:
                break

            if algorithm == 'nfsp':
                agents[0].sample_episode_policy()

            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions into agent memory, and train the agent
            for ts in trajectories[0]:
                agent.feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % evaluate_every == 0:
                logger.log_performance(
                    episode,
                    tournament(
                        env,
                        num_eval_games,
                    )[0]
                )

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    plot_curve(csv_path, fig_path, algorithm)

    # Save model
    save_path = os.path.join(log_dir, 'model.pth')
    torch.save(agent, save_path)
    print('Model saved in', save_path)
