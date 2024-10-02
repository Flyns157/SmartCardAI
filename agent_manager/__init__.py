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
    #dqn_agent = DQNAgent.from_checkpoint(checkpoint=torch.load(load_checkpoint_path)) # args.load_checkpoint_path

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
