Bonjour, j'ai besoin de ton aide chère camarade développeur !
Mon but est de créer un modèle d'intelligence artificiel utilisant le deeplearning capable de  jouer au UNO et qui soit la plus performante possible à ce jeux, en effet elle participera à un tournois d'IA. Dans le cadre de la compétition on dois développer en python et utiliser la bibliothèque python "rlcard" pour l'environnement (env = rlcard.make('uno')).
Afin de nous aider, voici des bouts de codes utiles fournis par les organisateurs :
import numpy as np

import rlcard
from rlcard.models.model import Model

class UNORuleAgentV2(object):
    ''' UNO Rule agent version 2
    '''

    def __init__(self):
        self.use_raw = True

    def step(self, state):
        ''' Predict the action given raw state. A naive rule. Choose the color
            that appears least in the hand from legal actions. Try to keep wild
            cards as long as it can.

        Args:
            state (dict): Raw state from the game

        Returns:
            action (str): Predicted action
        '''

        legal_actions = state['raw_legal_actions']
        state = state['raw_obs']
        if 'draw' in legal_actions:
            return 'draw'

        hand = state['hand']

        # If we have wild-4 simply play it and choose color that appears most in hand
        for action in legal_actions:
            if action.split('-')[1] == 'wild_draw_4':
                color_nums = self.count_colors(self.filter_wild(hand))
                action = max(color_nums, key=color_nums.get) + '-wild_draw_4'
                return action

        # Without wild-4, we randomly choose one
        action = np.random.choice(self.filter_wild(legal_actions))
        return action

    def eval_step(self, state):
        ''' Step for evaluation. The same to step
        '''
        return self.step(state), []

    @staticmethod
    def filter_wild(hand):
        ''' Filter the wild cards. If all are wild cards, we do not filter

        Args:
            hand (list): A list of UNO card string

        Returns:
            filtered_hand (list): A filtered list of UNO string
        '''
        filtered_hand = []
        for card in hand:
            if not card[2:6] == 'wild':
                filtered_hand.append(card)

        if len(filtered_hand) == 0:
            filtered_hand = hand

        return filtered_hand

    @staticmethod
    def count_colors(hand):
        ''' Count the number of cards in each color in hand

        Args:
            hand (list): A list of UNO card string

        Returns:
            color_nums (dict): The number cards of each color
        '''
        color_nums = {}
        for card in hand:
            color = card[0]
            if color not in color_nums:
                color_nums[color] = 0
            color_nums[color] += 1

        return color_nums

class UNORuleModelV2(Model):
    ''' UNO Rule Model version 2
    '''

    def __init__(self):
        ''' Load pretrained model
        '''
        env = rlcard.make('uno')

        rule_agent = UNORuleAgentV2()
        self.rule_agents = [rule_agent for _ in range(env.num_players)]

    @property
    def agents(self):
        ''' Get a list of agents for each position in a the game

        Returns:
            agents (list): A list of agents

        Note: Each agent should be just like RL agent with step and eval_step
              functioning well.
        '''
        return self.rule_agents

    @property
    def use_raw(self):
        ''' Indicate whether use raw state and action

        Returns:
            use_raw (boolean): True if using raw state and action
        '''
        return True

import rlcard
from rlcard import models
from rlcard.agents.human_agents.uno_human_agent import HumanAgent, _print_action

# Make environment
env = rlcard.make('uno')
human_agent = HumanAgent(env.num_actions)
rule_agent = UNORuleModelV2().agents[0]
env.set_agents([
    human_agent,
    rule_agent,
])

while (True):
    print(">> Start a new game")

    trajectories, payoffs = env.run(is_training=False)
    # If the human does not take the final action, we need to
    # print other players action
    final_state = trajectories[0][-1]
    action_record = final_state['action_record']
    state = final_state['raw_obs']
    _action_list = []
    for i in range(1, len(action_record)+1):
        if action_record[-i][0] == state['current_player']:
            break
        _action_list.insert(0, action_record[-i])
    for pair in _action_list:
        print('>> Player', pair[0], 'chooses ', end='')
        _print_action(pair[1])
        print('')

    print('===============     Result     ===============')
    if payoffs[0] > 0:
        print('You win!')
    else:
        print('You lose!')
    print('')
    user_input = input("Press any key to continue...")
    if user_input == '':
        break

# TODO
import rlcard
from rlcard.agents.random_agent import RandomAgent
from rlcard.models.uno_rule_models import UNORuleModelV1

def evaluate_agents(num_games=10000):
    # Créer l'environnement pour le jeu Uno
    env = rlcard.make('uno')

    # Initialiser l'agent basé sur des règles
    rule_agent = UNORuleModelV2().agents[0]

    # Initialiser un agent aléatoire
    random_agent = RandomAgent(num_actions=env.num_actions)

    # Associer les agents à l'environnement
    env.set_agents([rule_agent, random_agent])

    # Variables pour compter les résultats
    rule_agent_wins = 0
    random_agent_wins = 0

    # Lancer les parties
    for _ in range(num_games):
        # Exécuter une partie
        trajectories, payoffs = env.run(is_training=False)

        # Le payoff du premier joueur correspond à l'agent basé sur des règles
        if payoffs[0] > 0:
            rule_agent_wins += 1
        else:
            random_agent_wins += 1

    # Afficher les résultats finaux
    print(f"Après {num_games} parties :")
    print(f"Agent basé sur des règles a gagné {rule_agent_wins} fois")
    print(f"Agent aléatoire a gagné {random_agent_wins} fois")
    print(f"Taux de victoire de l'agent basé sur des règles : {rule_agent_wins / num_games:.2%}")
    print(f"Taux de victoire de l'agent aléatoire : {random_agent_wins / num_games:.2%}")

# Évaluer les agents sur 1000 parties
evaluate_agents(num_games=1000)

# Également un exemple d'agent "DQN" nous a été fournis :

import os
import argparse

import torch

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)

def train(args):

    # Check whether gpu is available
    device = get_device()

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(
        args.env,
        config={
            'seed': args.seed,
        }
    )

    # Initialize the agent and use random agents as opponents
    if args.algorithm == 'dqn':
        from rlcard.agents import DQNAgent
        agent = DQNAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            mlp_layers=[64,64],
            device=device,
        )
    elif args.algorithm == 'nfsp':
        from rlcard.agents import NFSPAgent
        agent = NFSPAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            hidden_layers_sizes=[64,64],
            q_mlp_layers=[64,64],
            device=device,
        )
    agents = [agent]
    for _ in range(1, env.num_players):
        agents.append(RandomAgent(num_actions=env.num_actions))
    env.set_agents(agents)

    # Start training
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):

            if args.algorithm == 'nfsp':
                agents[0].sample_episode_policy()

            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions into agent memory, and train the agent
            # Here, we assume that DQN always plays the first position
            # and the other players play randomly (if any)
            for ts in trajectories[0]:
                agent.feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % args.evaluate_every == 0:
                logger.log_performance(
                    episode,
                    tournament(
                        env,
                        args.num_eval_games,
                    )[0]
                )

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    plot_curve(csv_path, fig_path, args.algorithm)

    # Save model
    save_path = os.path.join(args.log_dir, 'model.pth')
    torch.save(agent, save_path)
    print('Model saved in', save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN/NFSP example in RLCard")
    parser.add_argument(
        '--env',
        type=str,
        default='leduc-holdem',
        choices=[
            'blackjack',
            'leduc-holdem',
            'limit-holdem',
            'doudizhu',
            'mahjong',
            'no-limit-holdem',
            'uno',
            'gin-rummy',
            'bridge',
        ],
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        default='dqn',
        choices=[
            'dqn',
            'nfsp',
        ],
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=5000,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=2000,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/leduc_holdem_dqn_result/',
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)

J'aimerais mettre en place un agent DQN, par ailleurs j'avais pensé à faire en sorte d'utiliser un système de "génération", c'est à dire qu'a chaque étape de l'entrainement, on crée des variations de notre agent que l'on met en concurrence et l'on ne garde que les meilleurs pour la génération suivante. Le but serait d'avoir plusieurs "techniques" explorées par l'IA lors de son entrainement.
