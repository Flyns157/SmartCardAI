import numpy as np

import rlcard
from rlcard.models.model import Model
from collections import deque
 

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


class UNORuleAgentV4(object):
    ''' UNO Rule agent version 6 with enhanced strategies:
        - Defensive use of +2 and +4
        - Chaining wild cards
        - Reactive color changes
        - Detect and counter frequently played colors by opponents
        - Prioritize playing the most abundant color in hand
    '''

    def __init__(self):
        self.use_raw = True
        self.prev_color = None  # To track the previous color of the game
        self.color_history = deque(maxlen=3)  # To track the last 3 colors played

    def step(self, state):
        ''' Step function implementing the enhanced strategy

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
        others_hand_sizes = [len(h) for h in state['others_hand']]
        min_others_hand_size = min(others_hand_sizes) if others_hand_sizes else 0
        top_card = state['top_card']
        current_color = top_card[0]  # The color of the top card in play

        # Update color history
        if current_color in ['R', 'G', 'B', 'Y']:
            self.color_history.append(current_color)

        # 1. Défense : Garder +2 et +4 sauf si adversaire proche de gagner
        for action in legal_actions:
            if 'draw_2' in action or 'wild_draw_4' in action:
                if min_others_hand_size <= 2:
                    return action

        # 2. Utiliser les jokers pour enchaîner
        wild_actions = [action for action in legal_actions if 'wild' in action and 'draw' not in action]
        if wild_actions:
            return np.random.choice(wild_actions)

        # 3. Détecter les couleurs surutilisées par les adversaires
        overused_color = self.detect_overused_color(state)
        if overused_color:
            # Tenter de changer de couleur pour contrer
            available_colors = self.count_colors(self.filter_wild(hand))
            # Choisir la couleur la moins courante dans la main
            new_color = self.least_common_color(available_colors)
            filtered_actions = [action for action in legal_actions if action.startswith(new_color)]
            if filtered_actions:
                return np.random.choice(filtered_actions)
            # Si impossible, continuer

        # 4. Changer de couleur en réponse à l'adversaire
        if self.prev_color and self.prev_color != current_color:
            # Tenter de changer à une couleur différente
            available_colors = self.count_colors(self.filter_wild(hand))
            new_color = self.most_common_color(available_colors)
            filtered_actions = [action for action in legal_actions if action.startswith(new_color)]
            if filtered_actions:
                return np.random.choice(filtered_actions)

        # 5. Prioriser la couleur dominante dans la main
        color_nums = self.count_colors(self.filter_wild(hand))
        if color_nums:
            dominant_color = self.most_common_color(color_nums)
            filtered_actions = [action for action in legal_actions if action.startswith(dominant_color)]
            if filtered_actions:
                return np.random.choice(filtered_actions)

        # 6. Jouer une carte Wild si aucune autre option n'est meilleure
        if wild_actions:
            return np.random.choice(wild_actions)

        # 7. Jouer une carte aléatoire restante
        action = np.random.choice(self.filter_wild(legal_actions))

        # Mettre à jour la couleur précédente
        self.prev_color = current_color
        return action

    def eval_step(self, state):
        ''' Step for evaluation (same as step)
        '''
        return self.step(state), []

    @staticmethod
    def filter_wild(hand):
        ''' Filter the wild cards from the hand (unless all cards are wild)

        Args:
            hand (list): A list of UNO card strings

        Returns:
            filtered_hand (list): A filtered list of UNO card strings
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
        ''' Count the number of cards of each color in hand

        Args:
            hand (list): A list of UNO card strings

        Returns:
            color_nums (dict): A dictionary of color counts
        '''
        color_nums = {}
        for card in hand:
            color = card[0]
            if color not in color_nums:
                color_nums[color] = 0
            color_nums[color] += 1

        return color_nums

    def detect_overused_color(self, state):
        ''' Detect if a color is being overused by opponents

        Args:
            state (dict): Raw state from the game

        Returns:
            overused_color (str or None): The color that is overused, or None
        '''
        # Analyse la color_history pour détecter une surutilisation
        color_counts = {}
        for color in self.color_history:
            if color not in color_counts:
                color_counts[color] = 0
            color_counts[color] += 1

        # Définir un seuil pour considérer une couleur comme surutilisée
        for color, count in color_counts.items():
            if count >= 2:  # Seuil de 2 fois
                return color

        return None

    def least_common_color(self, available_colors):
        ''' Choisir la couleur la moins courante dans la main

        Args:
            available_colors (dict): Un dictionnaire des couleurs et leur nombre

        Returns:
            color (str): La couleur la moins courante
        '''
        if not available_colors:
            return np.random.choice(['R', 'G', 'B', 'Y'])  # Choix aléatoire si pas de cartes

        # Retourner la couleur avec le moins de cartes
        return min(available_colors, key=available_colors.get)

    def most_common_color(self, available_colors):
        ''' Choisir la couleur la plus courante dans la main

        Args:
            available_colors (dict): Un dictionnaire des couleurs et leur nombre

        Returns:
            color (str): La couleur la plus courante
        '''
        if not available_colors:
            return np.random.choice(['R', 'G', 'B', 'Y'])  # Choix aléatoire si pas de cartes

        # Retourner la couleur avec le plus de cartes
        return max(available_colors, key=available_colors.get)

class UNORuleModelV4(Model):
    ''' UNO Rule Model version 6
    '''

    def __init__(self):
        ''' Initialize environment and load the agent
        '''
        env = rlcard.make('uno')

        # Charger l'agent amélioré
        rule_agent = UNORuleAgentV4()
        self.rule_agents = [rule_agent for _ in range(env.num_players)]

    @property
    def agents(self):
        ''' Get a list of agents for each position in the game

        Returns:
            agents (list): A list of agents
        '''
        return self.rule_agents

    @property
    def use_raw(self):
        ''' Indicate whether to use raw state and action

        Returns:
            use_raw (boolean): True if using raw state and action
        '''
        return True