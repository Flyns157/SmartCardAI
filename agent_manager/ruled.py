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

class UNORuleAgentV4(object):
    ''' UNO Rule agent version 4 with chaining wild cards and reactive color changes
    '''

    def __init__(self):
        self.use_raw = True
        self.prev_color = None  # To track the previous color of the game

    def step(self, state):
        ''' Step function implementing the strategy

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
        others_hand_size = min([len(hand) for hand in state['others_hand']])
        top_card = state['top_card']
        current_color = top_card[0]  # The color of the top card in play

        # Keep +2 and +4 for defensive use unless absolutely necessary
        for action in legal_actions:
            if 'draw_2' in action or 'wild_draw_4' in action:
                # If the opponent has few cards, play +2 or +4 as an attack
                if others_hand_size <= 2:
                    return action
                # Otherwise, keep them for later
                continue

        # Chain wild cards if multiple are present
        wild_cards = [action for action in legal_actions if 'wild' in action]
        if len(wild_cards) > 1:
            return np.random.choice(wild_cards)  # Chain wild cards if possible

        # Change color if the opponent just changed it
        if self.prev_color and self.prev_color != current_color:
            # Attempt to change to a different color
            available_colors = self.count_colors(self.filter_wild(hand))
            new_color = max(available_colors, key=available_colors.get)
            filtered_actions = [action for action in legal_actions if action.startswith(new_color)]
            if filtered_actions:
                return np.random.choice(filtered_actions)
        
        # Default to matching the dominant color or playing any valid card
        color_nums = self.count_colors(self.filter_wild(hand))
        dominant_color = max(color_nums, key=color_nums.get)
        filtered_actions = [action for action in legal_actions if action.startswith(dominant_color)]
        if filtered_actions:
            return np.random.choice(filtered_actions)

        # Play a random valid action if no better option
        action = np.random.choice(self.filter_wild(legal_actions))

        # Update the previous color
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
    
class UNORuleModelV4(Model):
    ''' UNO Rule Model version 4
    '''

    def __init__(self):
        ''' Initialize environment and load the agent
        '''
        env = rlcard.make('uno')

        # Load the new rule-based agent
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