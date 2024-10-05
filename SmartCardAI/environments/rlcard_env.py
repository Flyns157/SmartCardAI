import rlcard
from rlcard.utils import set_seed

class RLCardEnv:
    def __init__(self, game, seed=None):
        self.env = rlcard.make(game)
        if seed is not None:
            set_seed(seed)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def get_state_shape(self):
        return self.env.state_shape[0]

    def get_action_shape(self):
        return self.env.num_actions
