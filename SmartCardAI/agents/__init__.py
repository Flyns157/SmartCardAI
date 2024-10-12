import subprocess
import sys

reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
installed_packages = [r.decode().split('==')[0] for r in reqs.split()]

from .cfr_agent import CFRAgent
from .dqn_agent import DQNAgent
from .dueling_dqn_agent import DuelingDQNAgent
from .nfsp_agent import NFSPAgent
# from .dmc_agent import DMCAgent

__all__ = ['CFRAgent'] + (['DQNAgent', 'DuelingDQNAgent', 'NFSPAgent']if 'torch' in installed_packages else [])
