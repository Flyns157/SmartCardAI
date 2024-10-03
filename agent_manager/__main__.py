import os
import argparse
from __init__ import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN/NFSP in RLCard")
    parser.add_argument(
        '--env_type',
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
        '--dir',
        type=str,
        default='experiments/',
    )
    parser.add_argument(
        '--max_time',
        type=int,
        default=None,
        help='Maximum training time in seconds',
    )
    parser.add_argument(
        '--resume_training',
        type=str,
        default=None,
        help='Resume training from an existing model with the specified name',
    )
    parser.add_argument(
        '--train_against_self',
        action='store_true',
        help='Train the agent against a copy of itself',
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    # print(args.__dict__.keys())
    train(**(args.__dict__))
