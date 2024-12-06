"""
This module provides a logger class that can be used to save the running results
and help make plots from the results.
"""

import os
import csv

class Logger:
    ''' Logger saves the running results and helps make plots from the results
    '''
    fieldnames = ['episode', 'reward', 'elapsed_time']

    def __init__(self, log_dir):
        ''' Initialize the labels, legend and paths of the plot and log file.

        Args:
            log_path (str): The path the log files
        '''
        self.log_dir = log_dir
        self.txt_path = os.path.join(self.log_dir, 'log.txt')
        self.csv_path = os.path.join(self.log_dir, 'performance.csv')
        self.fig_path = os.path.join(self.log_dir, 'fig.png')

    def __enter__(self):
        self.txt_path = os.path.join(self.log_dir, 'log.txt')
        self.csv_path = os.path.join(self.log_dir, 'performance.csv')
        self.fig_path = os.path.join(self.log_dir, 'fig.png')

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.txt_file = open(self.txt_path, 'w', encoding='utf-8')
        self.csv_file = open(self.csv_path, 'w', encoding='utf-8')

        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        self.writer.writeheader()

        return self

    def log(self, text):
        ''' Write the text to log file then print it.
        Args:
            text(string): text to log
        '''
        self.txt_file.write(text+'\n')
        self.txt_file.flush()
        print(text)

    def log_performance(self, episode, reward, elapsed_time):
        ''' Log a point in the curve
        Args:
            episode (int): the episode of the current point
            reward (float): the reward of the current point
            elapsed_time (float): the elapsed time of the current point
        '''
        self.writer.writerow({'episode': episode, 'reward': reward, 'elapsed_time': elapsed_time})
        print('')
        self.log('----------------------------------------')
        self.log('  episode      |  ' + str(episode))
        self.log('  reward       |  ' + str(reward))
        self.log('  elapsed time |  ' + str(elapsed_time))
        self.log('----------------------------------------')

    def __exit__(self, type, value, traceback):
        if self.txt_path is not None:
            self.txt_file.close()
        if self.csv_path is not None:
            self.csv_file.close()
        print('\nLogs saved in', self.log_dir)
