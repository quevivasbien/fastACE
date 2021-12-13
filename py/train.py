"""A toolkit for helping with training.

main.py provides interface with the C++ library, which includes some basic training functionality
This module expands on that training functionality.
"""

import os
from main import RuntimeManager

os.chdir(os.path.dirname(os.path.realpath(__file__)))

SAVE_DIR = '../models/'  # this should be the same path as DEFAULT_SAVE_DIR in src/neural/neuralConstants.h


def move_model_files(origin: str, destination: str):
    files = [f for f in os.listdir(origin) if f.endswith('.pt')]
    if not os.path.isdir(destination):
        os.makedirs(destination)
    for f in files:
        os.rename(os.path.join(origin, f), os.path.join(destination, f))


class Trainer:

    def __init__(self, numPersons: int = 96, numFirms: int = 12, episodeLength: int = 40):
        self.numPersons = numPersons
        self.numFirms = numFirms
        self.episodeLength = episodeLength
    
    def make_attempt(self, fastLR: float, attemptLength: int) -> RuntimeManager:
        mgr = RuntimeManager(self.numPersons, self.numFirms)
        mgr.set_all_lrs(fastLR)
        mgr.train(attemptLength, self.episodeLength, plot=False)
        return mgr

    def explore(self, fastLR: float, numAttempts: int, attemptLength: int) -> RuntimeManager:
        mgrs = []
        # start training a bunch of models
        for i in range(numAttempts):
            print(f'Making exploratory attempt ({i+1} of {numAttempts})')
            mgrs.append(self.make_attempt(fastLR, attemptLength))
            move_model_files(SAVE_DIR, os.path.join(SAVE_DIR, f'attempt{i}'))
        # choose the best model
        best_attempt = min(
            range(numAttempts),
            key=lambda i: sum(mgrs[i].loss_history[-max(attemptLength // 2, 1):])
        )
        move_model_files(os.path.join(SAVE_DIR, f'attempt{best_attempt}'), SAVE_DIR)
        return mgrs[best_attempt]


    def train(self, fastLR: float, slowLR: float, numAttempts: int, attemptLength: int, numEpisodes: int) -> list:
        if fastLR is None:
            fastLR = 1e-3
        if slowLR is None:
            slowLR = 1e-6
        if numAttempts is None:
            numAttempts = 20
        if attemptLength is None:
            attemptLength = 4
        if numEpisodes is None:
            numEpisodes = 1000
        
        mgr = self.explore(fastLR, numAttempts, attemptLength)
        mgr.set_all_lrs(slowLR)
        return mgr.train(numEpisodes, self.episodeLength, fromPretrained=True, warnIfNotSynched=False)



def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train with advanced options')
    parser.add_argument('--npersons', type=int, help='Number of persons in the simulation, default=144')
    parser.add_argument('--nfirms', type=int, help='Number of firms in the simulation, default=24')
    parser.add_argument('--atteps', type=int, help='Number of episodes to run each attempt, default=4')
    parser.add_argument('--episodes', type=int, help='Number of episodes to run for after initial exploration, default=1000')
    parser.add_argument('--eplength', type=int, help='Length of each episode, default=40')
    parser.add_argument('--fast', type=float, help="The learning rate at which to start exploring")
    parser.add_argument('--slow', type=float, help="The learning rate to drop down to after initial exploration")
    parser.add_argument('--attempts', type=int, help="Number of exploratory attempts to make, default=20")
    
    args = parser.parse_args()

    trainer = Trainer()
    if args.npersons is not None:
        trainer.numPersons = args.npersons
    if args.nfirms is not None:
        trainer.numFirms = args.nfirms
    if args.eplength is not None:
        trainer.episodeLength = args.eplength

    trainer.train(args.fast, args.slow, args.attempts, args.atteps, args.episodes)


if __name__ == '__main__':
    main()
    