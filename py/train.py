"""A toolkit for helping with training.

main.py provides interface with the C++ library, which includes some basic training functionality
This module expands on that training functionality.
"""

import matplotlib.pyplot as plt
import typing
import os
from shutil import copyfile
from main import RuntimeManager

os.chdir(os.path.dirname(os.path.realpath(__file__)))

SAVE_DIR = '../models/'  # this should be the same path as DEFAULT_SAVE_DIR in src/neural/neuralConstants.h

N_PERSONS = 48
N_FIRMS = 12
EPISODE_LENGTH = 40

# Parameters for training as a swarm
SWARM_SIZE = 10
N_EPISODES = 100
INITIAL_PERTURBATION = 0.1
PERTURBATION_DECAY = 0.8
SWARM_ITERATIONS = 10
LR = 1e-5
N_TRIAL_EPISODES = 5
TRIAL_MARGIN = 1e4

# Parameters for training with an initial swarm, then single model
N_EXPLORATORY = 0
EXPLORATORY_EPISODES = 4
TRAIN_EPISODES = 500
FAST_LR = 1e-3
SLOW_LR = 1e-6

COLORS = [
    'tab:blue',
    'tab:orange',
    'tab:green',
    'tab:red',
    'tab:purple',
    'tab:brown',
    'tab:pink',
    'tab:gray',
    'tab:olive',
    'tab:cyan'
]


def move_model_files(origin: str, destination: str, copy=False):
    files = [f for f in os.listdir(origin) if f.endswith('.pt')]
    if not os.path.isdir(destination):
        os.makedirs(destination)
    move = copyfile if copy else os.rename
    for f in files:
        move(os.path.join(origin, f), os.path.join(destination, f))


class Trainer:

    def __init__(self, numPersons: int = N_PERSONS, numFirms: int = N_FIRMS, episodeLength: int = 40):
        self.numPersons = numPersons
        self.numFirms = numFirms
        self.episodeLength = episodeLength
    
    def make_attempt(self, lr: float, attemptLength: int) -> typing.Tuple[RuntimeManager, float]:
        mgr = RuntimeManager(self.numPersons, self.numFirms)
        mgr.set_all_lrs(lr)
        mgr.train(attemptLength, self.episodeLength, plot=False)
        loss_score = sum(mgr.loss_history[-max(attemptLength // 2, 1):])
        return mgr, loss_score

    def explore(self, lr: float, numExploratoryAttempts: int, attemptLength: int) -> RuntimeManager:
        mgrs = []
        loss_scores = []
        # start training a bunch of models
        for i in range(numExploratoryAttempts):
            print(f'Making exploratory attempt ({i+1} of {numExploratoryAttempts})')
            mgr, loss_score = self.make_attempt(lr, attemptLength)
            print(f'Attempt {i}: loss score = {loss_score:.3e}')
            mgrs.append(mgr)
            loss_scores.append(loss_score)
            move_model_files(SAVE_DIR, os.path.join(SAVE_DIR, f'attempt{i}'))
        # choose the best model
        best_attempt = min(range(numExploratoryAttempts), key=lambda i: loss_scores[i])
        move_model_files(os.path.join(SAVE_DIR, f'attempt{best_attempt}'), SAVE_DIR, copy=True)
        return mgrs[best_attempt]
    
    def _run_swarm_member(self, numEpisodes: int, perturbationSize: float, target_score: float):
        move_model_files(os.path.join(SAVE_DIR, 'host'), SAVE_DIR, copy=True)
        mgr = RuntimeManager(self.numPersons, self.numFirms)
        mgr.load_settings()
        # train for just a few episodes at first to see if this is a good init
        print('Running trial...')
        n_trial_episodes = min(numEpisodes, N_TRIAL_EPISODES)
        mgr.train(
            n_trial_episodes,
            self.episodeLength,
            fromPretrained=True,
            saveSettings=False,
            perturbationSize=perturbationSize,
            plot=False
        )
        if (
            sum(mgr.loss_history) / n_trial_episodes > target_score + TRIAL_MARGIN
            or numEpisodes == n_trial_episodes
        ):
            print('Trial failed')
            return mgr
        # otherwise, off to a good start, so keep training
        print('Trial passed')
        mgr.train(
            numEpisodes - n_trial_episodes,
            self.episodeLength,
            fromPretrained=True,
            warnIfNotSynched=False,
            saveSettings=False,
            plot=False
        )
        return mgr
    
    def _train_as_swarm(
        self,
        swarmSize: int,
        numEpisodes: int,
        perturbationSize: float,
        iter: int = 0,
        target_score: float = float('inf')
    ):
        move_model_files(SAVE_DIR, os.path.join(SAVE_DIR, 'host'))
        mgrs = []
        loss_scores = []
        for i in range(swarmSize):
            print(f'\nRunning swarm member {i+1} of {swarmSize}')
            mgr = self._run_swarm_member(numEpisodes, perturbationSize, target_score)
            mgr.plot_loss_history(
                show=False,
                scatter_label=None,
                line_label=None,
                scatter_color=COLORS[i % len(COLORS)],
                line_color=COLORS[i % len(COLORS)]
            )
            mgrs.append(mgr)
            # calculate loss score as average of last ceil(numEpisodes / 2) episode losses
            period_for_score = max(len(mgr.loss_history) // 2, 1)
            loss_score = sum(mgr.loss_history[-period_for_score:]) / period_for_score
            loss_scores.append(loss_score)
            print(f'Attempt {i + 1}: loss score = {loss_score:.3e}')
            move_model_files(SAVE_DIR, os.path.join(SAVE_DIR, f'attempt{i}'))
        # plot swarm history
        plt.xlabel('episode')
        plt.ylabel('loss')
        # plt.legend()
        plt.savefig(f'swarm{iter}.png')
        plt.close()
        # choose the best model
        best_attempt = min(range(swarmSize), key=lambda i: loss_scores[i])
        move_model_files(os.path.join(SAVE_DIR, f'attempt{best_attempt}'), SAVE_DIR, copy=True)
        best_mgr = mgrs[best_attempt]
        best_mgr.save_settings()
        return best_mgr, loss_scores[best_attempt]

    def train_as_swarm(
        self,
        swarmSize: int,
        numEpisodes: int,
        lr: float,
        initialPerturbation: float,
        perturbationDecay: float,
        numIterations: int
    ):
        if swarmSize is None:
            swarmSize = SWARM_SIZE
        if numEpisodes is None:
            numEpisodes = N_EPISODES
        if lr is None:
            lr = LR
        if initialPerturbation is None:
            initialPerturbation = INITIAL_PERTURBATION
        if perturbationDecay is None:
            perturbationDecay = PERTURBATION_DECAY
        if numIterations is None:
            numIterations = SWARM_ITERATIONS

        print('Initializing...')
        mgr = RuntimeManager(self.numPersons, self.numFirms)
        mgr.set_all_lrs(lr)
        mgr.train(1, plot=False)
        best_score = mgr.loss_history[-1]
        for i in range(numIterations):
            print(f'\n---\nRUNNING SWARM ITERATION {i+1} OF {numIterations}\n---')
            mgr, best_score = self._train_as_swarm(
                swarmSize, numEpisodes, initialPerturbation * perturbationDecay**i, i, best_score
            )
        return mgr

    def train(self, fastLR: float, slowLR: float, numExploratoryAttempts: int, attemptLength: int, numEpisodes: int) -> list:
        if fastLR is None:
            fastLR = FAST_LR
        if slowLR is None:
            slowLR = SLOW_LR
        if numExploratoryAttempts is None:
            numExploratoryAttempts = N_EXPLORATORY
        if attemptLength is None:
            attemptLength = EXPLORATORY_EPISODES
        if numEpisodes is None:
            numEpisodes = TRAIN_EPISODES
        
        if numExploratoryAttempts > 0:
            mgr = self.explore(fastLR, numExploratoryAttempts, attemptLength)
        else:
            mgr = RuntimeManager(self.numPersons, self.numFirms)
        mgr.set_all_lrs(slowLR)
        return mgr.train(numEpisodes, self.episodeLength, fromPretrained=numExploratoryAttempts > 0, warnIfNotSynched=False)



# def main():
#     import argparse
#     parser = argparse.ArgumentParser(description='Train with advanced options')
#     parser.add_argument('--npersons', type=int, help=f'Number of persons in the simulation, default={N_PERSONS}')
#     parser.add_argument('--nfirms', type=int, help='Number of firms in the simulation, default={N_FIRMS}')
#     parser.add_argument('--expeps', type=int, help='Number of episodes to run each attempt, default={EXPLORATORY_EPISODES}')
#     parser.add_argument('--episodes', type=int, help='Number of episodes to run for after initial exploration, default={N_EPISODES}')
#     parser.add_argument('--eplength', type=int, help='Length of each episode, default={EPISODE_LENGTH}')
#     parser.add_argument('--fast', type=float, help="The learning rate at which to start exploring, default={FAST_LR}")
#     parser.add_argument('--slow', type=float, help="The learning rate to drop down to after initial exploration, default={SLOW_LR}")
#     parser.add_argument('--attempts', type=int, help="Number of exploratory attempts to make, default={N_EXPLORATORY}")
    
#     args = parser.parse_args()

#     trainer = Trainer()
#     if args.npersons is not None:
#         trainer.numPersons = args.npersons
#     if args.nfirms is not None:
#         trainer.numFirms = args.nfirms
#     if args.eplength is not None:
#         trainer.episodeLength = args.eplength

#     trainer.train(args.fast, args.slow, args.attempts, args.expeps, args.episodes)


if __name__ == '__main__':
    # main()
    trainer = Trainer()
    mgr = trainer.train_as_swarm(None, None, None, None, None, None)
    mgr.plot_loss_history()
    