import ctypes
import os
import json
import numpy as np
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.realpath(__file__)))


lib = ctypes.CDLL("../bin/libpybindings.so")

class CustomScenarioParams(ctypes.Structure):
    _fields_ = [
        ('numPeople', ctypes.c_uint),
        ('numFirms', ctypes.c_uint),
        ('money_mu', ctypes.c_double),
        ('money_sigma', ctypes.c_double),
        ('good1_mu', ctypes.c_double),
        ('good1_sigma', ctypes.c_double),
        ('good2_mu', ctypes.c_double),
        ('good2_sigma', ctypes.c_double),
        ('labor_share_mu', ctypes.c_double),
        ('labor_share_sigma', ctypes.c_double),
        ('good1_share_mu', ctypes.c_double),
        ('good1_share_sigma', ctypes.c_double),
        ('good2_share_mu', ctypes.c_double),
        ('good2_share_sigma', ctypes.c_double),
        ('discount_mu', ctypes.c_double),
        ('discount_sigma', ctypes.c_double),
        ('elasticity_mu', ctypes.c_double),
        ('elasticity_sigma', ctypes.c_double),
        ('firm_money_mu', ctypes.c_double),
        ('firm_money_sigma', ctypes.c_double),
        ('firm_good1_mu', ctypes.c_double),
        ('firm_good1_sigma', ctypes.c_double),
        ('firm_good2_mu', ctypes.c_double),
        ('firm_good2_sigma', ctypes.c_double),
        ('firm_tfp1_mu', ctypes.c_double),
        ('firm_tfp1_sigma', ctypes.c_double),
        ('firm_tfp2_mu', ctypes.c_double),
        ('firm_tfp2_sigma', ctypes.c_double),
        ('firm_labor_share1_mu', ctypes.c_double),
        ('firm_labor_share1_sigma', ctypes.c_double),
        ('firm_good1_share1_mu', ctypes.c_double),
        ('firm_good1_share1_sigma', ctypes.c_double),
        ('firm_good2_share1_mu', ctypes.c_double),
        ('firm_good2_share1_sigma', ctypes.c_double),
        ('firm_labor_share2_mu', ctypes.c_double),
        ('firm_labor_share2_sigma', ctypes.c_double),
        ('firm_good1_share2_mu', ctypes.c_double),
        ('firm_good1_share2_sigma', ctypes.c_double),
        ('firm_good2_share2_mu', ctypes.c_double),
        ('firm_good2_share2_sigma', ctypes.c_double),
        ('firm_elasticity1_mu', ctypes.c_double),
        ('firm_elasticity1_sigma', ctypes.c_double),
        ('firm_elasticity2_mu', ctypes.c_double),
        ('firm_elasticity2_sigma', ctypes.c_double)
    ]


class TrainingParams(ctypes.Structure):
    _fields_ = [
        ('numEpisodes', ctypes.c_uint),
        ('episodeLength', ctypes.c_uint),
        ('updateEveryNEpisodes', ctypes.c_uint),
        ('checkpointEveryNEpisodes', ctypes.c_uint),
        ('stackSize', ctypes.c_uint),
        ('encodingSize', ctypes.c_uint),
        ('hiddenSize', ctypes.c_uint),
        ('nHidden', ctypes.c_uint),
        ('nHiddenSmall', ctypes.c_uint),
        ('purchaseNetLR', ctypes.c_double),
        ('firmPurchaseNetLR', ctypes.c_double),
        ('laborSearchNetLR', ctypes.c_double),
        ('consumptionNetLR', ctypes.c_double),
        ('productionNetLR', ctypes.c_double),
        ('offerNetLR', ctypes.c_double),
        ('jobOfferNetLR', ctypes.c_double),
        ('valueNetLR', ctypes.c_double),
        ('firmValueNetLR', ctypes.c_double),
        ('episodeBatchSizeForLRDecay', ctypes.c_uint),
        ('patienceForLRDecay', ctypes.c_uint),
        ('multiplierForLRDecay', ctypes.c_double),
        ('reverseAnnealingPeriod', ctypes.c_uint)
    ]


lib.create_scenario_params.argtypes = [
    ctypes.c_uint,
    ctypes.c_uint
]
lib.create_scenario_params.restype = CustomScenarioParams

lib.create_training_params.restype = TrainingParams

lib.run.argtypes = [
    CustomScenarioParams,
    TrainingParams
]
lib.run.restype = None

lib.train.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(CustomScenarioParams),
    ctypes.POINTER(TrainingParams),
    ctypes.c_bool,
    ctypes.c_double
]
lib.train.restype = None


def train(
    scenarioParams: CustomScenarioParams,
    trainingParams: TrainingParams,
    fromPretrained=False,
    perturbationSize=0.0
) -> list:
    losses = ctypes.ARRAY(ctypes.c_double, trainingParams.numEpisodes)()
    lib.train(
        losses,
        ctypes.POINTER(CustomScenarioParams)(scenarioParams),
        ctypes.POINTER(TrainingParams)(trainingParams),
        fromPretrained,
        perturbationSize
    )
    return list(losses)


def dict_from_cstruct(struct):
    fields = [field[0] for field in getattr(struct, '_fields_')]
    return {field: getattr(struct, field) for field in fields}


def moving_average(arr, binsize):
    out = np.cumsum(arr)
    first_elems = out[:binsize-1] / np.arange(1, min(binsize-1, len(arr)) + 1)
    if binsize > len(arr):
        first_elems
    out[binsize:] = out[binsize:] - out[:-binsize]
    return np.concatenate((first_elems, out[binsize-1:] / binsize))


class RuntimeManager:

    def __init__(
        self,
        numPersons: int,
        numFirms: int,
        # saveDir: str
    ):
        self.scenarioParams = lib.create_scenario_params(
            numPersons,
            numFirms
        )
        self.trainingParams = lib.create_training_params()
        self.loss_history = []

        if not os.path.isfile('settings.json'):
            self.save_settings()
    
    def edit_scenario_params(self, attr: str, new_value):
        setattr(self.scenarioParams, attr, new_value)
    
    def view_scenario_params(self):
        return dict_from_cstruct(self.scenarioParams)

    def edit_training_params(self, attr: str, new_value):
        setattr(self.trainingParams, attr, new_value)
    
    def view_training_params(self):
        return dict_from_cstruct(self.trainingParams)
    
    def set_all_lrs(self, new_lr: float):
        for attr in self.view_training_params().keys():
            if attr.endswith('LR'):
                self.edit_training_params(attr, new_lr)
    
    def save_settings(self):
        settings = {
            'scenarioParams': self.view_scenario_params(),
            'trainingParams': self.view_training_params()
        }
        with open('settings.json', 'w') as fh:
            json.dump(settings, fh)
    
    def load_settings(self, set=True):
        with open('settings.json', 'r') as fh:
            settings = json.load(fh)
        if set:
            for key, value in settings['scenarioParams'].items():
                self.edit_scenario_params(key, value)
            for key, value in settings['trainingParams'].items():
                self.edit_training_params(key, value)
        return settings

    def settings_synched(self) -> bool:
        settings = self.load_settings(False)
        synched = (
            settings['scenarioParams'] == self.view_scenario_params()
            and settings['trainingParams'] == self.view_training_params()
        )
        if not synched:
            print(
                "You're trying to load a model that doesn't match the settings you're currently using.\n\n"
                "What do you want to do?\n"
                "Abort [1/default]\n"
                "run .load_settings() first [2]\n"
                "continue without loading (may cause errors) [3]\n"
            )
            action = input('Type a number:\n> ')
            if action == '2':
                self.load_settings()
                return True
            elif action == '3':
                return True
            else:
                return False
        return True
    
    def set_episode_params(
        self,
        numEpisodes=None,
        episodeLength=None,
        updateEveryNEpisodes=None,
        checkpointEveryNEpisodes=None
    ):
        if numEpisodes is not None:
            self.edit_training_params('numEpisodes', numEpisodes)
        if episodeLength is not None:
            self.edit_training_params('episodeLength', episodeLength)
        if updateEveryNEpisodes is not None:
            self.edit_training_params('updateEveryNEpisodes', updateEveryNEpisodes)
        if checkpointEveryNEpisodes is not None:
            self.edit_training_params('checkpointEveryNEpisodes', checkpointEveryNEpisodes)
    
    def plot_loss_history(
        self,
        slice=0,
        moving_avg_binsize=10,
        show=True,
        scatter_label='episode loss',
        line_label='moving average',
        scatter_color=None,
        line_color=None
    ):
        loss_history = self.loss_history[slice:]
        plt.plot(loss_history, marker='.', linestyle='', alpha=0.5, label=scatter_label, color=scatter_color)
        moving_avg = moving_average(loss_history, moving_avg_binsize)
        plt.plot(moving_avg, label=line_label, color=line_color)
        if show:
            plt.xlabel('episode')
            plt.ylabel('loss')
            plt.legend()
            plt.show()
    
    def train(
        self,
        numEpisodes=None,
        episodeLength=None,
        updateEveryNEpisodes=None,
        checkpointEveryNEpisodes=None,
        plot=True,
        fromPretrained=False,#
        warnIfNotSynched=True,
        saveSettings=True,
        perturbationSize=0.0
    ) -> list:

        self.set_episode_params(
            numEpisodes, episodeLength, updateEveryNEpisodes, checkpointEveryNEpisodes
        )
        # print(self.view_training_params())
        # print(self.view_scenario_params())

        if fromPretrained and warnIfNotSynched and not self.settings_synched():
            return []
        
        if saveSettings:
            self.save_settings()

        losses = train(self.scenarioParams, self.trainingParams, fromPretrained, perturbationSize)
        self.loss_history += losses
        if plot:
            self.plot_loss_history(slice=-self.trainingParams.numEpisodes)
        
        return losses
    
    def run(self, episodeLength=None):
        if not self.settings_synched():
            return
        self.set_episode_params(episodeLength=episodeLength)
        self.save_settings()
        lib.run(self.scenarioParams, self.trainingParams)



def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train or run a model from the command line, will by default load settings from settings.json unless overriden')
    parser.add_argument('--train', action='store_true', help='If provided, will train, otherwise just run for a single episode; a pre-trained model must exist if not training')
    parser.add_argument('--load', action='store_true', help="If provided, will load saved models when training rather than starting from a randomly initialized state")
    parser.add_argument('--npersons', type=int, help='Number of persons in the simulation')
    parser.add_argument('--nfirms', type=int, help='Number of firms in the simulation')
    parser.add_argument('--episodes', type=int, help='Number of episodes to run for, ignored if not training')
    parser.add_argument('--eplength', type=int, help='Length of each episode')
    parser.add_argument('--lr', type=float, help="The learning rate at which to start training, ignored if not training")

    args = parser.parse_args()
    mgr = RuntimeManager(1, 1)
    mgr.load_settings()
    if args.npersons is not None:
        mgr.scenarioParams.numPersons = args.npersons
    if args.nfirms is not None:
        mgr.scenarioParams.numFirms = args.nfirms
    if args.lr is not None:
        mgr.set_all_lrs(args.lr)
    if args.train:
        mgr.train(args.episodes, args.eplength, fromPretrained=args.load)
    else:
        mgr.run(args.eplength)
    

if __name__ == '__main__':
    main()
