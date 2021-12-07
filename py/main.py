import ctypes
import os
import json
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.realpath(__file__)))


lib = ctypes.CDLL("../bin/libpybindings.so")

class Config(ctypes.Structure):
    _fields_ = [
        ('verbose', ctypes.c_uint),
        ('eps', ctypes.c_double),
        ('largeNumber', ctypes.c_double),
        ('multithreaded', ctypes.c_bool),
        ('numThreads', ctypes.c_uint)
    ]

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
        ('purchaseNetLR', ctypes.c_float),
        ('firmPurchaseNetLR', ctypes.c_float),
        ('laborSearchNetLR', ctypes.c_float),
        ('consumptionNetLR', ctypes.c_float),
        ('productionNetLR', ctypes.c_float),
        ('offerNetLR', ctypes.c_float),
        ('jobOfferNetLR', ctypes.c_float),
        ('valueNetLR', ctypes.c_float),
        ('firmValueNetLR', ctypes.c_float),
        ('episodeBatchSizeForLRDecay', ctypes.c_uint),
        ('patienceForLRDecay', ctypes.c_uint),
        ('multiplierForLRDecay', ctypes.c_float)
    ]


lib.get_config.restype = ctypes.POINTER(Config)


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
    ctypes.POINTER(ctypes.c_float),
    CustomScenarioParams,
    TrainingParams,
    ctypes.c_bool
]
lib.train.restype = None


def train(
    scenarioParams: CustomScenarioParams,
    trainingParams: TrainingParams,
    fromPretrained=False
) -> list:
    losses = ctypes.ARRAY(ctypes.c_float, trainingParams.numEpisodes)()
    lib.train(
        losses, scenarioParams, trainingParams, fromPretrained
    )
    return list(losses)


def dict_from_cstruct(struct):
    fields = [field[0] for field in getattr(struct, '_fields_')]
    return {field: getattr(struct, field) for field in fields}


class RuntimeManager:

    def __init__(
        self,
        numPersons: int,
        numFirms: int,
        # saveDir: str
    ):
        self.config = lib.get_config()
        self.scenarioParams = lib.create_scenario_params(
            numPersons,
            numFirms
        )
        self.trainingParams = lib.create_training_params()
        # self.saveDir = saveDir
    
    def edit_config(self, attr: str, new_value):
        setattr(self.config.contents, attr, new_value)
    
    def view_config(self):
        return dict_from_cstruct(self.config.contents)
    
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
            'config': self.view_config(),
            'scenarioParams': self.view_scenario_params(),
            'trainingParams': self.view_training_params()
        }
        with open('settings.json', 'w') as fh:
            json.dump(settings, fh)
    
    def load_settings(self, set=True):
        with open('settings.json', 'r') as fh:
            settings = json.load(fh)
        if set:
            for key, value in settings['config'].items():
                self.edit_config(key, value)
            for key, value in settings['scenarioParams'].items():
                self.edit_scenario_params(key, value)
            for key, value in settings['trainingParams'].items():
                self.edit_training_params(key, value)
            self.save_settings()
        return settings

    def settings_synched(self) -> bool:
        settings = self.load_settings(False)
        synched = (
            settings['scenarioParams'] == self.view_scenario_params()
            and settings['trainingParams'] == self.view_training_params()
        )
        if not synched:
            print(
                "You're trying to load a model that doesn't match the settings you're currently using.\n"
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
    
    def train(
        self,
        numEpisodes=None,
        episodeLength=None,
        updateEveryNEpisodes=None,
        checkpointEveryNEpisodes=None,
        plot=True,
        fromPretrained=False
    ) -> list:
        self.set_episode_params(
            numEpisodes, episodeLength, updateEveryNEpisodes, checkpointEveryNEpisodes
        )

        if fromPretrained and not self.settings_synched():
            return []

        losses = train(self.scenarioParams, self.trainingParams, fromPretrained)
        if plot:
            plt.plot(losses, marker='.', linestyle='', alpha=0.3)
            plt.xlabel('episode')
            plt.ylabel('loss')
            plt.show()
        

        # model_files = [f for f in os.listdir(os.getcwd()) if f.endswith('.pt')]
        # if not os.path.isdir(self.saveDir):
        #     os.makedirs(self.saveDir)
        # for f in model_files:
        #     os.rename(f, os.path.join(self.saveDir, f))

        return losses
    
    def run(self, episodeLength=None):
        if not self.settings_synched():
            return
        self.set_episode_params(episodeLength=episodeLength)
        lib.run(self.scenarioParams, self.trainingParams)
    
