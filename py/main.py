import ctypes
import sys
import os
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

lib.create_training_params.argtypes = [
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint
]
lib.create_training_params.restype = TrainingParams

lib.train.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    CustomScenarioParams,
    TrainingParams
]
lib.train.restype = None


def train(scenarioParams, trainingParams):
    losses = ctypes.ARRAY(ctypes.c_float, trainingParams.numEpisodes)()
    lib.train(
        losses, scenarioParams, trainingParams
    )
    return list(losses)


def dict_from_cstruct(struct):
    fields = [field[0] for field in getattr(struct, '_fields_')]
    return {field: getattr(struct, field) for field in fields}


class RuntimeManager:

    def __init__(
        self,
        numPersons,
        numFirms,
        numEpisodes,
        episodeLength,
        updateEveryNEpisodes=10,
        checkpointEveryNEpisodes=10
    ):
        self.config = lib.get_config()
        self.scenarioParams = lib.create_scenario_params(
            numPersons,
            numFirms
        )
        self.trainingParams = lib.create_training_params(
            numEpisodes,
            episodeLength,
            updateEveryNEpisodes,
            checkpointEveryNEpisodes
        )
    
    def edit_config(self, attr, new_value):
        setattr(self.config.contents, attr, new_value)
    
    def view_config(self):
        return dict_from_cstruct(self.config.contents)
    
    def edit_scenario_params(self, attr, new_value):
        setattr(self.scenarioParams, attr, new_value)
    
    def view_scenario_params(self):
        return dict_from_cstruct(self.scenarioParams)

    def edit_training_params(self, attr, new_value):
        setattr(self.trainingParams, attr, new_value)
    
    def view_training_params(self):
        return dict_from_cstruct(self.trainingParams)
    
    def train(self, plot=True):
        losses = ctypes.ARRAY(
            ctypes.c_float, self.trainingParams.numEpisodes
        )()
        lib.train(
            losses, self.scenarioParams, self.trainingParams
        )
        losses = list(losses)
        if plot:
            plt.plot(losses, marker='.', linestyle='', alpha=0.3)
            plt.xlabel('episode')
            plt.ylabel('loss')
            plt.show()
        return losses
    
