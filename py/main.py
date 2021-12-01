import ctypes
import sys
import os
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


if __name__ == '__main__':
    args = sys.argv
    numPersons = int(args[1])
    numFirms = int(args[2])
    numEpisodes = int(args[3])
    episodeLength = int(args[4])

    scenarioParams = lib.create_scenario_params(numPersons, numFirms)
    trainingParams = lib.create_training_params(numEpisodes, episodeLength, 10, 10)

    losses = train(scenarioParams, trainingParams)
    plt.plot(losses, marker='.', linestyle='', alpha=0.3)
    plt.xlabel('episode')
    plt.ylabel('loss')
    plt.show()
