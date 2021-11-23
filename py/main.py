import ctypes
import sys
import os
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.realpath(__file__)))


lib = ctypes.CDLL("../bin/libpybindings.so")
lib.train.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint
]

lib.train.restype = None


def train(numPersons, numFirms, numEpisodes, episodeLength):
    losses = ctypes.ARRAY(ctypes.c_float, numEpisodes)()
    lib.train(
        losses,
        ctypes.c_uint(numPersons),
        ctypes.c_uint(numFirms),
        ctypes.c_uint(numEpisodes),
        ctypes.c_uint(episodeLength)
    )
    return list(losses)


if __name__ == '__main__':
    args = sys.argv
    numPersons = int(args[1])
    numFirms = int(args[2])
    numEpisodes = int(args[3])
    episodeLength = int(args[4])

    losses = train(numPersons, numFirms, numEpisodes, episodeLength)
    plt.plot(losses, marker='.', linestyle='', alpha=0.3)
    plt.xlabel('episode')
    plt.ylabel('loss')
    plt.show()
