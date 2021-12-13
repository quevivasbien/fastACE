#ifndef NEURAL_CONSTANTS_H
#define NEURAL_CONSTANTS_H

#define _USE_MATH_DEFINES
#include <cmath>

namespace neural {

// this gets used in calculating the log pdf of a normal distribution
const double SQRT2PI = 2 / (M_2_SQRTPI * M_SQRT1_2);

// for defining default decision net architecture
const int DEFAULT_STACK_SIZE = 10;
const int DEFAULT_ENCODING_SIZE = 10;
const int DEFAULT_HIDDEN_SIZE = 100;
const int DEFAULT_N_HIDDEN = 6;
const int DEFAULT_N_HIDDEN_SMALL = 3;

// for defining default training behavior
const unsigned int DEFAULT_NUM_EPISODES = 100;
const unsigned int DEFAULT_EPISODE_LENGTH = 20;
const unsigned int DEFAULT_UPDATE_EVERY_N_EPISODES = 10;
const unsigned int DEFAULT_CHECKPOINT_EVERY_N_EPISODES = 10;

const double DEFAULT_LEARNING_RATE = 1e-5;
const unsigned int DEFAULT_EPISODE_BATCH_SIZE_FOR_LR_DECAY = 10;
const unsigned int DEFAULT_PATIENCE_FOR_LR_DECAY = 5;
const double DEFAULT_MULTIPLIER_FOR_LR_DECAY = 0.5;
const unsigned int DEFAULT_REVERSE_ANNEALING_PERIOD = 2;

// where trained models save by default
const char DEFAULT_SAVE_DIR[] = "../models/";


} // namespace neural

#endif