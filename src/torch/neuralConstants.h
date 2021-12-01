#ifndef NEURAL_CONSTANTS_H
#define NEURAL_CONSTANTS_H

#define _USE_MATH_DEFINES
#include <cmath>

namespace neural {

// this gets used in calculating the log pdf of a normal distribution
const double SQRT2PI = 2 / (M_2_SQRTPI * M_SQRT1_2);

// for defining default decision net architecture
const int DEFAULT_stackSize = 10;
const int DEFAULT_encodingSize = 10;
const int DEFAULT_hiddenSize = 100;
const int DEFAULT_nHidden = 6;
const int DEFAULT_nHiddenSmall = 3;

// for defining default training behavior
const float DEFAULT_LEARNING_RATE = 0.001;
const unsigned int DEFAULT_EPISODE_BATCH_SIZE_FOR_LR_DECAY = 10;
const unsigned int DEFAULT_PATIENCE_FOR_LR_DECAY = 5;
const float DEFAULT_MULTIPLIER_FOR_LR_DECAY = 0.5;

} // namespace neural

#endif