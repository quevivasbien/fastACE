CC = g++

SRC = src
PERSONS_SRC = $(SRC)/persons
FIRMS_SRC = $(SRC)/firms
FUNCTIONS_SRC = $(SRC)/functions
OBJ = obj
BIN = bin

# set EIGEN_PATH to the path to the Eigen library
EIGEN_PATH = ../eigen
EIGEN_UNSUPP = $(EIGEN_PATH)/unsupported

INCLUDES = -I $(SRC) -I $(EIGEN_PATH) -I $(EIGEN_UNSUPP)

# set CFLAGS however you want, though don't remove $(INCLUDES)
CFLAGS = $(INCLUDES) -Wall -O2

# the following just figure out where all the source code and object files are
SOURCES = $(wildcard $(SRC)/*.cpp)
OBJECTS = $(patsubst $(SRC)/%.cpp, $(OBJ)/%.o, $(SOURCES))
PERSONS_SOURCES = $(wildcard $(PERSONS_SRC)/*.cpp)
PERSONS_OBJECTS = $(patsubst $(PERSONS_SRC)/%.cpp, $(OBJ)/%.o, $(PERSONS_SOURCES))
FIRMS_SOURCES = $(wildcard $(FIRMS_SRC)/*.cpp)
FIRMS_OBJECTS = $(patsubst $(FIRMS_SRC)/%.cpp, $(OBJ)/%.o, $(FIRMS_SOURCES))
FUNCTIONS_SOURCES = $(wildcard $(FUNCTIONS_SRC)/*.cpp)
FUNCTIONS_OBJECTS = $(patsubst $(FUNCTIONS_SRC)/%.cpp, $(OBJ)/%.o, $(FUNCTIONS_SOURCES))

# link all object files
$(BIN)/main: $(OBJECTS) $(PERSONS_OBJECTS) $(FIRMS_OBJECTS) $(FUNCTIONS_OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@

# compile source code
$(OBJ)/%.o: $(SRC)/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ)/%.o: $(PERSONS_SRC)/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ)/%.o: $(FUNCTIONS_SRC)/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@
