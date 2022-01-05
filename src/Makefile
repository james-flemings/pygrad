CFLAGS = -O3 -fopenmp --std=c++17
CC = g++
SRC = main.cpp Model.cpp DenseLayer.cpp Layer.cpp Initializer.cpp Neuron.cpp
OBJ = $(SRC:.cpp = .o)

pygrad: $(OBJ)
	$(CC) $(CFLAGS) -o pygrad $(OBJ)