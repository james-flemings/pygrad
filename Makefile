CFLAGS = -Iinclude -O3 -fopenmp --std=c++17 
CC = g++
TARGET = examples
SRC_IRIS = $(TARGET)/iris-example.cpp $(wildcard src/*.cpp)
OBJ_IRIS = $(SRC_IRIS:.cpp = .o) 
SRC_MNIST = $(TARGET)/mnist-example.cpp $(wildcard src/*.cpp)
OBJ_MNIST = $(SRC_MNIST:.cpp = .o) 

.PHONY: all clean

all: $(TARGET)/iris-example $(TARGET)/mnist-example

$(TARGET)/iris-example: $(OBJ_IRIS)
	$(CC) $(CFLAGS) -o $(TARGET)/iris-example $(OBJ_IRIS)

$(TARGET)/mnist-example: $(OBJ_MNIST)
	$(CC) $(CFLAGS) -o $(TARGET)/mnist-example $(OBJ_MNIST)

clean: 
	rm -f src/*.o