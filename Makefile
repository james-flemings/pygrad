CFLAGS = -Iinclude -O3 -fopenmp --std=c++17 
CC = g++
TARGET = examples
SRC = $(TARGET)/iris-example.cpp $(wildcard src/*.cpp)
OBJ = $(SRC:.cpp = .o) 

.PHONY: all clean

all: $(TARGET)/iris-example

$(TARGET)/iris-example: $(OBJ)
	$(CC) $(CFLAGS) -o $(TARGET)/iris-example $(OBJ)

clean: 
	rm -f src/*.o