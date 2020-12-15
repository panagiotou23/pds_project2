CC=gcc
MPICC=mpicc
CILKCC=/usr/local/OpenCilk-9.0.1-Linux/bin/clang
CFLAGS=-O3

BIN_DIR=bin
SRC_DIR=src
$(info $(shell mkdir -p $(BIN_DIR)))

default: main

blas: 
	$(CC) $(CFLAGS) -o $(BIN_DIR)/blas $(SRC_DIR)/blas.c -lopenblas -lpthread

main:
	$(MPICC) $(CFLAGS) -o $(BIN_DIR)/test $(SRC_DIR)/main.c -lopenblas

.PHONY: clean

clean:
	rm -rf $(BIN_DIR)
