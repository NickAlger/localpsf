default: all

HLIBPRO_DIR := /home/nick/hlibpro-2.8.1
EIGEN_INCLUDE := /home/nick/anaconda3/envs/fenics2/include/eigen3

HLIBPRO_LIB := $(HLIBPRO_DIR)/lib
HLIBPRO_INCLUDE := $(HLIBPRO_DIR)/include
HLIBPRO_FLAGS := $(shell $(HLIBPRO_DIR)/bin/hlib-config --cflags --lflags)

PYFLAGS  = $(shell python3 -m pybind11 --includes)
PYSUFFIX = $(shell python3-config --extension-suffix)

SRC_DIR  := ./src
OBJ_DIR  := ./obj
# OBJ_DIR  := .
# BUILD_DIR  := ./build
BUILD_DIR  := .

LDFLAGS  = -shared -L$(HLIBPRO_LIB)
CXXFLAGS = -O3 -Wall -shared -fPIC -std=c++11

LIBS := -lhpro -Wl,-rpath,$(HLIBPRO_LIB)

ALL_COMPILE_STUFF = $(CXXFLAGS) $(HLIBPRO_FLAGS) $(PYFLAGS) \
					-I $(HLIBPRO_INCLUDE) -I$(EIGEN_INCLUDE) \
					$(LDFLAGS) $(LIBS)

HLIBPRO_BINDINGS_TARGET = $(addsuffix $(PYSUFFIX), hlibpro_bindings)

# EXAMPLE_TARGET = $(addsuffix $(PYSUFFIX), example)

# all: $(BUILD_DIR)/$(EXAMPLE_TARGET)
# 	@echo 'Finished building target: $@'
# 	@echo ' '

all: $(BUILD_DIR)/$(HLIBPRO_BINDINGS_TARGET)
	@echo 'Finished building target: $@'
	@echo ' '

$(BUILD_DIR)/$(HLIBPRO_BINDINGS_TARGET): $(OBJ_DIR)/hlibpro_bindings.o $(OBJ_DIR)/grid_interpolate.o $(OBJ_DIR)/product_convolution_hmatrix.o
	@echo 'Building target: $@'
	g++ -o "$@" $^ $(ALL_COMPILE_STUFF)
	@echo 'Finished building target: $@'
	@echo ' '

$(OBJ_DIR)/hlibpro_bindings.o: $(SRC_DIR)/hlibpro_bindings.cpp
	@echo 'Building target: $@'
	g++ -o "$@" -c "$<" $(ALL_COMPILE_STUFF)
	@echo 'Finished building target: $@'
	@echo ' '

$(OBJ_DIR)/product_convolution_hmatrix.o: $(SRC_DIR)/product_convolution_hmatrix.cpp
	@echo 'Building target: $@'
	g++ -o "$@" -c "$<" $(ALL_COMPILE_STUFF)
	@echo 'Finished building target: $@'
	@echo ' '

$(OBJ_DIR)/grid_interpolate.o: $(SRC_DIR)/grid_interpolate.cpp
	@echo 'Building target: $@'
	g++ -o "$@" -c "$<" $(ALL_COMPILE_STUFF)
	@echo 'Finished building target: $@'
	@echo ' '


$(BUILD_DIR)/$(EXAMPLE_TARGET): $(OBJ_DIR)/example.o $(OBJ_DIR)/ex1.o $(OBJ_DIR)/ex2.o
	@echo 'Building target: $@'
	g++ -o "$@" "$<" $(ALL_COMPILE_STUFF)
	@echo 'Finished building target: $@'
	@echo ' '

$(OBJ_DIR)/example.o: $(SRC_DIR)/example.cpp
	@echo 'Building target: $@'
	g++ -o "$@" -c "$<" $(ALL_COMPILE_STUFF)
	@echo 'Finished building target: $@'
	@echo ' '

$(OBJ_DIR)/ex1.o: $(SRC_DIR)/ex1.cpp
	@echo 'Building target: $@'
	g++ -o "$@" -c "$<" $(ALL_COMPILE_STUFF)
	@echo 'Finished building target: $@'
	@echo ' '

$(OBJ_DIR)/ex2.o: $(SRC_DIR)/ex2.cpp
	@echo 'Building target: $@'
	g++ -o "$@" -c "$<" $(ALL_COMPILE_STUFF)
	@echo 'Finished building target: $@'
	@echo ' '

clean:
	-rm -rf $(BUILD_DIR)/$(EXAMPLE_TARGET)
	-rm -rf $(OBJ_DIR)/example.o
	-rm -rf $(OBJ_DIR)/ex2.o
	-rm -rf $(OBJ_DIR)/ex1.o
	-rm -rf $(OBJ_DIR)/product_convolution_hmatrix.o
	-rm -rf $(OBJ_DIR)/hlibpro_bindings.o
	-rm -rf $(OBJ_DIR)/grid_interpolate.o
	-rm -rf $(BUILD_DIR)/$(HLIBPRO_BINDINGS_TARGET)
	-@echo ' '

.PHONY: all clean dependents
