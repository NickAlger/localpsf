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
# BUILD_DIR  := ./build
BUILD_DIR  := ./

LDFLAGS  = -shared -L$(HLIBPRO_LIB)
CXXFLAGS = -O3 -Wall -shared -fPIC -std=c++11

LIBS := -lhpro -Wl,-rpath,$(HLIBPRO_LIB)

HLIBPRO_BINDINGS_TARGET = $(addsuffix $(PYSUFFIX), hlibpro_bindings)

all: $(BUILD_DIR)/$(HLIBPRO_BINDINGS_TARGET)
	@echo 'Finished building target: $@'
	@echo ' '

$(BUILD_DIR)/$(HLIBPRO_BINDINGS_TARGET): $(OBJ_DIR)/hlibpro_bindings.o $(OBJ_DIR)/grid_interpolate.o
	@echo 'Building target: $@'
	g++ -o "$@" "$<" \
	    $(CXXFLAGS) $(HLIBPRO_FLAGS) $(PYFLAGS) \
	    -I $(HLIBPRO_INCLUDE) -I$(EIGEN_INCLUDE) \
	    $(LDFLAGS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

$(OBJ_DIR)/hlibpro_bindings.o: $(SRC_DIR)/hlibpro_bindings.cpp
	@echo 'Building target: $@'
	g++ -o "$@" "$<" \
		$(CXXFLAGS) $(HLIBPRO_FLAGS) $(PYFLAGS) \
		-I $(HLIBPRO_INCLUDE) -I$(EIGEN_INCLUDE) \
		$(LDFLAGS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

$(OBJ_DIR)/grid_interpolate.o: $(SRC_DIR)/grid_interpolate.cpp
	@echo 'Building target: $@'
	g++ -o "$@" -c "$<" $(CXXFLAGS) -I$(EIGEN_INCLUDE)
	@echo 'Finished building target: $@'
	@echo ' '

clean:
	-rm -rf $(OBJ_DIR)/hlibpro_bindings.o
	-rm -rf $(OBJ_DIR)/grid_interpolate.o
	-rm -rf $(BUILD_DIR)/$(HLIBPRO_BINDINGS_TARGET)
	-@echo ' '

.PHONY: all clean dependents
