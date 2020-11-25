default: all

HLIBPRO_DIR := /home/nick/hlibpro-2.8.1
EIGEN_INCLUDE := /home/nick/anaconda3/envs/fenics2/include/eigen3

HLIBPRO_LIB := $(HLIBPRO_DIR)/lib
HLIBPRO_INCLUDE := $(HLIBPRO_DIR)/include
HLIBPRO_FLAGS := $(shell $(HLIBPRO_DIR)/bin/hlib-config --cflags --lflags)

# The following two lines may (or may not) require 
# the proper environment variables to be set
PYFLAGS  = $(shell python3 -m pybind11 --includes)
PYSUFFIX = $(shell python3-config --extension-suffix)

# SRC_DIR  := ../src
# OBJ_DIR  := ../obj
# DEP_DIR  := ../obj
# CPP_SRCS := $(wildcard $(SRC_DIR)/*.cpp)
# OBJS     := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SRCS))
# CPP_DEPS := $(patsubst $(SRC_DIR)/%.cpp,$(DEP_DIR)/%.d,$(CPP_SRCS))
LDFLAGS  = -shared -L$(HLIBPRO_LIB)
CXXFLAGS = -I$(HLIBPRO_INCLUDE) \
           -I$(EIGEN_INCLUDE) \
		   -O3 -Wall -shared -fPIC -std=c++11

LIBS := -lhpro -Wl,-rpath,$(HLIBPRO_LIB)

HLIBPRO_BINDINGS_TARGET = $(addsuffix $(PYSUFFIX), hlibpro_bindings)

# all: $(TARGET_NAME) sparse_hlibpro_error
all: $(HLIBPRO_BINDINGS_TARGET)
	@echo 'Finished building target: $@'
	@echo ' '

# Tool invocations
$(HLIBPRO_BINDINGS_TARGET): hlibpro_bindings.o grid_interpolate.o
	@echo 'Building target: $@'
# 	g++ $(CXXFLAGS) $(PYFLAGS) -o "$@" "$<" $(HLIBPRO_FLAGS) $(LDFLAGS) -I $(HLIBPRO_INCLUDE) $(LIBS)
	g++ $(CXXFLAGS) $(PYFLAGS) -o "$@" hlibpro_bindings.o grid_interpolate.o $(HLIBPRO_FLAGS) $(LDFLAGS) -I $(HLIBPRO_INCLUDE) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

hlibpro_bindings.o: hlibpro_bindings.cpp
	@echo 'Building target: $@'
# 	g++ $(CXXFLAGS) $(PYFLAGS) -o "$@" "$<" $(HLIBPRO_FLAGS) $(LDFLAGS) -I $(HLIBPRO_INCLUDE) $(LIBS)
	g++ $(CXXFLAGS) $(PYFLAGS) -o hlibpro_bindings.o -c hlibpro_bindings.cpp $(HLIBPRO_FLAGS) $(LDFLAGS) -I $(HLIBPRO_INCLUDE) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

grid_interpolate.o: grid_interpolate.cpp
	@echo 'Building target: $@'
	g++ $(CXXFLAGS) $(PYFLAGS) -o grid_interpolate.o -c grid_interpolate.cpp $(HLIBPRO_FLAGS) $(LDFLAGS) -I $(HLIBPRO_INCLUDE) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '


# sparse_hlibpro_error: sparse_hlibpro_error.cpp
# 	@echo 'Building target: $@'
# 	g++ -o "$@" "$<" $(HLIBPRO_FLAGS)
# 	@echo 'Finished building target: $@'
# 	@echo ' '

-include $(OBJS:.o=.d)

# Other Targets
RM := rm -rf

clean:
	-$(RM) $(CPP_DEPS) $(OBJS) $(TARGET_NAME)
	-@echo ' '

.PHONY: all clean dependents
