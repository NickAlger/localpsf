#!/bin/sh

g++ -Wall -shared -std=c++11 -undefined dynamic_lookup \
	`python3 -m pybind11 --includes` \
	-I/Users/gaol/study/pybind/hlibpro/hlibpro-2.8.1/include -I/usr/local/Cellar/fftw/3.3.8_2/include \
	-I/usr/local/Cellar/gsl/2.6/include -L/Users/gaol/study/pybind/hlibpro/hlibpro-2.8.1/lib -lhpro -Wl,-framework,Accelerate \
	-lboost_filesystem -lboost_system -lboost_program_options -lboost_iostreams -ltbb -lz \
	-L/usr/local/Cellar/fftw/3.3.8_2/lib -lfftw3 -L/usr/local/Cellar/gsl/2.6/lib -lgsl -lgslcblas -lm \
	../src/user.cpp -o ./user`python3-config --extension-suffix`

# -Wl,-rpath,/Users/gaol/study/pybind/hlibpro/hlibpro-2.8.1/lib \