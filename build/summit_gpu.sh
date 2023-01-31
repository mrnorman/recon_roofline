#!/bin/bash

source $MODULESHOME/init/bash
module purge
module load cuda/11.4.2 cmake gcc

./cmakeclean.sh

unset GATOR_DISABLE
unset GATOR_INITIAL_MB

export CXX=g++
unset CXXFLAGS

cmake -DYAKL_ARCH="CUDA"                                                     \
      -DYAKL_CUDA_FLAGS="-DORD=9 -O3 --use_fast_math -arch sm_70 -ccbin g++" \
      ..

