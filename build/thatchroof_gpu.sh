#!/bin/bash

source /usr/share/modules/init/bash
module purge

./cmakeclean.sh

unset GATOR_DISABLE
unset GATOR_INITIAL_MB

export CXX=g++
unset CXXFLAGS

cmake -DYAKL_ARCH="CUDA"                                                     \
      -DYAKL_CUDA_FLAGS="-DORD=9 -O3 --use_fast_math -arch sm_86 -ccbin g++" \
      ..

