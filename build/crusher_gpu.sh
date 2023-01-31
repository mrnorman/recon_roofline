#!/bin/bash

source $MODULESHOME/init/bash
module load PrgEnv-amd amd/5.4.0 cmake

./cmakeclean.sh

unset GATOR_DISABLE
unset GATOR_INITIAL_MB

export CXX=hipcc
unset CXXFLAGS

cmake -DYAKL_ARCH="HIP"                                                                \
      -DYAKL_HIP_FLAGS="-DORD=9 -O3 -ffast-math --offload-arch=gfx90a -Wno-unused-result" \
      ..

