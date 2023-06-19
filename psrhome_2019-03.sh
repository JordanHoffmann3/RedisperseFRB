#!/usr/bin/bash

export SYS_ARCH=skylake
export LOGIN_ARCH=linux_64

# build location for modulefiles
module use /apps/users/pulsar/common/modulefiles
module use /apps/users/pulsar/${SYS_ARCH}/modulefiles

# environment variables for building code etc
module load psrhome/latest

###############################################################################
# GNU compilers
export CFLAGS="-mtune=native -O2 -ffast-math -pthread -fPIC"
export CXXFLAGS="-mtune=native -O2 -ffast-math -pthread -fPIC -std=c++11"
export CC=gcc
export CXX=g++
export FC=gfortran
export F77=gfortran

##############################################################################
# Shell niceties
#
umask 0022
