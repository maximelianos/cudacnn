################ MAKEFILE TEMPLATE ################

# Author : Lucas Carpenter

# Usage : make target1

# What compiler are we using? (gcc, g++, nvcc, etc)
LINK = nvcc

# Name of our binary executable
OUT_FILE = denoiser

# Any weird flags ( -O2/-O3/-Wno-deprecated-gpu-targets/-fopenmp/etc)
FLAGS = -Wno-deprecated-gpu-targets -O2 -Xcompiler -fopenmp -std=c++11
LINK_FLAGS = --cudart static --relocatable-device-code=false -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30

all: denoiser

denoiser: src/sobel.cu src/imageLoader.cpp src/lodepng.cpp
	$(LINK) -o $(OUT_FILE) $(FLAGS) $(LINK_FLAGS) $^

clean: 
	rm -f *.o *~ core

distclean: clean
	rm $(OUT_FILE)
