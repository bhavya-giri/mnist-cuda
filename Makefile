NVCC ?= nvcc
CC   ?= gcc
# Colab T4 = sm_75; override: make v5 CUDA_ARCH=sm_80
CUDA_ARCH ?= sm_75

NVCC_FLAGS = -O2 -arch=$(CUDA_ARCH) -Wno-deprecated-gpu-targets

.PHONY: all v3 v4 v5 clean

all: v3 v5

v3: v3.c
	$(CC) -O2 -lm v3.c -o v3

v4: v4.cu
	$(NVCC) $(NVCC_FLAGS) v4.cu -o v4

# cuBLAS must be linked explicitly (-lcublas)
v5: v5.cu
	$(NVCC) $(NVCC_FLAGS) v5.cu -o v5 -lcublas

clean:
	rm -f v3 v4 v5
