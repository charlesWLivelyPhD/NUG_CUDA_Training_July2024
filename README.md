# CUDA C++ Examples

This repository contains four introductory CUDA C++ programs that demonstrate basic concepts of CUDA programming. These examples are designed to be run on Perlmutter at NERSC and cover various aspects of GPU programming, including memory management, kernel launches, and parallel processing using threads and blocks.

## Prerequisites

1. Access to the Perlmutter HPC system at NERSC.
2. Basic knowledge of programming.
3. Modules for CUDA and necessary compilers should be available.

## Overview

1. **Hello World**
2. **Vector Addition**
3. **Matrix Multiplication**
4. **Dot Product**

## Programs

### 1. Hello World

This example introduces the basic structure of a CUDA program with a simple kernel that prints "Hello World!" from the device.

**File:** `hello_world.cu`

It must be noted that to build this example `nvcc` is being passed a `-arch=sm_80` flag, this is to make sure that code is built for devices with `Compute Capability 8.0` i.e. NVIDIA A100 devices that are available on Perlmutter. To build this example make sure that module `cudatoolkit` has been loaded and then follow the steps below:
```bash
cd NUG_CUDA_Training_July2024
make clean
make
```
To run:

```bash
sbatch batch.sh
```
which contains
```
./hello_world
```
### 2. Vector Addition
**File:** `vector_add.cu`
The file vector_add.cu contains the following components:

Kernel Function: __global__ void vectorAdd(int *a, int *b, int *c, int n) that performs the addition of two vectors.
Main Function: Initializes vectors, allocates memory on host and device, copies data between host and device, launches the kernel, and prints the results.

To execute this update batch.sh with ./vector_add instead of hello_world
### 3. Vector Addition
The file matrix_mul.cu contains the following components:

Kernel Function: __global__ void matrixMul(int *a, int *b, int *c, int width) that performs matrix multiplication.
Main Function: Initializes matrices, allocates memory on host and device, copies data between host and device, launches the kernel, and prints the results.

### 4. Dot Product


The file dot_product.cu contains the following components:
**File:** `dot_product.cu`
Kernel Function: __global__ void dotProduct(int *a, int *b, int *c, int n) that performs the dot product of two vectors.
Main Function: Initializes vectors, allocates memory on host and device, copies data between host and device, launches the kernel, and prints the result.


