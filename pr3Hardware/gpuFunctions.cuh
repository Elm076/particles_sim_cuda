/*****************************************************************//**
 * @file   gpuFunctions.cuh
 * @brief  Header file with functions necessary for particle simulation
 * @author fotop
 * @date   June 2024
 *********************************************************************/

#ifndef gpuFunctions_cuh
#define gpuFunctions_cuh

#include <cuda_runtime.h>
#include <device_launch_parameters.h> // Needed to recognize block and thread identification variables
#include <glm/fwd.hpp>
#include <glm/vec2.hpp> // These GLM headers are needed to use the Coord type as defined below
#include <math.h>
#include <iostream>

using Coord = glm::vec2;

/** Function to check for CUDA errors */
void checkCudaError(cudaError_t result, const char* msg);

/** Function to allocate necessary GPU memory for the simulation */
void instantiateGPUMemory(Coord** posAnt, Coord** posAct, Coord** posNext, float** velocities, unsigned int numBytesPos, unsigned int numBytesVel, Coord* partAnt, Coord* partAct);

/** Kernel function to calculate the new particle positions */
extern "C" __global__ void calculateNewPositionGPU(Coord* posAntGPU, Coord* posActGPU, Coord* posNextGPU, int numParticles, float timeStep, float gravConstant);

/** Kernel function to calculate particle velocities */
extern "C" __global__ void calculateVelocitiesGPU(Coord* posAntGPU, Coord* posActGPU, Coord* posNextGPU, float* velocitiesGPU, int numParticles, float timeStep, float gravConstant);

/** Function to launch position calculation kernel from .cpp file */
void launchKernelCalculatePosition(Coord* posAnt, Coord* posAct, Coord* posNext, int numParticles, float timeStep, float gravConstant, int numBlocks, int blockSize);

/** Function to launch velocity calculation kernel from .cpp file */
void launchKernelCalculateVelocity(Coord* posAntGPU, Coord* posActGPU, Coord* posNextGPU, float* velocitiesGPU, int numParticles, float timeStep, float gravConstant, int numBlocks, int blockSize);

/** Function to transfer position calculation results from GPU to CPU */
void getPositionResultGPU(Coord* posAnt, Coord* posAct, Coord* posNext, int numBytes, Coord* partAct);

/** Function to transfer velocity calculation results from GPU to CPU */
void getVelocityResultGPU(Coord* posAnt, Coord* posAct, Coord* posNext, float* velocitiesGPU, float* velocitiesHost, int numBytesParticles, int numBytesVel);

/** Function to free GPU memory allocated for the simulation */
void freeGPUResources(Coord* posAnt, Coord* posAct, Coord* posNext, float* velocitiesGPU);

#endif

