#include "gpuFunctions.cuh"

void checkCudaError(cudaError_t result, const char* msg)
{
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " : " << cudaGetErrorString(result) << std::endl;
        system("PAUSE");
        exit(result);
    }
}

void instantiateGPUMemory(Coord** posAntGPU, Coord** posActGPU, Coord** posNextGPU, float** velocitiesGPU, unsigned int numBytesPos, unsigned int numBytesVel, Coord* partAnt, Coord* partAct)
{
    checkCudaError(cudaMalloc(posAntGPU, numBytesPos), "cudaMalloc1 posAntGPU");
    checkCudaError(cudaMalloc(posActGPU, numBytesPos), "cudaMalloc2 posActGPU");
    checkCudaError(cudaMalloc(posNextGPU, numBytesPos), "cudaMalloc3 posNextGPU");
    checkCudaError(cudaMalloc(velocitiesGPU, numBytesVel), "cudaMalloc4 velocitiesGPU");

    checkCudaError(cudaMemcpy(*posAntGPU, partAnt, numBytesPos, cudaMemcpyHostToDevice), "cudaMemcpy1 posAntGPU");
    checkCudaError(cudaMemcpy(*posActGPU, partAct, numBytesPos, cudaMemcpyHostToDevice), "cudaMemcpy2 posActGPU");
}

extern "C" __global__ void calculateNewPositionGPU(Coord * posAntGPU, Coord * posActGPU, Coord * posNextGPU, int numParticles, float timeStep, float gravConstant)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numParticles)
    {
        Coord Pi = posActGPU[index];
        Coord PiAnt = posAntGPU[index];
        Coord force = { 0.0f, 0.0f };

        // Epsilon parameter for Plummer Softening
        // You can adjust this value. If it's too large, gravity will be "spongy".
        // If it's too small, you'll get extreme bounces again.
        float epsilon = 0.05f; // Large buffer to prevent infinite accelerations
        float epsilonSqr = epsilon * epsilon;

        // Calculate gravitational force
        for (int j = 0; j < numParticles; ++j) {
            if (index != j) {
                Coord Pj = posActGPU[j];
                Coord dirVector = { Pj.x - Pi.x, Pj.y - Pi.y };

                // 1. Calculate squared distance (r^2)
                float distSqr = dirVector.x * dirVector.x + dirVector.y * dirVector.y;

                // 2. Apply Plummer softening: r^2 + epsilon^2
                float distSoftSqr = distSqr + epsilonSqr;

                // 3. Calculate (r^2 + epsilon^2)^(3/2)
                // Using (x * sqrt(x)) is more efficient in CUDA than using pow()
                float distCubic = distSoftSqr * sqrtf(distSoftSqr);

                // 4. Calculate final force factor
                float factor = gravConstant / distCubic;

                force.x += factor * dirVector.x;
                force.y += factor * dirVector.y;
            }
        }
        // Verlet integration
        float nextX = 2 * Pi.x - PiAnt.x + (timeStep * timeStep) * force.x;
        float nextY = 2 * Pi.y - PiAnt.y + (timeStep * timeStep) * force.y;

        float limit = 1.0f;
        float bounce = 0.8f;

        // Collision check in X
        if (nextX > limit) {
            nextX = limit;
            // Reverse inertia by modifying previous position
            posActGPU[index].x = nextX + (Pi.x - PiAnt.x) * bounce;
        } else if (nextX < -limit) {
            nextX = -limit;
            posActGPU[index].x = nextX + (Pi.x - PiAnt.x) * bounce;
        }

        // Collision check in Y
        if (nextY > limit) {
            nextY = limit;
            posActGPU[index].y = nextY + (Pi.y - PiAnt.y) * bounce;
        } else if (nextY < -limit) {
            nextY = -limit;
            posActGPU[index].y = nextY + (Pi.y - PiAnt.y) * bounce;
        }

        // Assign final safe position
        posNextGPU[index].x = nextX;
        posNextGPU[index].y = nextY;
    }
}

extern "C" __global__ void calculateVelocitiesGPU(Coord * posAntGPU, Coord * posActGPU, Coord * posNextGPU, float* velocitiesGPU, int numParticles, float timeStep, float gravConstant)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numParticles)
    {
        Coord Pi = posActGPU[index];
        Coord PiAnt = posAntGPU[index];
        Coord force = { 0.0f, 0.0f };
        float minDist = 0.002f; // Adjust this parameter accordingly. The smaller it is, the more particles disperse

        // Calculate gravitational force
        for (int j = 0; j < numParticles; ++j) {
            if (index != j) {
                Coord Pj = posActGPU[j];
                Coord dirVector = { Pj.x - Pi.x, Pj.y - Pi.y };
                float dist = sqrtf(dirVector.x * dirVector.x + dirVector.y * dirVector.y);
                if (dist > minDist)
                {
                    float distCubic = dist * dist * dist;
                    float factor = gravConstant / distCubic;
                    force.x += factor * dirVector.x;
                    force.y += factor * dirVector.y;
                }
            }
        }
        // Verlet integration
        posNextGPU[index].x = 2 * Pi.x - PiAnt.x + (timeStep * timeStep) * force.x;
        posNextGPU[index].y = 2 * Pi.y - PiAnt.y + (timeStep * timeStep) * force.y;

        // Calculate velocity of each particle
        float velX = (posNextGPU[index].x - posActGPU[index].x) / timeStep;
        float velY = (posNextGPU[index].y - posActGPU[index].y) / timeStep;
        velocitiesGPU[index] = sqrtf(velX * velX + velY * velY);
    }
}

void launchKernelCalculatePosition(Coord* posAntGPU, Coord* posActGPU, Coord* posNextGPU, int numParticles, float timeStep, float gravConstant, int numBlocks, int blockSize)
{
    calculateNewPositionGPU << <numBlocks, blockSize >> > (posAntGPU, posActGPU, posNextGPU, numParticles, timeStep, gravConstant);
}

void launchKernelCalculateVelocity(Coord* posAntGPU, Coord* posActGPU, Coord* posNextGPU, float* velocitiesGPU, int numParticles, float timeStep, float gravConstant, int numBlocks, int blockSize)
{
    calculateVelocitiesGPU << <numBlocks, blockSize >> > (posAntGPU, posActGPU, posNextGPU, velocitiesGPU, numParticles, timeStep, gravConstant);
}

void getPositionResultGPU(Coord* posAntGPU, Coord* posActGPU, Coord* posNextGPU, int numBytes, Coord* partAct)
{
    cudaDeviceSynchronize(); // Ensure all CUDA thread calculations have completed

    // Update particle position vectors in GPU memory
    checkCudaError(cudaMemcpy(posAntGPU, posActGPU, numBytes, cudaMemcpyDeviceToDevice), "cudaMemcpy3 posAntGPU");
    // Calculation result is saved in the CURRENT particle vector
    checkCudaError(cudaMemcpy(partAct, posNextGPU, numBytes, cudaMemcpyDeviceToHost), "cudaMemcpy4 partAct");
    // Update particle position vectors in GPU memory
    checkCudaError(cudaMemcpy(posActGPU, posNextGPU, numBytes, cudaMemcpyDeviceToDevice), "cudaMemcpy5 posActGPU");

    // The NEXT particle vector can keep its current content as garbage as it doesn't affect functionality in principle
}

void getVelocityResultGPU(Coord* posAntGPU, Coord* posActGPU, Coord* posNextGPU, float* velocitiesGPU, float* velocitiesHost, int numBytesParticles, int numBytesVel)
{
    cudaDeviceSynchronize(); // Ensure all CUDA thread calculations have completed

    // Update particle position vectors in GPU memory
    checkCudaError(cudaMemcpy(posAntGPU, posActGPU, numBytesParticles, cudaMemcpyDeviceToDevice), "cudaMemcpy3 posAntGPU");
    // Update particle position vectors in GPU memory
    checkCudaError(cudaMemcpy(posActGPU, posNextGPU, numBytesParticles, cudaMemcpyDeviceToDevice), "cudaMemcpy4 posActGPU");

    // The NEXT particle vector can keep its current content as garbage as it doesn't affect functionality in principle

    // Update velocities vector on host
    checkCudaError(cudaMemcpy(velocitiesHost, velocitiesGPU, numBytesVel, cudaMemcpyDeviceToHost), "cudaMemcpy5 velocitiesHost");
}

void freeGPUResources(Coord* posAntGPU, Coord* posActGPU, Coord* posNextGPU, float* velocitiesGPU)
{
    cudaFree(posAntGPU);
    cudaFree(posActGPU);
    cudaFree(posNextGPU);
    cudaFree(velocitiesGPU);
}

