#include "funcionesGPU.cuh"

void checkCudaError(cudaError_t result, const char* msg)
{
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " : " << cudaGetErrorString(result) << std::endl;
        system("PAUSE");
        exit(result);
    }
}
void instanciarMemGPU(Coord** posAntGPU, Coord** posActGPU, Coord** posSigGPU, float** velocidadesGPU, unsigned int numBytesPos, unsigned int numBytesVel, Coord* partAct)
{
    checkCudaError(cudaMalloc(posAntGPU, numBytesPos), "cudaMalloc1 posAntGPU");
    checkCudaError(cudaMalloc(posActGPU, numBytesPos), "cudaMalloc2 posActGPU");
    checkCudaError(cudaMalloc(posSigGPU, numBytesPos), "cudaMalloc3 posSigGPU");
    checkCudaError(cudaMalloc(velocidadesGPU, numBytesVel), "cudaMalloc4 velocidadesGPU");

    checkCudaError(cudaMemcpy(*posAntGPU, partAct, numBytesPos, cudaMemcpyHostToDevice), "cudaMemcpy1 posAntGPU");
    checkCudaError(cudaMemcpy(*posActGPU, partAct, numBytesPos, cudaMemcpyHostToDevice), "cudaMemcpy2 posActGPU");
}

extern "C" __global__ void calculoNuevaPosicionGPU(Coord * posAntGPU, Coord * posActGPU, Coord * posSigGPU, int numParticulas, float pasoTiempo, float constGrav)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numParticulas)
    {
        Coord Pi = posActGPU[index];
        Coord PiAnt = posAntGPU[index];
        Coord fuerza = { 0.0f, 0.0f };

        // Parámetro epsilon para el Plummer Softening
        // Puedes jugar con este valor. Si es muy grande, la gravedad será "esponjosa".
        // Si es muy pequeño, volverás a tener rebotes extremos.
        float epsilon = 0.0012f;
        float epsilonSqr = epsilon * epsilon;
        
        // Calcular la fuerza gravitacional
        for (int j = 0; j < numParticulas; ++j) {
            if (index != j) {
                Coord Pj = posActGPU[j];
                Coord dirVector = { Pj.x - Pi.x, Pj.y - Pi.y };

                // 1. Calculamos la distancia al cuadrado (r^2)
                float distSqr = dirVector.x * dirVector.x + dirVector.y * dirVector.y;

                // 2. Aplicamos el suavizado de Plummer: r^2 + epsilon^2
                float distSoftSqr = distSqr + epsilonSqr;

                // 3. Calculamos (r^2 + epsilon^2)^(3/2)
                // Usar (x * sqrt(x)) es más eficiente en CUDA que usar pow()
                float distCubica = distSoftSqr * sqrtf(distSoftSqr);

                // 4. Calculamos el factor de fuerza final
                float factor = constGrav / distCubica;

                fuerza.x += factor * dirVector.x;
                fuerza.y += factor * dirVector.y;
            }
        }
        //Integraci�n de Verlet
        posSigGPU[index].x = 2 * Pi.x - PiAnt.x + (pasoTiempo * pasoTiempo) * fuerza.x;
        posSigGPU[index].y = 2 * Pi.y - PiAnt.y + (pasoTiempo * pasoTiempo) * fuerza.y;
    }
}

extern "C" __global__ void calculoVelocidadesGPU(Coord * posAntGPU, Coord * posActGPU, Coord * posSigGPU, float* velocidadesGPU, int numParticulas, float pasoTiempo, float constGrav)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numParticulas)
    {
        Coord Pi = posActGPU[index];
        Coord PiAnt = posAntGPU[index];
        Coord fuerza = { 0.0f, 0.0f };
        float minDist = 0.002f; //aqu� tengo que ir ajustando este par�metro. CU�NTO M�S PEQUE�O M�S SE DISPERSA
        
        // Calcular la fuerza gravitacional
        for (int j = 0; j < numParticulas; ++j) {
            if (index != j) {
                Coord Pj = posActGPU[j];
                Coord dirVector = { Pj.x - Pi.x, Pj.y - Pi.y };
                float dist = sqrtf(dirVector.x * dirVector.x + dirVector.y * dirVector.y);
                if (dist > minDist)
                {
                    float distCubica = dist * dist * dist;
                    float factor = constGrav / distCubica;
                    fuerza.x += factor * dirVector.x;
                    fuerza.y += factor * dirVector.y;
                }
            }
        }
        //Integraci�n de Verlet
        posSigGPU[index].x = 2 * Pi.x - PiAnt.x + (pasoTiempo * pasoTiempo) * fuerza.x;
        posSigGPU[index].y = 2 * Pi.y - PiAnt.y + (pasoTiempo * pasoTiempo) * fuerza.y;

        //C�lculo de la velocidad de cada part�cula
        float velX = (posSigGPU[index].x - posActGPU[index].x) / pasoTiempo;
        float velY = (posSigGPU[index].y - posActGPU[index].y) / pasoTiempo;
        velocidadesGPU[index] = sqrtf(velX * velX + velY * velY);
    }
}

void lanzarKernelCalculoPosicion(Coord* posAntGPU, Coord* posActGPU, Coord* posSigGPU, int numParticulas, float pasoTiempo, float constGrav, int numBlocks, int blockSize)
{
    calculoNuevaPosicionGPU << <numBlocks, blockSize >> > (posAntGPU, posActGPU, posSigGPU, numParticulas, pasoTiempo, constGrav);
}

void lanzarKernelCalculoVelocidad(Coord* posAntGPU, Coord* posActGPU, Coord* posSigGPU, float* velocidadesGPU, int numParticulas, float pasoTiempo, float constGrav, int numBlocks, int blockSize)
{
    calculoVelocidadesGPU << <numBlocks, blockSize >> > (posAntGPU, posActGPU, posSigGPU, velocidadesGPU, numParticulas, pasoTiempo, constGrav);
}

void obtenerResultadoPosicionGPU(Coord* posAntGPU, Coord* posActGPU, Coord* posSigGPU, int numBytes, Coord* partAct)
{
    cudaDeviceSynchronize(); //Para asegurarnos que ya todos los calculos de las hebras de CUDA han terminado

    //Actualizamos los vectores de las posiciones de part�culas dentro de la memoria de la GPU
    checkCudaError(cudaMemcpy(posAntGPU, posActGPU, numBytes, cudaMemcpyDeviceToDevice), "cudaMemcpy3 posAntGPU");
    //Resultado del calculo se guarda en el vector de particulas ACTUAL
    checkCudaError(cudaMemcpy(partAct, posSigGPU, numBytes, cudaMemcpyDeviceToHost), "cudaMemcpy4 partAct");
    //Actualizamos los vectores de las posiciones de part�culas dentro de la memoria de la GPU
    checkCudaError(cudaMemcpy(posActGPU, posSigGPU, numBytes, cudaMemcpyDeviceToDevice), "cudaMemcpy5 posActGPU");

    //En principio el vector de part�culas SIGUIENTE se puede quedar con su contenido actual a modo de basura sin que en principio esto afecte al funcionamiento

}

void obtenerResultadoVelocidadGPU(Coord* posAntGPU, Coord* posActGPU, Coord* posSigGPU, float* velocidadesGPU, float* velocidadesHost, int numBytesParticulas, int numBytesVel)
{
    cudaDeviceSynchronize(); //Para asegurarnos que ya todos los calculos de las hebras de CUDA han terminado

    //Actualizamos los vectores de las posiciones de part�culas dentro de la memoria de la GPU
    checkCudaError(cudaMemcpy(posAntGPU, posActGPU, numBytesParticulas, cudaMemcpyDeviceToDevice), "cudaMemcpy3 posAntGPU");
    //Actualizamos los vectores de las posiciones de part�culas dentro de la memoria de la GPU
    checkCudaError(cudaMemcpy(posActGPU, posSigGPU, numBytesParticulas, cudaMemcpyDeviceToDevice), "cudaMemcpy4 posActGPU");
    
    //En principio el vector de part�culas SIGUIENTE se puede quedar con su contenido actual a modo de basura sin que en principio esto afecte al funcionamiento
    
    //Actualizamos el vector de velocidades en el host
    checkCudaError(cudaMemcpy(velocidadesHost, velocidadesGPU, numBytesVel, cudaMemcpyDeviceToHost), "cudaMemcpy5 velocidadesHost");
}

void liberarRecursosGPU(Coord* posAntGPU, Coord* posActGPU, Coord* posSigGPU, float* velocidadesGPU)
{
    cudaFree(posAntGPU);
    cudaFree(posActGPU);
    cudaFree(posSigGPU);
    cudaFree(velocidadesGPU);
}


