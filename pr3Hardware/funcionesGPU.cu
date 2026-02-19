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
       /*
        __shared Coord particulasBloque[256];

        int inicioBloque = 0;


        particulasBloque[threadIdx.x] = postActGPU[inicioBloque + threadIdx.x]
        _    
        */
        Coord Pi = posActGPU[index];
        Coord PiAnt = posAntGPU[index];
        Coord fuerza = { 0.0f, 0.0f };
        float minDist = 0.0012f; //aquí tengo que ir ajustando este parámetro. CUÁNTO MÁS PEQUEÑO MÁS SE DISPERSA
        
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
        //Integración de Verlet
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
        float minDist = 0.002f; //aquí tengo que ir ajustando este parámetro. CUÁNTO MÁS PEQUEÑO MÁS SE DISPERSA
        
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
        //Integración de Verlet
        posSigGPU[index].x = 2 * Pi.x - PiAnt.x + (pasoTiempo * pasoTiempo) * fuerza.x;
        posSigGPU[index].y = 2 * Pi.y - PiAnt.y + (pasoTiempo * pasoTiempo) * fuerza.y;

        //Cálculo de la velocidad de cada partícula
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

    //Actualizamos los vectores de las posiciones de partículas dentro de la memoria de la GPU
    checkCudaError(cudaMemcpy(posAntGPU, posActGPU, numBytes, cudaMemcpyDeviceToDevice), "cudaMemcpy3 posAntGPU");
    //Resultado del calculo se guarda en el vector de particulas ACTUAL
    checkCudaError(cudaMemcpy(partAct, posSigGPU, numBytes, cudaMemcpyDeviceToHost), "cudaMemcpy4 partAct");
    //Actualizamos los vectores de las posiciones de partículas dentro de la memoria de la GPU
    checkCudaError(cudaMemcpy(posActGPU, posSigGPU, numBytes, cudaMemcpyDeviceToDevice), "cudaMemcpy5 posActGPU");

    //En principio el vector de partículas SIGUIENTE se puede quedar con su contenido actual a modo de basura sin que en principio esto afecte al funcionamiento

}

void obtenerResultadoVelocidadGPU(Coord* posAntGPU, Coord* posActGPU, Coord* posSigGPU, float* velocidadesGPU, float* velocidadesHost, int numBytesParticulas, int numBytesVel)
{
    cudaDeviceSynchronize(); //Para asegurarnos que ya todos los calculos de las hebras de CUDA han terminado

    //Actualizamos los vectores de las posiciones de partículas dentro de la memoria de la GPU
    checkCudaError(cudaMemcpy(posAntGPU, posActGPU, numBytesParticulas, cudaMemcpyDeviceToDevice), "cudaMemcpy3 posAntGPU");
    //Actualizamos los vectores de las posiciones de partículas dentro de la memoria de la GPU
    checkCudaError(cudaMemcpy(posActGPU, posSigGPU, numBytesParticulas, cudaMemcpyDeviceToDevice), "cudaMemcpy4 posActGPU");
    
    //En principio el vector de partículas SIGUIENTE se puede quedar con su contenido actual a modo de basura sin que en principio esto afecte al funcionamiento
    
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


