/*****************************************************************//**
 * @file   funcionesGPU.cuh
 * @brief  Cabecera de las funciones necesarias para la simulaci�n de las part�culas
 * @author fotop
 * @date   June 2024
 *********************************************************************/

#ifndef funcionesGPU_cuh
#define funcionesGPU_cuh

#include <cuda_runtime.h>
#include <device_launch_parameters.h> //me ha hecho falta incluirla para que me reconozca las variables de identificaci�n de bloque, thread, etc.
#include <glm/fwd.hpp>
#include <glm/vec2.hpp> //estas dos cabeceras de glm es para poder usar el tipo Coord como se especifica abajo
#include <math.h>
#include <iostream>

using Coord = glm::vec2;

/** Funci�n para verificar errores de CUDA */
void checkCudaError(cudaError_t result, const char* msg);

/** Funci�n para establecer la memoria necesaria en GPU para la simulaci�n*/
void instanciarMemGPU(Coord** posAnt, Coord** posAct, Coord** posSig, float** velocidades, unsigned int numBytesPos, unsigned int numBytesVel, Coord* partAnt, Coord* partAct);

/** Funci�n que define el kernel para el c�lculo de la nueva posici�n de las part�culas */
extern "C" __global__ void calculoNuevaPosicionGPU(Coord* posAntGPU, Coord* posActGPU, Coord* posSigGPU, int numParticulas, float pasoTiempo, float constGrav);

/** Funci�n que define el kernel para el c�lculo de las velocidades de las part�culas */
extern "C" __global__ void calculoVelocidadesGPU(Coord* posAntGPU, Coord* posActGPU, Coord* posSigGPU, float* velocidadesGPU, int numParticulas, float pasoTiempo, float constGrav);

/** Funci�n para lanzar el kernel desde el archivo .cpp */
void lanzarKernelCalculoPosicion(Coord* posAnt, Coord* posAct, Coord* posSig, int numParticulas, float pasoTiempo, float constGrav, int numBlocks, int blockSize);

/** Funci�n para lanzar el kernel desde el archivo .cpp */
void lanzarKernelCalculoVelocidad(Coord* posAntGPU, Coord* posActGPU, Coord* posSigGPU, float* velocidadesGPU, int numParticulas, float pasoTiempo, float constGrav, int numBlocks, int blockSize);

/** Funci�n para pasar el resultado del c�lculo de las nuevas posiciones de la GPU a CPU */
void obtenerResultadoPosicionGPU(Coord* posAnt, Coord* posAct, Coord* posSig, int numBytes, Coord* partAct);

/** Funci�n para pasar el resultado del c�lculo de las velocidades de la GPU a CPU */
void obtenerResultadoVelocidadGPU(Coord* posAnt, Coord* posAct, Coord* posSig, float* velocidadesGPU, float* velocidadesHost, int numBytesParticulas, int numBytesVel);

/** Funci�n para liberar la memoria asignada a la gpu para la simulaci�n */
void liberarRecursosGPU(Coord* posAnt, Coord* posAct, Coord* posSig, float* velocidadesGPU);

#endif