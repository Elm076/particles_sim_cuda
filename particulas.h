//  Created by Antonio J. Rueda on 21/3/2024.
//  Copyright © 2024 Antonio J. Rueda.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#ifndef particulas_h
#define particulas_h

#include <glm/glm.hpp>
#include "glad/glad/glad.h"
#include "glfwindow/glfwwindow.h"
#include "config.h"
#include "funcionesGPU.cuh"

using namespace glm;

class ParticulasApp : public GLFWWindow {
    
    using Coord = vec2;
        
    /** Calcular frames por segundo */
    void mostrarFR();

    /** Liberar recursos asignados */
    void liberarRecursos();
        
    /** Variables visualización --------------------------- */
    
    /** Programa shader visualización */
    GLuint progVis;
    
    /** Parámetro del shader asociado a las coordenadas de la partícula */
    const GLuint attribVisCoord = 0;
    
    /** VAO & array de vértices para particulas */
    GLuint vaoVis, vboVisCoords;

    /** Otros atributos -------------------------------------- */
    
    int numParticulas = NUM_PARTICULAS;
    
    /** Array de coordenadas de las partículas. NOTA: SOLO HACE FALTA UNO EN LA PARTE DEL HOST PARA HACER LA GESTIÓN DE ESTAS */
    std::vector<Coord> coordsAct;

    /** Array de velocidades de las partículas. Necesario para el cálculo de la partícula más rápida en el tiempo de simulación */
    std::vector<float> velocidadesAct;

    /** Velocidad máxima y partícula asociada que vamos a almacenar en cada ejecución */
    float velocidadMax = 0;
    int partMasRapida;

    /**
     * Punteros para reserva de memoria de las partículas en la GPU.
     * NOTA: NECESITAMOS 3 INSTANCIAS TEMPORALES DE LAS POSICIONES DE LAS PARTÍCULAS PARA HACER LA INTEGRACIÓN DE VERLET
     */
    Coord* particulasGPUAnterior;
    Coord* particulasGPUActual;
    Coord* particulasGPUSiguiente;
    float* velocidadesGPU;

    /** Tamanio en Bytes de la estructura de las partículas y velocidades */
    unsigned int numBytesParticulas = sizeof(Coord) * numParticulas;
    unsigned int numBytesVelocidades = sizeof(float) * numParticulas;

    /** Tiempo de simulación */
    float t;
    
    /** Variables auxilisares F PS */
    double tVisUltimoFR;
    int numVisFrames;
    
public:
    /** Constructor */
    ParticulasApp(unsigned numParticulas);
    
    /** Preparar escena */
    void prepararVisualizacion();

    /** Dibujar escena */
    void visualizar();
    
    /** Establecer estado inicial de las partículas */
    void prepararSimulacion();
    
    /** Recalcular nueva posición de partículas */
    void simular();

    /** Bucle de rendering y simulación */
    void ejecutar();

    /** Calcular velocidad máxima después de cada ejecución de la simulación */
    void encontrarVelMax();

    /** Método para mostrar la velocidad máxima de la simulación y la partícula que la ha generado */
    void mostrarVelocidadMaxSim();

    /** Getter para la velocidad máxima */
    float getVelMax()
    {
        return velocidadMax;
    }
};

#endif
