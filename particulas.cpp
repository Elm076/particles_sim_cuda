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

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <sstream>
#include <algorithm>
#include <chrono>

using namespace std;

#include "util.h"
#include "particulas.h"

// Vertex shader para visualización de partículas
static const char *codVisVS = R"(
    #version 330
              
    layout(location = 0) in vec2 coord;
                                               
    void main() {
        gl_Position = vec4(coord, 0.0, 1.0);
        gl_PointSize = 10.0;
    }
)";


// Fragment shader para visualización de partículas
static const char *codVisFS = R"(
    #version 330
                                               
    out vec4 color;
                                               
    void main() {
        // Dibujar los puntos como círculos
        vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
        if (dot(circCoord, circCoord) > 1.0) {
            discard;
        }
                                               
       color = vec4(0.0, 0.0, 1.0, 0.25);
    }
)";


ParticulasApp::ParticulasApp(unsigned numParticulas) :
            GLFWWindow(512, 512, "Partículas"),
            numParticulas(numParticulas) {
    glfwSwapInterval(0);
    makeContextCurrent();
}


void ParticulasApp::mostrarFR() {
    double tiempoActual = glfwGetTime();
    double delta = tiempoActual - tVisUltimoFR;
    numVisFrames++;
    
    if (delta >= 1.0) {
        double fps = double(numVisFrames) / delta;
        std::stringstream ss;
        ss << "Partículas" << " [" << fps << " FPS]";
        
        setTitle(ss.str().c_str());
        
        numVisFrames = 0;
        tVisUltimoFR = tiempoActual;
    }
}


void ParticulasApp::prepararVisualizacion() {
    progVis = crearProgramaShader(codVisVS, codVisFS);
    
    // Activación de viewport
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // Crear y activar VAO de partículas
    glGenVertexArrays(1, &vaoVis);
    glBindVertexArray(vaoVis);

    // Crear y activar VBO para coordenadas de las partículas, asociado al VAO de las partículas
    glGenBuffers(1, &vboVisCoords);
    glBindBuffer(GL_ARRAY_BUFFER, vboVisCoords);

    // Asociar VBO de coordenadas al parámetro vertex del shader
    glEnableVertexAttribArray(attribVisCoord);
    glVertexAttribPointer(attribVisCoord, 2, GL_FLOAT, GL_FALSE, 0, NULL);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
            
    tVisUltimoFR = glfwGetTime();
    numVisFrames = 0;
}


void ParticulasApp::visualizar() {    
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);

    // Activar shaders
    glUseProgram(progVis);
    
    // Activar array de vértices
    glBindVertexArray(vaoVis);
    glBindBuffer(GL_ARRAY_BUFFER, vboVisCoords);
    // Transferir coordenadas actualizadas al vbo
    glBufferData(GL_ARRAY_BUFFER, numParticulas * sizeof(Coord), &coordsAct[0], GL_DYNAMIC_DRAW);

    glDrawArrays(GL_POINTS, 0, (GLsizei) numParticulas);
    
    glBindVertexArray(0);
    glUseProgram(0);

    glDisable(GL_BLEND);
    glDisable(GL_PROGRAM_POINT_SIZE);
    swapBuffers();
}


void ParticulasApp::ejecutar() {
#ifdef VISUALIZAR
    // Bucle de simulación/renderizado
    while (!shouldClose()) {
        simular();
        visualizar();
        mostrarFR();
        
        glfwPollEvents();
    }
#else
    // Bucle de solo simulación
    std::cout << "Comienzo de la simulacion." << std::endl << "Duracion: " << SIM_DURACION << " seg." << std::endl << std::endl;

    /**
     * Esto es para gestionar el tiempo de la simulación correctamente.
     * Antes haciéndolo con pasoTiempo no iba coordinado con el tiempo en la vida real,
     * por lo tanto la duración duraba más de los segundos que quisieramos ejecutarla
     */
    std::chrono::seconds duracion(SIM_DURACION);
    auto inicio = std::chrono::steady_clock::now();

    while (std::chrono::steady_clock::now() - inicio < duracion) {
        simular();
        encontrarVelMax();
    }
    mostrarVelocidadMaxSim();
#endif
}


void ParticulasApp::prepararSimulacion() {
    coordsAct.clear();
    for (unsigned c = 0; c < numParticulas; ++c) {
        Coord nuevaCoordenada;
        nuevaCoordenada.x = aleatorio(-1.0f, 1.0f); nuevaCoordenada.y = aleatorio(-1.0f, 1.0f);
        coordsAct.push_back(nuevaCoordenada);
    }

    velocidadesAct.resize(NUM_PARTICULAS);
    
    // -----------------------------------------------------------------------
    // LLamar aquí a función en fichero externo para reservar memoria en GPU y
    // transferir la posición de las partículas a GPU en CUDA/OPENCL
    // -----------------------------------------------------------------------

    instanciarMemGPU(&particulasGPUAnterior, &particulasGPUActual, &particulasGPUSiguiente, &velocidadesGPU, numBytesParticulas, numBytesVelocidades, coordsAct.data());

    // Iniciar contador de tiempo
    t = 0;
}


void ParticulasApp::simular() {

    // -----------------------------------------------------------------------
    // Llamar aquí a función en fichero externo para lanzar la ejecución
    // en CUDA/OPENCL y transferir posiciones desde memoria de la GPU
    // al array coords
    // -----------------------------------------------------------------------
        
    int blockSize = 256;
    int numBlocks = (NUM_PARTICULAS + blockSize - 1) / blockSize;

#ifdef VISUALIZAR

    lanzarKernelCalculoPosicion(particulasGPUAnterior, particulasGPUActual, particulasGPUSiguiente, NUM_PARTICULAS, SIM_PASO_TIEMPO, CONST_GRAV, numBlocks, blockSize);
    obtenerResultadoPosicionGPU(particulasGPUAnterior, particulasGPUActual, particulasGPUSiguiente, numBytesParticulas, coordsAct.data());

#else
    lanzarKernelCalculoVelocidad(particulasGPUAnterior, particulasGPUActual, particulasGPUSiguiente, velocidadesGPU, NUM_PARTICULAS, SIM_PASO_TIEMPO, CONST_GRAV, numBlocks, blockSize);
    obtenerResultadoVelocidadGPU(particulasGPUAnterior, particulasGPUActual, particulasGPUSiguiente, velocidadesGPU, velocidadesAct.data(), numBytesParticulas, numBytesVelocidades);

#endif
    // Avanzar variable de tiempo
    t += SIM_PASO_TIEMPO;
}
    

void ParticulasApp::liberarRecursos() {
#ifdef VISUALIZAR
    glDeleteBuffers(1, &vboVisCoords);
    glDeleteVertexArrays(1, &vaoVis);
    glDeleteBuffers(1, &vboVisCoords);
    glDeleteProgram(progVis);
#endif
    
    // -----------------------------------------------------------------------
    // Llamar aquí a función en fichero externo para liberar los recursos
    // asignados por CUDA/OPENCL
    // -----------------------------------------------------------------------

    liberarRecursosGPU(particulasGPUAnterior, particulasGPUActual, particulasGPUSiguiente, velocidadesGPU);

}

void ParticulasApp::encontrarVelMax()
{
    for (int i = 0; i < NUM_PARTICULAS; i++)
    {
        if (velocidadesAct[i] > velocidadMax)
        {
            velocidadMax = velocidadesAct[i];
            partMasRapida = i;
        }
    }
}

void ParticulasApp::mostrarVelocidadMaxSim()
{
    std::cout << "La velocidad maxima de la simulacion ha sido de " << velocidadMax << " m/s por la particula numero " << partMasRapida << "." << std::endl;
    system("PAUSE");
}


int main(int argc, char** argv) {
    // Iniciación glfw
    if (glfwInit() != GL_TRUE) {
        std::cerr << "Fallo en inicialización de GLFW" << std::endl;
        return 1;
    }
        
    // Application & window init
    ParticulasApp::hint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    ParticulasApp::hint(GLFW_CONTEXT_VERSION_MINOR, 1);
    ParticulasApp::hint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    ParticulasApp::hint(GLFW_RESIZABLE, GLFW_FALSE);
    
#ifndef VISUALIZAR
    ParticulasApp::hint(GLFW_VISIBLE, GLFW_FALSE);
#endif

#ifdef __APPLE__
    ParticulasApp::hint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
 
    ParticulasApp particulasApp(NUM_PARTICULAS);
    
    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
        std::cerr << "Fallo al cargar OpenGL" << std::endl;
        return 1;
    }

    particulasApp.prepararSimulacion();
#ifdef VISUALIZAR
    particulasApp.prepararVisualizacion();
#endif
    particulasApp.ejecutar();
    
    // Terminar GLFW
    glfwTerminate();
    
    return 0;
}
