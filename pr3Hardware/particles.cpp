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
#include <random>

using namespace std;

#include "util.h"
#include "particles.h"

// Vertex shader for particle visualization
static const char *codVisVS = R"(
    #version 330

    layout(location = 0) in vec2 coord;

    void main() {
        gl_Position = vec4(coord, 0.0, 1.0);
        gl_PointSize = 10.0;
    }
)";


// Fragment shader for particle visualization
static const char *codVisFS = R"(
    #version 330

    out vec4 color;

    void main() {
        // Draw points as circles
        vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
        if (dot(circCoord, circCoord) > 1.0) {
            discard;
        }

       color = vec4(0.0, 0.0, 1.0, 0.25);
    }
)";


ParticlesApp::ParticlesApp(unsigned numParticles) :
            GLFWWindow(512, 512, "Particles"),
            numParticles(numParticles) {
    glfwSwapInterval(1); //To force VSync
    makeContextCurrent();
}


void ParticlesApp::displayFPS() {
    double currentTime = glfwGetTime();
    double delta = currentTime - tVisLastFPS;
    numVisFrames++;

    if (delta >= 1.0) {
        double fps = double(numVisFrames) / delta;
        std::stringstream ss;
        ss << "Particles" << " [" << fps << " FPS]";

        setTitle(ss.str().c_str());

        numVisFrames = 0;
        tVisLastFPS = currentTime;
    }
}


void ParticlesApp::prepareVisualization() {
    progVis = createShaderProgram(codVisVS, codVisFS);

    // Viewport activation
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Create and activate particle VAO
    glGenVertexArrays(1, &vaoVis);
    glBindVertexArray(vaoVis);

    // Create and activate VBO for particle coordinates, associated with particle VAO
    glGenBuffers(1, &vboVisCoords);
    glBindBuffer(GL_ARRAY_BUFFER, vboVisCoords);

    // Associate coordinate VBO with shader vertex parameter
    glEnableVertexAttribArray(attribVisCoord);
    glVertexAttribPointer(attribVisCoord, 2, GL_FLOAT, GL_FALSE, 0, NULL);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    tVisLastFPS = glfwGetTime();
    numVisFrames = 0;
}


void ParticlesApp::visualize() {
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);

    // Activate shaders
    glUseProgram(progVis);

    // Activate vertex array
    glBindVertexArray(vaoVis);
    glBindBuffer(GL_ARRAY_BUFFER, vboVisCoords);
    // Transfer updated coordinates to vbo
    glBufferData(GL_ARRAY_BUFFER, numParticles * sizeof(Coord), &coordsCurrent[0], GL_DYNAMIC_DRAW);

    glDrawArrays(GL_POINTS, 0, (GLsizei) numParticles);

    glBindVertexArray(0);
    glUseProgram(0);

    glDisable(GL_BLEND);
    glDisable(GL_PROGRAM_POINT_SIZE);
    swapBuffers();
}


void ParticlesApp::execute() {
#ifdef VISUALIZE
    // Simulation/rendering loop
    while (!shouldClose()) {
        simulate();
        visualize();
        displayFPS();

        glfwPollEvents();

    }
#else
    // Simulation-only loop
    std::cout << "Beginning of simulation." << std::endl << "Duration: " << SIM_DURATION << " sec." << std::endl << std::endl;

    /**
     * This is to properly manage simulation time.
     * Previously with timeStep it wasn't coordinated with real time,
     * so the duration would last longer than the desired seconds of execution
     */
    std::chrono::seconds duration(SIM_DURATION);
    auto start = std::chrono::steady_clock::now();

    while (std::chrono::steady_clock::now() - start < duration) {
        simulate();
        findMaxVelocity();
    }
    displayMaxVelocitySim();
#endif
}


void ParticlesApp::prepareSimulation() {
    coordsCurrent.clear();
    coordsPrevious.clear();
    float dt = SIM_TIME_STEP;

    // --- MODERN RANDOM GENERATOR ---
    std::random_device rd;
    std::mt19937 gen(rd()); // Mersenne Twister engine (Very precise)
    std::uniform_real_distribution<float> distRadius(0.1f, 0.9f);
    std::uniform_real_distribution<float> distAngle(0.0f, 6.28318f);

    for (unsigned c = 0; c < numParticles; ++c) {
        Coord current;
        float radius = distRadius(gen);
        float angle = distAngle(gen);

        current.x = radius * cos(angle);
        current.y = radius * sin(angle);
        coordsCurrent.push_back(current);

        // Orbital velocity without division by zero risk
        float orbitalVel = 0.3f / sqrt(radius);

        float vx = -sin(angle) * orbitalVel;
        float vy = cos(angle) * orbitalVel;

        Coord previous;
        previous.x = current.x - vx * dt;
        previous.y = current.y - vy * dt;
        coordsPrevious.push_back(previous);
    }

    velocitiesCurrent.resize(NUM_PARTICLES);

    // -----------------------------------------------------------------------
    // Call function here in external file to allocate GPU memory and
    // transfer particle positions to GPU in CUDA/OPENCL
    // -----------------------------------------------------------------------

    instantiateGPUMemory(&particlesGPUPrevious, &particlesGPUCurrent, &particlesGPUNext, &velocitiesGPU, numBytesParticles, numBytesVelocities, coordsPrevious.data(), coordsCurrent.data());

    // Start time counter
    t = 0;
}


void ParticlesApp::simulate() {

    // -----------------------------------------------------------------------
    // Call function here in external file to launch execution
    // in CUDA/OPENCL and transfer positions from GPU memory
    // to coords array
    // -----------------------------------------------------------------------

    int blockSize = 256;
    int numBlocks = (NUM_PARTICLES + blockSize - 1) / blockSize;

#ifdef VISUALIZE

    launchKernelCalculatePosition(particlesGPUPrevious, particlesGPUCurrent, particlesGPUNext, NUM_PARTICLES, SIM_TIME_STEP, CONST_GRAV, numBlocks, blockSize);
    getPositionResultGPU(particlesGPUPrevious, particlesGPUCurrent, particlesGPUNext, numBytesParticles, coordsCurrent.data());

#else
    launchKernelCalculateVelocity(particlesGPUPrevious, particlesGPUCurrent, particlesGPUNext, velocitiesGPU, NUM_PARTICLES, SIM_TIME_STEP, CONST_GRAV, numBlocks, blockSize);
    getVelocityResultGPU(particlesGPUPrevious, particlesGPUCurrent, particlesGPUNext, velocitiesGPU, velocitiesCurrent.data(), numBytesParticles, numBytesVelocities);

#endif
    // Advance time variable
    t += SIM_TIME_STEP;
}


void ParticlesApp::freeResources() {
#ifdef VISUALIZE
    glDeleteBuffers(1, &vboVisCoords);
    glDeleteVertexArrays(1, &vaoVis);
    glDeleteBuffers(1, &vboVisCoords);
    glDeleteProgram(progVis);
#endif

    // -----------------------------------------------------------------------
    // Call function here in external file to free resources
    // allocated by CUDA/OPENCL
    // -----------------------------------------------------------------------

    freeGPUResources(particlesGPUPrevious, particlesGPUCurrent, particlesGPUNext, velocitiesGPU);

}

void ParticlesApp::findMaxVelocity()
{
    for (int i = 0; i < NUM_PARTICLES; i++)
    {
        if (velocitiesCurrent[i] > maxVelocity)
        {
            maxVelocity = velocitiesCurrent[i];
            fastestParticle = i;
        }
    }
}

void ParticlesApp::displayMaxVelocitySim()
{
    std::cout << "The maximum velocity of the simulation was " << maxVelocity << " m/s from particle number " << fastestParticle << "." << std::endl;
    system("PAUSE");
}


int main(int argc, char** argv) {
    // GLFW initialization
    if (glfwInit() != GL_TRUE) {
        std::cerr << "GLFW initialization failed" << std::endl;
        return 1;
    }

    // Application & window init
    ParticlesApp::hint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    ParticlesApp::hint(GLFW_CONTEXT_VERSION_MINOR, 1);
    ParticlesApp::hint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    ParticlesApp::hint(GLFW_RESIZABLE, GLFW_FALSE);

#ifndef VISUALIZE
    ParticlesApp::hint(GLFW_VISIBLE, GLFW_FALSE);
#endif

#ifdef __APPLE__
    ParticlesApp::hint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    ParticlesApp particlesApp(NUM_PARTICLES);

    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
        std::cerr << "Failed to load OpenGL" << std::endl;
        return 1;
    }

    particlesApp.prepareSimulation();
#ifdef VISUALIZE
    particlesApp.prepareVisualization();
#endif
    particlesApp.execute();

    // Terminate GLFW
    glfwTerminate();

    return 0;
}

