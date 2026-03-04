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

#ifndef particles_h
#define particles_h

#include "glm/glm/glm.hpp"
#include "glad/glad/glad.h"
#include "glfwindow/glfwwindow.h"
#include "config.h"
#include "gpuFunctions.cuh"

using namespace glm;

class ParticlesApp : public GLFWWindow {

    using Coord = vec2;

    /** Calculate frames per second */
    void displayFPS();

    /** Free allocated resources */
    void freeResources();

    /** Visualization variables --------------------------- */

    /** Shader program for visualization */
    GLuint progVis;

    /** Shader parameter associated with particle coordinates */
    const GLuint attribVisCoord = 0;

    /** VAO & vertex array for particles */
    GLuint vaoVis, vboVisCoords;

    /** Other attributes -------------------------------------- */

    int numParticles = NUM_PARTICLES;

    /** Array of particle coordinates. NOTE: ONLY ONE IS NEEDED ON THE HOST SIDE FOR MANAGEMENT */
    std::vector<Coord> coordsCurrent;
    std::vector<Coord> coordsPrevious; // To store previous positions

    /** Array of particle velocities. Necessary for calculating the fastest particle during simulation time */
    std::vector<float> velocitiesCurrent;

    /** Maximum velocity and associated particle that we will store in each execution */
    float maxVelocity = 0;
    int fastestParticle;

    /**
     * Pointers for GPU memory allocation of particles.
     * NOTE: WE NEED 3 TEMPORARY INSTANCES OF PARTICLE POSITIONS FOR VERLET INTEGRATION
     */
    Coord* particlesGPUPrevious;
    Coord* particlesGPUCurrent;
    Coord* particlesGPUNext;
    float* velocitiesGPU;

    /** Size in Bytes of particle and velocity structures */
    unsigned int numBytesParticles = sizeof(Coord) * numParticles;
    unsigned int numBytesVelocities = sizeof(float) * numParticles;

    /** Simulation time */
    float t;

    /** Auxiliary variables for FPS */
    double tVisLastFPS;
    int numVisFrames;

public:
    /** Constructor */
    ParticlesApp(unsigned numParticles);

    /** Prepare visualization scene */
    void prepareVisualization();

    /** Draw scene */
    void visualize();

    /** Set initial state of particles */
    void prepareSimulation();

    /** Recalculate new particle positions */
    void simulate();

    /** Rendering and simulation loop */
    void execute();

    /** Calculate maximum velocity after each simulation execution */
    void findMaxVelocity();

    /** Method to display the maximum velocity of the simulation and the particle that generated it */
    void displayMaxVelocitySim();

    /** Getter for maximum velocity */
    float getMaxVel()
    {
        return maxVelocity;
    }
};

#endif

