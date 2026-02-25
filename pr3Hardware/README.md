# Particle Simulation with CUDA

A GPU-accelerated N-body gravitational particle simulation application using NVIDIA CUDA and OpenGL visualization.

## Overview

This project implements an interactive particle simulation that models gravitational interactions between particles in 2D space. The simulation uses NVIDIA CUDA for high-performance GPU computation and OpenGL with GLFW for real-time visualization.

### Key Features

- **GPU-Accelerated Computation**: Leverages NVIDIA CUDA for efficient parallel computation of gravitational forces
- **Verlet Integration**: Uses the Verlet integration method for stable particle dynamics
- **Plummer Softening**: Implements softening to prevent singularities and extreme accelerations
- **Real-time Visualization**: Interactive OpenGL rendering with GLFW window management
- **Particle Physics**:
  - Gravitational force calculations
  - Velocity computations
  - Boundary collision detection with elastic bouncing

## Project Structure

```
pr3Hardware/
├── CMakeLists.txt                 # CMake build configuration
├── config.h                        # Project configuration header
├── funcionesGPU.cu                 # CUDA kernel implementations
├── funcionesGPU.cuh                # CUDA function declarations
├── Header.cuh                      # Additional CUDA headers
├── particulas.cpp                  # Main particle system implementation
├── particulas.h                    # Particle class definition
├── util.cpp                        # Utility functions
├── util.h                          # Utility function declarations
├── glfwindow/                      # GLFW window management
│   ├── glfwwindow.cpp
│   └── glfwwindow.h
├── glad/                           # OpenGL loader (GLAD)
├── glfw/                           # GLFW library headers and static library
└── glm/                            # GLM vector/matrix mathematics library
```

## Requirements

### Hardware
- NVIDIA GPU with CUDA Compute Capability 3.0 or higher
- 2GB+ VRAM (depending on particle count)

### Software
- **CUDA Toolkit** 11.0 or higher
- **CMake** 3.18 or higher
- **Visual Studio** 2019+ (on Windows) or GCC/Clang (on Linux)
- **OpenGL** 3.3+
- **GLFW** 3.x library
- **GLM** vector mathematics library
- **GLAD** OpenGL loader

## Building the Project

### Windows (Visual Studio)

1. Ensure CUDA Toolkit is installed and properly configured
2. Create a build directory:
   ```bash
   mkdir build
   cd build
   ```
3. Generate Visual Studio project files:
   ```bash
   cmake .. -G "Visual Studio 16 2019" -A x64
   ```
4. Build the project:
   ```bash
   cmake --build . --config Debug
   ```
5. Run the executable:
   ```bash
   ./Debug/particles_sim_cuda.exe
   ```

### Linux/macOS

1. Install dependencies:
   ```bash
   sudo apt-get install cuda-toolkit libglfw3-dev libglm-dev
   ```
2. Create and build:
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ./particles_sim_cuda
   ```

## How It Works

### GPU Computation Pipeline

1. **Memory Allocation**: Particle positions and velocities are allocated on GPU memory
2. **Kernel Execution**: Two main CUDA kernels compute:
   - `calculoNuevaPosicionGPU`: Calculates new particle positions using Verlet integration
   - `calculoVelocidadesGPU`: Computes particle velocities from position differences
3. **Force Calculation**: 
   - For each particle, gravitational forces from all other particles are computed
   - Plummer softening with epsilon ≈ 0.05 prevents singularities
   - Accelerations are applied using the Verlet integration method
4. **Results Transfer**: Computed positions and velocities are copied back to CPU/memory

### Physics Parameters

- **Plummer Softening Parameter (ε)**: 0.05
  - Controls the "softness" of gravitational interactions
  - Prevents infinite accelerations when particles get very close
  
- **Gravity Constant**: User-configurable
  - Controls the overall strength of gravitational attraction
  
- **Time Step (Δt)**: User-configurable
  - Smaller values = more accurate but slower simulation
  
- **Bounce Factor**: 0.8
  - Coefficient of restitution for boundary collisions
  - Range: 0.0 (absorb) to 1.0 (perfectly elastic)

- **Boundary Limits**: ±1.0 in both X and Y axes

## CUDA Kernel Details

### `calculoNuevaPosicionGPU`
Computes new particle positions using the Verlet integration formula:
```
x(t+1) = 2*x(t) - x(t-1) + a(t)*Δt²
```

Features:
- Plummer softening for stable gravitational force calculation
- Boundary collision detection with velocity reversal
- One thread per particle for maximum parallelism

### `calculoVelocidadesGPU`
Calculates particle velocities from position derivatives:
```
v = (x(t+1) - x(t)) / Δt
```

Features:
- Computes forces with minDist threshold (0.002) to avoid singularities
- Calculates velocity magnitude from position differences
- Updates all particle velocity data

## Configuration

Edit `config.h` to customize:
- Number of particles
- Simulation time step
- Gravity constant
- Initial particle distribution
- Other runtime parameters

## Controls

*(Specific controls depend on the GLFWWindow implementation. Common controls typically include:)*
- Mouse: Interactive particle manipulation (if implemented)
- Keyboard: Pause/resume, speed adjustment, camera control
- ESC: Exit application

## Performance Optimization Tips

1. **Block Size**: Adjust CUDA block size in `lanzarKernel*` functions (currently 256 threads)
2. **Particle Count**: Start with fewer particles to ensure smooth performance
3. **Time Step**: Larger time steps are faster but less accurate
4. **Softening Parameter**: Higher epsilon = faster computation but less realistic
5. **GPU Memory**: Monitor VRAM usage with larger particle counts

## Known Issues & Limitations

- Maximum performance scales with GPU compute capability
- Very high particle counts (>100k) may cause memory issues
- Plummer softening may make gravity appear less realistic
- 2D simulation (extension to 3D is possible with minor modifications)

## License

This project is licensed under the GNU General Public License v3.0 or later.

```
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
```

## Authors

- **Original Author**: Antonio J. Rueda
- **CUDA Implementation**: fotop
- **Date**: June 2024

## Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## References

- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [Verlet Integration](https://en.wikipedia.org/wiki/Verlet_integration)
- [Plummer Softening](https://en.wikipedia.org/wiki/Gravitational_softening)
- [GLFW Documentation](https://www.glfw.org/documentation.html)
- [GLM Mathematics Library](https://glm.g-truc.net/)

## Support

For issues or questions, please check:
1. The project's issue tracker
2. CUDA and OpenGL documentation
3. GLFW and GLM documentation

---

**Last Updated**: March 2026

