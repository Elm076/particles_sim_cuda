cmake_minimum_required(VERSION 3.18)
# IMPORTANT: This tells CMake to search for the CUDA compiler (nvcc)
project(particles_sim_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Include local folders so it finds glfw, glm, etc.
include_directories(.)
include_directories(${CMAKE_SOURCE_DIR}/glfw)
include_directories(${CMAKE_SOURCE_DIR}/glm)

# Here we put all the source files
add_executable(particles_sim_cuda
        config.h
        gpuFunctions.cu
        gpuFunctions.cuh
        Header.cuh
        particles.cpp
        particles.h
        util.cpp
        util.h
        glfwindow/glfwwindow.cpp
        glfwindow/glfwwindow.h
        glad/glad.cpp
)

# Enable separable compilation for CUDA
set_target_properties(particles_sim_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Find native system OpenGL
find_package(OpenGL REQUIRED)

# Link static libraries we have in the project and OpenGL
# Note: I assume you have glfw3.lib in the glfw/ folder
target_link_libraries(particles_sim_cuda PRIVATE
        OpenGL::GL
        ${CMAKE_SOURCE_DIR}/glfw/glfw3.lib
)

# Tell the project to include CUDA's default folders
target_include_directories(particles_sim_cuda PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})