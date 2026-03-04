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

#include "util.h"
#include <sys/stat.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <random>

float random(float min, float max) {
    return min + (max - min) * rand() / RAND_MAX;
}

void checkShaderError(GLint status, GLint shader, const char *msg) {
    if (status == GL_FALSE) {
        std::cerr << msg << std::endl;
        
        char log[1024];
        GLsizei written;
        glGetShaderInfoLog(shader, sizeof(log), &written, log);
        std::cerr << "Log: " << log << std::endl;
        exit(1);
    }
}

void checkShaderProgramError(GLint status, GLint program, const char *msg) {
    if (status == GL_FALSE) {
        std::cerr << msg << std::endl;
        
        char log[1024];
        GLsizei written;
        glGetProgramInfoLog(program, sizeof(log), &written, log);
        std::cerr << "Log: " << log << std::endl;
        exit(1);
    }
}

GLuint createShaderProgram(const char *vs, const char *fs) {
    GLint status;
    
    // Shader preparation
    GLint vShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vShader, 1, (const GLchar **) &vs, 0);
    glCompileShader(vShader);
    glGetShaderiv(vShader, GL_COMPILE_STATUS, &status);
    checkShaderError(status, vShader, "Vertex shader compilation error.");

    GLint fShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fShader, 1, (const GLchar **) &fs, 0);
    glCompileShader(fShader);
    glGetShaderiv(fShader, GL_COMPILE_STATUS, &status);
    checkShaderError(status, fShader, "Fragment shader compilation error.");

    GLuint program = glCreateProgram();
    glAttachShader(program, vShader);
    glAttachShader(program, fShader);

    glLinkProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    checkShaderProgramError(status, program, "Shader program linking error");

    glDetachShader(program, vShader);
    glDetachShader(program, fShader);
    glDeleteShader(vShader);
    glDeleteShader(fShader);

    return program;
}

GLuint createTexture2DVec2(unsigned size, const float *data) {
    GLuint tex;

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, SIM_DATA_TEXTURE_WIDTH, ceil((float)size / SIM_DATA_TEXTURE_WIDTH), 0, GL_RG, GL_FLOAT, data);
    glBindTexture(GL_TEXTURE_2D, 0);

    return tex;
}

