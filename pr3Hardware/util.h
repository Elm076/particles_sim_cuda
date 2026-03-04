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

#ifndef util_h
#define util_h

#include "glad/glad/glad.h"

const unsigned SIM_DATA_TEXTURE_WIDTH = 512;

float random(float min, float max);

void checkShaderError(GLint status, GLint shader, const char *msg);

void checkShaderProgramError(GLint status, GLint program, const char *msg);

/** Create program with vertex and fragment shader
 @arg vs Vertex shader code
 @arg fs Fragment shader code */
GLuint createShaderProgram(const char *vs, const char *fs);

/** Create 2D square texture of vec2 vectors
 @arg size Texture width and height
 @arg data Pointer to data array to initialize texture (optional)
 @return Texture identifier */
GLuint createTexture2DVec2(unsigned size, const float *data = 0);

#endif
