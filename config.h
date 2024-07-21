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


#ifndef config_h
#define config_h

/* Parámetros de simulación */
const float NUM_PARTICULAS = 5000;
const float SIM_PASO_TIEMPO = 0.0005f;
//const float SIM_PASO_TIEMPO = 0.001f;

//const float CONST_GRAV = 6.67430e-11f;
const float CONST_GRAV = 0.005f;
//const float CONST_GRAV = 0.1f;
//const float CONST_GRAV = 1;

//duración en SEGUNDOS
const int SIM_DURACION = 5;

#define VISUALIZAR

#endif
