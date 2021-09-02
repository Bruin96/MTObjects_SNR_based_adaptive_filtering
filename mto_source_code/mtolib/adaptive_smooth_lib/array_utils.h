#ifndef ARRAY_UTILS_H
#define ARRAY_UTILS_H

float** init_array(int width, int height);
void free_array(float**im, int height);
void swap(float** a, float** b);
float** reflection_padding(float** im, int width, int height, int pad_size);
float** convert1Ddoubleto2Dfloat(double* im, int height, int width);
double* convert2Dfloatto1Ddouble(float** im, int height, int width);

#endif
