#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "error.cuh"

#define MESSAGE 1

void read_matrix(int** r_ptr, int** c_ind, float** v, const char* fname, int* r_count, int* v_count);
__global__ void mat_vector_multiply(const int num_rows, const int* ptr, const int* indices, const float* data, const float* x, float* y);
int main() {
    int* row_ptr;
    int* col_ind;
    float* values;
    int r_count, v_count;
    const char* fname = "matrix.txt";

    read_matrix(&row_ptr, &col_ind, &values, fname, &r_count, &v_count);
    float* x = (float*)malloc(r_count * sizeof(float));
    float* y = (float*)calloc(r_count, sizeof(float));
    for (int i = 0; i < r_count; i++) {
        x[i] = 1.0;
    }
#if MESSAGE
    fprintf(stdout, "Initial Matrix\n");
    for (int i = 0; i < r_count; i++) {
        if (i + 1 <= r_count) {
            for (int k = row_ptr[i]; k < row_ptr[i + 1]; k++) {
                fprintf(stdout, "%d %d %.10lf\n", i, col_ind[k], values[k]);
            }
        }
    }
    fprintf(stdout, "Initial vector\n");
    for (int i = 0; i < r_count; i++) {
        fprintf(stdout, "%f\n", x[i]);
    }
#endif

    int *d_row_ptr, *d_col_ind;
    float *d_values, *d_x, *d_y;
    CHECK(cudaMalloc((void**)&d_row_ptr, (r_count+1) * sizeof(int)));
    CHECK(cudaMalloc((void**)&d_col_ind, v_count * sizeof(int)));
    CHECK(cudaMalloc(&d_values, v_count*sizeof(float)));
    CHECK(cudaMalloc((void**)&d_x, r_count * sizeof(float)));
    CHECK(cudaMalloc((void**)&d_y, r_count * sizeof(float)));

    CHECK(cudaMemcpy(d_row_ptr, row_ptr, (r_count+1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_col_ind, col_ind, v_count * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_values, values, v_count * sizeof(float), cudaMemcpyHostToDevice));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    CHECK(cudaEventRecord(start));

    CHECK(cudaMemcpy(d_x, x, r_count * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, y, r_count * sizeof(float), cudaMemcpyHostToDevice));

    int blocksize = 64;
    int blocknum = (r_count + blocksize - 1) / blocksize;
    mat_vector_multiply<<<blocknum, blocksize>>>(r_count, d_row_ptr, d_col_ind, d_values, d_x, d_y);
    CHECK(cudaMemcpy(y, d_y, r_count * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaEventRecord(stop));
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
#if MESSAGE
    int count=0;
    fprintf(stdout, "Resulting Vector\n");
    for (int i = 0; i < r_count; i++) {
        if (y[i] != 0) {
            fprintf(stdout, "%.10f\n", y[i]);
            count++;
        }
    }
    fprintf(stdout, "count = %d\n", count);
#endif
    fprintf(stdout, "time = %f\n", milliseconds);
    CHECK(cudaFree(d_row_ptr));
    CHECK(cudaFree(d_col_ind));
    CHECK(cudaFree(d_values));
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));
    free(x);
    free(y);
    free(row_ptr);
    free(col_ind);
    free(values);
    return 0;
}

__global__ void mat_vector_multiply(const int num_rows, const int* ptr, const int* indices, const float* data, const float* x, float* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int row_start, row_end;
    float dot = 0;
    if (row < num_rows) {
        row_start = ptr[row];
        row_end = ptr[row + 1];
        for (int i = row_start; i < row_end; i++) {
            dot += data[i] * x[indices[i]];
        }
    }
    y[row] += dot;
}

void read_matrix(int** r_ptr, int** c_ind, float** v, const char* fname, int* r_count, int* v_count) {
    FILE* file;
    if (NULL == (file = fopen(fname, "r+"))) {
        fprintf(stderr, "Cannot open output file.\n");
        return;
    }
    int colum_count, row_count, values_count;
    fscanf(file, "%d%d%d\n", &row_count, &colum_count, &values_count);
    *r_count = row_count;
    *v_count = values_count;
    int* row_ptr = (int*)malloc((row_count + 1) * sizeof(int));
    int* col_ind = (int*)malloc(values_count * sizeof(int));
    for (int i = 0; i < values_count; i++) {
        col_ind[i] = -1;
    }
    float* values = (float*)malloc(values_count * sizeof(float));
    int row, column;
    float value;
    while (1) {
        int ret = fscanf(file, "%d %d %f\n", &row, &column, &value);
        if (3 == ret) {
            row_ptr[row]++;
        } else if (ret == EOF) {
            break;
        } else {
            fprintf(stderr, "No match.\n");
        }
    }
    rewind(file);
    int index = 0;
    int val = 0;
    for (int i = 0; i < row_count; i++) {
        val = row_ptr[i];
        row_ptr[i] = index;
        index += val;
    }
    row_ptr[row_count] = values_count;

    fscanf(file, "%d%d%d\n", &row_count, &colum_count, &values_count);
    int i = 0;
    while (1) {
        int ret = fscanf(file, "%d %d %f\n", &row, &column, &value);
        if (3 == ret) {
            while (col_ind[i + row_ptr[row]] != -1) {
                i++;
            }
            col_ind[i + row_ptr[row]] = column;
            values[i + row_ptr[row]] = value;
            i = 0;
        } else if (EOF == ret) {
            break;
        } else {
            fprintf(stderr, "No match\n");
        }
    }
    fclose(file);
    *r_ptr = row_ptr;
    *c_ind = col_ind;
    *v = values;
}