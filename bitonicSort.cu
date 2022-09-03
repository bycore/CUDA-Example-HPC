#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "error.cuh"

void bitonic_sort(float* values, int size);
__global__ void bitonic_sort_step(float* dev_values, int j, int k);

int main(void) {
    int size = 100;
    float* values = (float*)malloc(size * sizeof(float));
    srand(time(NULL));
    for (int i = 0; i < size; ++i) {
        values[i] = size - i;
    }

    clock_t start = clock();
    bitonic_sort(values, size);
    FILE* fp = NULL;
    fp = fopen("gz.txt", "w+");
    for (int i = 0; i < size; i++) {
        fprintf(fp,"%d\t%f \n",i, values[i]);
    }
}
__global__ void bitonic_sort_step(float* dev_values, int j, int k) {
    unsigned int i, ixj; /* Sorting partners: i and ixj */
    i = threadIdx.x + blockDim.x * blockIdx.x;
    ixj = i ^ j;

    /* The threads with the lowest ids sort the array. */
    if ((ixj) > i) {
        if ((i & k) == 0) {
            /* Sort ascending */
            if (dev_values[i] > dev_values[ixj]) {
                /* exchange(i,ixj); */
                float temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
        if ((i & k) != 0) {
            /* Sort descending */
            if (dev_values[i] < dev_values[ixj]) {
                /* exchange(i,ixj); */
                float temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
    }
}

void bitonic_sort(float* values, int size) {
    float* dev_values;
    // Allocate space for device copies of values
    cudaMalloc((void**)&dev_values, size * sizeof(float));
    cudaMemcpy(dev_values, values, size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blocks(32, 1);                              /* Number of blocks   */
    dim3 threads((size + blocks.x - 1) / blocks.x, 1); /* Number of threads  */

    for (int k = 2; k <= size; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
        }
    }
    cudaMemcpy(values, dev_values, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_values);
}
