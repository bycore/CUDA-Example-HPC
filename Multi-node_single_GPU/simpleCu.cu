#include <stdio.h>
#include <stdlib.h>
#include "simpleMPI.h"

__global__ void simpleMPIKernel(float* input, float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    output[tid] = pow(input[tid], 2);
}

void initData(float* data, int dataSize) {
    for (int i = 0; i < dataSize; i++) {
        data[i] = (float)(rand() % 10 + 1);
    }
}

void printTotalData(const char* name, float* data, int dataSize) {
    printf("%s\n", name);
    for (int i = 0; i < dataSize; i++) {
        printf("%f\t", data[i]);
    }
    printf("\n");
}

void printNodeData(int commRank, float* data, int dataSize) {
    printf("进程%d:\n", commRank);
    for (int i = 0; i < dataSize; i++) {
        printf("%f\t", data[i]);
    }
    printf("\n");
}

void computeGPU(float* hostData, int blockSize, int gridSize) {
    int dataSize = blockSize * gridSize;
    
    float* deviceInputData = NULL;
    cudaMalloc((void**)&deviceInputData, dataSize * sizeof(float));

    float* deviceOutputData = NULL;
    cudaMalloc((void**)&deviceOutputData, dataSize * sizeof(float));

    cudaMemcpy(deviceInputData, hostData, dataSize * sizeof(float), cudaMemcpyHostToDevice);
    dim3 threadPerBlock(gridSize);
    dim3 blockNum(blockSize);
    simpleMPIKernel<<<threadPerBlock, blockNum>>>(deviceInputData, deviceOutputData);

    cudaMemcpy(hostData, deviceOutputData, dataSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(deviceInputData);
    cudaFree(deviceOutputData);
}
