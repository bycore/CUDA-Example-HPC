#include "error.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>


using namespace std;
void bitonic_sort(float* values, int size);
__global__  void bitonic_sort_step(float* dev_values, int j, int k);

int main(void){
	int size=10000;
	float* values = (float*)malloc(size * sizeof(float));
	srand(time(NULL));
	for (int i = 0; i < size; ++i){
		values[i] = (float)rand() / (float)RAND_MAX;
	}

	clock_t start = clock();
	dim3 threadPerBlock(1024);
	dim3 blockNum();
	bitonic_sort(values, size);

	cout << "Time: " << ((double)(clock() - start)) / CLOCKS_PER_SEC << " seconds." << endl;
	for (int i = 0; i < size; i ++){
		printf("%f \n", values[i]);
	}
}
__global__  void bitonic_sort_step(float* dev_values, int j, int k){
	unsigned int i, ixj; /* Sorting partners: i and ixj */
	i = threadIdx.x + blockDim.x * blockIdx.x;
	ixj = i ^ j;

	/* The threads with the lowest ids sort the array. */
	if ((ixj) > i){
		if ((i & k) == 0){
			/* Sort ascending */
			if (dev_values[i] > dev_values[ixj]){
				/* exchange(i,ixj); */
				float temp = dev_values[i];
				dev_values[i] = dev_values[ixj];
				dev_values[ixj] = temp;
			}
		}
		if ((i & k) != 0){
			/* Sort descending */
			if (dev_values[i] < dev_values[ixj]){
				/* exchange(i,ixj); */
				float temp = dev_values[i];
				dev_values[i] = dev_values[ixj];
				dev_values[ixj] = temp;
			}
		}
	}
}

void bitonic_sort(float* values, int size){
	float* dev_values;
	// Allocate space for device copies of values
	cudaMalloc((void**)&dev_values, size * sizeof(float));
	cudaMemcpy(dev_values, values, size * sizeof(float), cudaMemcpyHostToDevice);

	dim3 blocks(1024, 1);    /* Number of blocks   */
	dim3 threads((size + blocks.x - 1) / blocks.x, 1);  /* Number of threads  */

	for (int k = 2; k <= size; k <<= 1){
		for (int j = k >> 1; j > 0; j >>= 1){
			bitonic_sort_step << <blocks, threads >> > (dev_values, j, k);
		}
	}
	cudaMemcpy(values, dev_values, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(dev_values);
}
