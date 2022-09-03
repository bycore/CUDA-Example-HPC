#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <Windows.h>
#include "error.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void OddEvenSwitchSort(int nums[], int size);
int CheckFun(int* nums1, int* nums2, int n);
void SequentialBubbleSort(int nums[], int size);
__global__ void even_swapper(int* X, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i % 2 == 0 && i < N - 1) {
		if (X[i + 1] < X[i]) {
			int temp = X[i];
			X[i] = X[i + 1];
			X[i + 1] = temp;
		}
	}
}

__global__ void odd_swapper(int* X, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i % 2 != 0 && i < N - 2) {
		if (X[i + 1] < X[i]) {
			int temp = X[i];
			X[i] = X[i + 1];
			X[i + 1] = temp;
		}
	}
}
int main() {
	
	int size = 66536;
	int* nums1 = (int*)malloc(sizeof(int) * size);
	int* nums2 = (int*)malloc(sizeof(int) * size);

	srand(time(0));
	for (int i = 0; i < size; i++) {
		int num = rand();
		nums1[i] = num;
		nums2[i] = num;
	}
	printf("Number of numbers: %d\n", size);
	float  s = GetTickCount();
	OddEvenSwitchSort(nums1, size);
	float tp = GetTickCount() - s;
	printf("Parallel Sorting Time: %fms\n", tp);
	s = GetTickCount();
	SequentialBubbleSort(nums2, size);
	float ts = GetTickCount() - s;
	printf("Sequential Sorting Time: %fms\n", ts);
	printf("Acceleration Factor: %f\n", ts / tp);
	do {
		int _ = CheckFun(nums1, nums2, size);
		if (_ == 0) {
			printf("successf!\n");
		}
		else {
			printf("wrong!\n");
		}
	} while (0);
	free(nums1);
	free(nums2);
}

void OddEvenSwitchSort(int nums[], int size) {
	int* dNums;
	CHECK(cudaMalloc((void**)&dNums, sizeof(int) * size));
	CHECK(cudaMemcpy(dNums, nums, sizeof(int) * size, cudaMemcpyHostToHost));

	dim3 threadPerBlock(1024);
	dim3 blockNum((size + threadPerBlock.x - 1) / threadPerBlock.x);
	for (int i = 0; i < size; i++) {
		even_swapper << <blockNum, threadPerBlock >> > (dNums, size);
		odd_swapper << <blockNum, threadPerBlock >> > (dNums, size);
	}
	CHECK(cudaMemcpy(nums, dNums, sizeof(int) * size, cudaMemcpyDeviceToHost));

	CHECK(cudaFree(dNums));
}



void SequentialBubbleSort(int nums[], int size) {
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size - 1; ++j) {
			if (nums[j] > nums[j + 1]) {
				int temp = nums[j];
				nums[j] = nums[j + 1];
				nums[j + 1] = temp;
			}
		}
	}
}

int CheckFun(int nums1[], int nums2[], int n) {
	for (int i = 0; i < n; i++) {
		if (nums1[i] != nums2[i]) {
			printf("%d:%d\t%d\n", i, nums1[i], nums2[i]);
		}
	}
	return 0;
}