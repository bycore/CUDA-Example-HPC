#include "error.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>

void mergesort(int* data, int size, dim3 threadsPerBlock, dim3 blocksPerGrid);
__global__ void gpu_mergesort(int* source, int* dest, int size, int width, int slices, dim3* threads, dim3* blocks);
__device__ void gpu_bottomUpMerge(int* source, int* dest, int start, int middle, int end);
__device__ unsigned int getIdx(dim3* threads, dim3* blocks);
void merge(int a[], int start, int mid, int end);
void merge_sort_up2down(int a[], int start, int end);

int CheckFun(int nums1[], int nums2[], int n);

int main() {
	int size = 66532;
	int* nums1 = (int*)malloc(sizeof(int) * size);
	int* nums2 = (int*)malloc(sizeof(int) * size);

	srand(time(0));
	for (int i = 0; i < size; i++) {
		int num = rand();
		nums1[i] = num;
		nums2[i] = num;
	}

	dim3 threadPerBlock(32,1,1);
	dim3 blockPerGrid((size+threadPerBlock.x - 1)/threadPerBlock.x,1,1);
	mergesort(nums1, size, threadPerBlock, blockPerGrid);
	merge_sort_up2down(nums2, 0,size-1);
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

void mergesort(int* data, int size, dim3 threadPerBlock, dim3 blockPerGrid) {
	int* D_data;
	int* D_swp;
	dim3* D_threads;
	dim3* D_blocks;
	CHECK(cudaMalloc((void**)&D_data, size * sizeof(int)));
	CHECK(cudaMalloc((void**)&D_swp, size * sizeof(int)));

	CHECK(cudaMemcpy(D_data, data, size * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMalloc((void**)&D_threads, sizeof(dim3)));
	CHECK(cudaMalloc((void**)&D_blocks, sizeof(dim3)));

	CHECK(cudaMemcpy(D_threads, &threadPerBlock, sizeof(dim3), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(D_blocks, &blockPerGrid, sizeof(dim3), cudaMemcpyHostToDevice));
	int* A = D_data;
	int* B = D_swp;
	int nThreads = threadPerBlock.x * threadPerBlock.y * threadPerBlock.z *
		blockPerGrid.x * blockPerGrid.y * blockPerGrid.z;
	for (int width = 1; width < (size << 1); width <<= 1) {
		int slices = size / ((nThreads)*width) + 1;
		gpu_mergesort << <blockPerGrid, threadPerBlock >> > (A, B, size, width, slices, D_threads, D_blocks);
		A = A == D_data ? D_swp : D_data;
		B = B == D_data ? D_swp : D_data;
	}
	CHECK(cudaMemcpy(data, D_data, size * sizeof(int), cudaMemcpyDeviceToHost));
	cudaFree(A);
	cudaFree(B);
}
__global__ void gpu_mergesort(int* source, int* dest, int size, int width, int slices, dim3* threads, dim3* blocks) {
	unsigned int idx = getIdx(threads, blocks);
	int start = width * idx * slices,
		middle,
		end;

	for (int slice = 0; slice < slices; slice++) {
		if (start >= size)
			break;

		middle = min(start + (width >> 1), size);
		end = min(start + width, size);
		gpu_bottomUpMerge(source, dest, start, middle, end);
		start += width;
	}
}
__device__ void gpu_bottomUpMerge(int* source, int* dest, int start, int middle, int end) {
	int i = start;
	int j = middle;
	for (int k = start; k < end; k++) {
		if (i < middle && (j >= end || source[i] < source[j])) {
			dest[k] = source[i];
			i++;
		} else {
			dest[k] = source[j];
			j++;
		}
	}
}
__device__ unsigned int getIdx(dim3* threads, dim3* blocks) {
	int x;
	return threadIdx.x +
		threadIdx.y * (x = threads->x) +
		threadIdx.z * (x *= threads->y) +
		blockIdx.x * (x *= threads->z) +
		blockIdx.y * (x *= blocks->z) +
		blockIdx.z * (x *= blocks->y);
}
void merge(int a[], int start, int mid, int end) {
	int* tmp = (int*)malloc((end - start + 1) * sizeof(int));
	int i = start; 
	int j = mid + 1; 
	int k = 0; 

	while (i <= mid && j <= end) {
		if (a[i] <= a[j])
			tmp[k++] = a[i++];
		else
			tmp[k++] = a[j++];
	}

	while (i <= mid)
		tmp[k++] = a[i++];

	while (j <= end)
		tmp[k++] = a[j++];

	for (i = 0; i < k; i++)
		a[start + i] = tmp[i];

	free(tmp);
}


void merge_sort_up2down(int a[], int start, int end) {
	if (a == NULL || start >= end)
		return;

	int mid = (end + start) / 2;
	merge_sort_up2down(a, start, mid);
	merge_sort_up2down(a, mid + 1, end);

	merge(a, start, mid, end);
}

int CheckFun(int nums1[], int nums2[], int n){
	for (int i = 0; i < n; i++) {
		if (nums1[i] != nums2[i]) {
			printf("%d:%d\t%d\n", i, nums1[i], nums2[i]);
			return 1;
		}
	}
	return 0;
}