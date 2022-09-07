#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "error.cuh"

void bitonic_sort(float* values, int size);
__global__ void bitonic_sort_step(float* dev_values, int j, int k);

__device__ void swap_float(float* f1, float* f2) {
    float tmp = *f1;
    *f1 = *f2;
    *f2 = tmp;
}

__global__ void _bitonic_sort(float* d_arr, unsigned stride, unsigned inner_stride) {
    unsigned flipper = inner_stride >> 1;
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned tid_other = tid ^ flipper;

    if (tid < tid_other) {
        // 操纵左侧的半部分
        if ((tid & stride) == 0) {
            // 此处将留升序
            if (d_arr[tid] > d_arr[tid_other]) {
                swap_float(&d_arr[tid], &d_arr[tid_other]);
            }
        } else {
            // 此处将留降序
            if (d_arr[tid] < d_arr[tid_other]) {
                swap_float(&d_arr[tid], &d_arr[tid_other]);
            }
        }
    }
}

/// entry point for gpu float sorting
/// \param arr memory on gpu device
/// \param len the length of the array
void float_sort(float arr[], int len) {
    // 首先检查长度是否为 2 的幂
    unsigned twoUpper = 1;
    for (; twoUpper < len; twoUpper <<= 1) {
        if (twoUpper == len) {
            break;
        }
    }
    // 如果是 host 指针，返回
    cudaPointerAttributes attrs;
    cudaPointerGetAttributes(&attrs, arr);
    if (attrs.type != cudaMemoryTypeDevice) {
        return;
    }

    float* d_input_arr;
    unsigned input_arr_len;
    if (twoUpper == len) {
        input_arr_len = len;
        d_input_arr = arr;
    } else {
        // 需要 padding
        input_arr_len = twoUpper;
        cudaMalloc(&d_input_arr, sizeof(float) * input_arr_len);
        // 然后初始化
        cudaMemcpy(d_input_arr, arr, sizeof(float) * len, cudaMemcpyHostToDevice);
        cudaMemset(d_input_arr + len, 0x7f, sizeof(float) * (input_arr_len - len));
    }

    dim3 grid_dim((input_arr_len / 256 == 0) ? 1 : input_arr_len / 256);
    dim3 block_dim((input_arr_len / 256 == 0) ? input_arr_len : 256);

    // 排序过程(重点)
    for (unsigned stride = 2; stride <= input_arr_len; stride <<= 1) {
        for (unsigned inner_stride = stride; inner_stride >= 2; inner_stride >>= 1) {
            _bitonic_sort<<<grid_dim, block_dim>>>(d_input_arr, stride, inner_stride);
        }
    }

    // 如果 padding 过，则此处还原
    if (twoUpper != len) {
        cudaMemcpy(arr, d_input_arr, sizeof(float) * len, cudaMemcpyDeviceToDevice);
        cudaFree(d_input_arr);
    }
}

int main(void) {
    int size = 100;
    float* values = (float*)malloc(size * sizeof(float));
    float* dNums;
    srand(time(NULL));
    for (int i = 0; i < size; ++i) {
        values[i] = size - i;
    }
    CHECK(cudaMalloc((void**)&dNums, sizeof(float) * size));
    CHECK(cudaMemcpy(dNums, values, sizeof(float) * size, cudaMemcpyHostToHost));
    float_sort(dNums, size);
    CHECK(cudaMemcpy(values,dNums,sizeof(float) * size, cudaMemcpyDeviceToHost));
    FILE* fp = NULL;
    fp = fopen("gz.txt", "w+");
    for (int i = 0; i < size; i++) {
        fprintf(fp, "%d\t%f \n", i, values[i]);
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

    dim3 blocks(32, 1);                                /* Number of blocks   */
    dim3 threads((size + blocks.x - 1) / blocks.x, 1); /* Number of threads  */

    for (int k = 2; k <= size; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
        }
    }
    cudaMemcpy(values, dev_values, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_values);
}
