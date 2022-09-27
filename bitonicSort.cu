#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "error.cuh"

void SequentialBubbleSort(float* nums, int size);
int CheckFun(float* nums1, float* nums2, int n);
void float_sort(float arr[], int len);
__global__ void _bitonic_sort(float* d_arr, unsigned stride, unsigned inner_stride);

__device__ void swap_float(float* f1, float* f2) {
    float tmp = *f1;
    *f1 = *f2;
    *f2 = tmp;
}




int main(void) {
    int size = 66536;
    float* values = (float*)malloc(size * sizeof(float));
    float* values_cpu = (float*)malloc(size * sizeof(float));
    float* dNums;
    srand(time(NULL));
    for (int i = 0; i < size; ++i) {
        float num=random();
        values[i] = num;
        values_cpu[i] = num;
    }
    CHECK(cudaMalloc((void**)&dNums, sizeof(float) * size));
    CHECK(cudaMemcpy(dNums, values, sizeof(float) * size, cudaMemcpyHostToHost));
    float_sort(dNums, size);
    CHECK(cudaMemcpy(values,dNums,sizeof(float) * size, cudaMemcpyDeviceToHost));
    SequentialBubbleSort(values_cpu,size);
    int _ = CheckFun(values, values_cpu, size);
    if (_ == 0) {
        printf("successf!\n");
    } else {
        printf("wrong!\n");
    }
    free(values);
    free(values_cpu);
    cudaFree(dNums);
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

void SequentialBubbleSort(float* nums, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size - 1; j++) {
            if (nums[j] > nums[j + 1]) {
                int temp = nums[j];
                nums[j] = nums[j + 1];
                nums[j + 1] = temp;
            }
        }
    }
}

int CheckFun(float* nums1, float* nums2, int n) {
    FILE* fp = NULL;
    fp = fopen("gz.txt", "w+");
    fprintf(fp, "%d\n", n);
    for (int i = 0; i < n; i++) {
        fprintf(fp, "%d:%f\t%f\n", i, nums1[i], nums2[i]);
        if (nums1[i] != nums2[i]) {
            //fprintf(fp, "%d:%d\t%d\n", i, nums1[i], nums2[i]);
            return 1;
        }
    }
    return 0;
}

//https://blog.csdn.net/xbinworld/article/details/76408595