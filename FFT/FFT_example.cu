#include <stdio.h>
#include "Complex.cuh"

int GetBits(int n);
__device__ int BinaryReverse(int i, int bits);
__device__ void Bufferfly(Complex* a, Complex* b, Complex factor);
__global__ void FFT(Complex nums[], Complex result[], int n, int bits);
void printSequence(Complex nums[], const int N);

int main() {
    srand(time(0));           // 设置随机数种子
    const int TPB = 1024;     // 每个Block的线程数，即blockDim.x
    const int N = 1024 * 32;  // 数列大小
    const int bits = GetBits(N);

    // 随机生成实数数列
    Complex *nums = (Complex*)malloc(sizeof(Complex) * N), *dNums, *dResult;
    for (int i = 0; i < N; ++i) {
        nums[i].GetRandomReal();
    }
    printf("Length of Sequence: %d\n", N);
    printf("Before FFT: \n");
    printSequence(nums, N);

    // 分配device内存，拷贝数据到device
    cudaMalloc((void**)&dNums, sizeof(Complex) * N);
    cudaMalloc((void**)&dResult, sizeof(Complex) * N);
    cudaMemcpy(dNums, nums, sizeof(Complex) * N, cudaMemcpyHostToDevice);

    // 调用kernel
    dim3 threadPerBlock = dim3(TPB);
    dim3 blockNum = dim3((N + threadPerBlock.x - 1) / threadPerBlock.x);
    FFT<<<blockNum, threadPerBlock>>>(dNums, dResult, N, bits);

    // 拷贝回结果
    cudaMemcpy(nums, dResult, sizeof(Complex) * N, cudaMemcpyDeviceToHost);

    printf("After FFT: \n");
    printSequence(nums, N);

    // 释放内存
    free(nums);
    cudaFree(dNums);
    cudaFree(dResult);
}

int GetBits(int n) {
    int bits = 0;
    while (n >>= 1) {
        bits++;
    }
    return bits;
}

__device__ int BinaryReverse(int i, int bits) {
    int r = 0;
    do {
        r += i % 2 << --bits;
    } while (i /= 2);
    return r;
}

__device__ void Bufferfly(Complex* a, Complex* b, Complex factor) {
    Complex a1 = (*a) + factor * (*b);
    Complex b1 = (*a) - factor * (*b);
    *a = a1;
    *b = b1;
}

__global__ void FFT(Complex nums[], Complex result[], int n, int bits) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n)
        return;
    for (int i = 2; i < 2 * n; i *= 2) {
        if (tid % i == 0) {
            int k = i;
            if (n - tid < k)
                k = n - tid;
            for (int j = 0; j < k / 2; ++j) {
                Bufferfly(&nums[BinaryReverse(tid + j, bits)], &nums[BinaryReverse(tid + j + k / 2, bits)], W(k, j));
            }
        }
        __syncthreads();
    }
    result[tid] = nums[BinaryReverse(tid, bits)];
}

void printSequence(Complex nums[], const int N) {
    printf("[");
    for (int i = 0; i < N; ++i) {
        double real = nums[i].real(), imag = nums[i].imag();
        if (imag == 0)
            printf("%.16f", real);
        else {
            if (imag > 0)
                printf("%.16f+%.16fi", real, imag);
            else
                printf("%.16f%.16fi", real, imag);
        }
        if (i != N - 1)
            printf(", ");
    }
    printf("]\n");
}
