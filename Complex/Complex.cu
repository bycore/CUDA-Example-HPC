#include "Complex.cuh"
#include <math.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif


CUDA_CALLABLE_MEMBER Complex W(int n) {
    Complex res(cos(2.0 * M_PI / n), sin(2.0 * M_PI / n));
    return res;
}

CUDA_CALLABLE_MEMBER Complex W(int n,int k) {
    Complex res(cos(2.0 * M_PI / n), sin(2.0 * M_PI / n));
    return res;
}