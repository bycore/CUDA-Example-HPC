#include "Complex.cuh"

__device__ Complex:: Complex getw(int n,int k){
    Complex res = Complex(cos(2.0 * M_PI /n),sin(2.0 * M_PI /n));
}