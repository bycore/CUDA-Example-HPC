#ifndef __COMPLEX__
#define __COMPLEX__

#include<stdio.h>
#include<math.h>

class Complex{

public:
    double real;
    double imag;
    Complex(){}

    __host__ __device__ Complex(double x,double y){
        this->real=x;this->imag=y;
    }

    __device__ Complex W(int n);
    __device__ Complex W(int n,int k);
};



#endif