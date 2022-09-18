#ifndef __COMPLEX__
#define __COMPLEX__

#include<stdio.h>
#include<math.h>

class Complex{

public:
    double real;
    double imag;
    Complex(){}
    
    Complex getComplex(double x,double y){
        Complex r;
        r.real=x;r.imag=y;
        return r;
    }

    __device__ Complex(double x,double y){
        this->real=x;this->imag=y;
    }

    __device__ Complex W(int n);
    __device__ Complex W(int n,int k);
};



#endif