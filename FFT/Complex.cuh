#ifndef __COMPLEX__
#define __COMPLEX__

#include<stdio.h>
#include<math.h>

class Complex{

public:
    double real;
    double imag;
    
    Complex(double x,double y){
        real=x;imag=y;
    }
};

Complex W(int n);

#endif