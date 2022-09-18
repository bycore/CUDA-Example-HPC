#include <stdio.h>
#include "Complex.cuh"

int main(){
    Complex a;
    a=a.getComplex(1,1);
    printf("%lf %lf\n",a.real,a.imag);
    return 0;
}