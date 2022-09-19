#include <stdio.h>
#include "Complex.cuh"

int main(){
    Complex a(1,1);
    Complex b(2,2);
    Complex c=a+b;
    printf("%lf %lf\n",c.real(),c.imag());
    return 0;
}