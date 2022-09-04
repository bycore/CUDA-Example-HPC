#include <stdio.h>
#include "Complex.cuh"

int main(){
    Complex x(1,1);
    printf("%lf %lf\n",x.getW(1).real,x.getW(1).imag);
    return 0;
}