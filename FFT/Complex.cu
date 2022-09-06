#include "Complex.cuh"
#include <math.h>

Complex Complex::W(int n) {
    Complex res(cos(2.0 * M_PI / n), sin(2.0 * M_PI / n));
    return res;
}