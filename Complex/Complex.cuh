#ifndef __COMPLEX__
#define __COMPLEX__

#include <math.h>
#include <stdio.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

class Complex {
   private:
    double re;
    double im;

   public:
    CUDA_CALLABLE_MEMBER Complex() {}
    CUDA_CALLABLE_MEMBER Complex(double x, double y) {
        this->re = x;
        this->im = y;
    }
    CUDA_CALLABLE_MEMBER double real() const { return re; }
    CUDA_CALLABLE_MEMBER double imag() const { return im; }
    CUDA_CALLABLE_MEMBER void real(double __re) { this->re = __re; }
    CUDA_CALLABLE_MEMBER void imag(double __im) { this->im = __im; }

    CUDA_CALLABLE_MEMBER void GetRandomComplex() {
        this->re = (double)rand() / rand();
        this->im = (double)rand() / rand();
    }

    CUDA_CALLABLE_MEMBER Complex& operator=(const Complex& c) {
        this->re = c.real();
        this->im = c.imag();
        return *this;
    }
    CUDA_CALLABLE_MEMBER Complex& operator+=(const Complex& c) {
        this->re += c.re;
        this->im += c.im;
        return *this;
    }
    CUDA_CALLABLE_MEMBER Complex& operator-=(const Complex& c) {
        this->re -= c.re;
        this->im -= c.im;
        return *this;
    }
    CUDA_CALLABLE_MEMBER Complex operator+(const Complex& c) {
        Complex res(this->re + c.re,this->im + c.im);
        return res;
    }
    CUDA_CALLABLE_MEMBER Complex operator-(const Complex& c) {
        Complex res(this->re - c.re,this->im - c.im);
        return res;
    }
    CUDA_CALLABLE_MEMBER Complex operator*(const Complex& c) {
        Complex res(this->re * c.re - this->im * c.im, this->im * c.re + this->re * c.im);
        return res;
    }

    CUDA_CALLABLE_MEMBER Complex& operator+=(const double& __re) {
        this->re += __re;
        return *this;
    }
    CUDA_CALLABLE_MEMBER Complex& operator-=(const double& __re) {
        this->re -= __re;
        return *this;
    }
    CUDA_CALLABLE_MEMBER Complex& operator*=(const double& __re) {
        this->re *= __re;
        this->im *= __re;
        return *this;
    }
    CUDA_CALLABLE_MEMBER Complex& operator/=(const double& __re) {
        this->re /= __re;
        this->im /= __re;
        return *this;
    }
};

CUDA_CALLABLE_MEMBER Complex W(int n);
CUDA_CALLABLE_MEMBER Complex W(int n, int k);

#endif