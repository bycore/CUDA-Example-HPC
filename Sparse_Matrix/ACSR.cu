#include "io.h"
#include "error.cuh"
//#include "utilities.h"
#include <math.h>

#define BLOCK_SIZE 1024

int main(){
    cudaEvent_t start,stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    int nrow=0,ncolumns=0,nnnz=0;
    int *ptr_nrow,*ptr_ncolumns,*ptr_nnnz;
    ptr_nrow=&nrow;ptr_ncolumns=&ncolumns;ptr_nnnz=&nnnz;
    int *col_idx, *row_off;//free
    float* values;//free
    conv(col_idx, row_off, values,ptr_nrow,ptr_ncolumns,ptr_nnnz);
    printf("%d %d %d\n",nrow,ncolumns,nnnz);

    free(col_idx);
    free(row_off);
    free(values);
}