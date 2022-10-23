//#include "io.h"
#include "error.cuh"
//#include "utilities.h"
#include <math.h>

#define BLOCK_SIZE 1024

int main(){
    cudaEvent_t start,stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    int *nrow,*ncolumns,*nnnz;
    int *col_idx, *row_off;//free
    float* values;//free
    conv(col_idx, row_off, values,nrow,ncolumns,nnnz);
    printf("%d %d %d\n",*nrow,*ncolumns,*nnnz);

    free(col_idx);
    free(row_off);
    free(values);
}