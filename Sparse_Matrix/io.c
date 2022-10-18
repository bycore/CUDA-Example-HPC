#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
int cpeek(FILE* file) {
    char ans = getc(file);
    ans = ungetc(ans, file);
    return ans;
}

void FEIL_ignore(FILE* file, size_t n, int delim) {
    while (n--) {
        const int c = getc(file);
        if (c == EOF)
            break;
        if (delim != EOF && delim == c) {
            break;
        }
    }
}
static void conv(int* nnz, int* row_length, int* column_length, int* nnz_max, int* nnz_avg, int* nnz_dev, bool isData) {
    const char* filename = "demo.txt";
    FILE* file = fopen(filename, "r");
    while (cpeek(file) == '%' && cpeek(file)!=EOF) {
        FEIL_ignore(file, 2048, '\n');
    }
    fscanf(file,"%d %d %d",row_length,column_length,nnz);
    
    return;
}
int main() {
    int* nnz; int* row_length; int* column_length; int* nnz_max; int* nnz_avg; int* nnz_dev; bool isData=false;
    nnz=malloc(sizeof(int));
    row_length=malloc(sizeof(int));
    column_length=malloc(sizeof(int));
    conv(nnz, row_length, column_length, nnz_max, nnz_avg, nnz_dev, isData);
}