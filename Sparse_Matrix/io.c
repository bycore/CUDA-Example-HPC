#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void conv(int* col_idx, int* row_off, float* values, int* row_length, int* column_length, int* nnz);
void sort(int* col_idx, float* a, int start, int end);
void coo2csr(int row_length, int nnz, float* values, int* row, int* col, float* csr_values, int* col_idx, int* row_start);

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
static void conv(int* col_idx, int* row_off, float* values, int* row_length, int* column_length, int* nnz) {
    const char* filename = "demo.txt";
    FILE* file = fopen(filename, "r");
    while (cpeek(file) == '%' && cpeek(file) != EOF) {
        FEIL_ignore(file, 2048, '\n');
    }
    fscanf(file, "%d %d %d", row_length, column_length, nnz);
    int *row, *column;
    float* coovalues;
    row = (int*)malloc(sizeof(int) * (*nnz));
    column = (int*)malloc(sizeof(int) * (*nnz));
    coovalues = (float*)malloc(sizeof(int) * (*nnz));
    row_off = (int*) calloc((*row_length+1),sizeof(int));
    col_idx = (int*)calloc((*nnz),sizeof(int));
    values=(float*)calloc((*nnz),sizeof(float));
    for (int i = 0; i < *nnz; i++) {
        int m, n;
        float data;
        fscanf(file, "%d %d %f", &m, &n, &data);
        row[i] = m;
        column[i] = n;
        coovalues[i] = data;
    }
    coo2csr(*row_length, *nnz, coovalues, row, column, values, col_idx, row_off);
    // int nnz_max, nnz_avg, nnz_dev;  //稀疏矩阵的特征
    // nnz_max = 0;
    // int tot_nnz = 0, tot_nnz_square = 0;
    // for (int i = 0; i < -1 + *row_length; i++) {
    //     int cur_nnz = row_off[i + 1] - row_off[i];
    //     tot_nnz += cur_nnz;
    //     tot_nnz_square += cur_nnz * cur_nnz;
    //     if (cur_nnz > nnz_max)
    //         nnz_max = cur_nnz;
    // }
    // tot_nnz += nnz - row_off[row_length - 1];
    // tot_nnz_square += (nnz - row_off[row_length - 1]);

    // if ((nnz - row_off[row_length - 1]) > nnz_max)
    //     nnz_max = nnz - row_off[row_length - 1];

    // nnz_avg = tot_nnz / row_length;
    // nnz_dev = (int)sqrt(tot_nnz_square / row_length - (nnz_avg * nnz_avg));

    // row_off[row_length] = nnz;
    free(row);
    free(column);
    free(coovalues);
    return;
}
int main() {
    int m=0,n=0,cnt=0;
    int *col_idx, *row_off, *nnz_max=&m, *nnz_avg=&n, *nnz_dev=&cnt;
    float* values;
    conv(col_idx, row_off, values, nnz_max, nnz_avg, nnz_dev);
    printf("%d %d %d\n",m, n, cnt);
}

void coo2csr(int row_length, int nnz, float* values, int* row, int* col, float* csr_values, int* col_idx, int* row_start) {
    int i, l;

    for (i = 0; i <= row_length; i++)
        row_start[i] = 0;

    /* determine row lengths */
    for (i = 0; i < nnz; i++)
        row_start[row[i] + 1]++;

    for (i = 0; i < row_length; i++)
        row_start[i + 1] += row_start[i];

    /* go through the structure  once more. Fill in output matrix. */
    for (l = 0; l < nnz; l++) {
        i = row_start[row[l]];
        csr_values[i] = values[l];
        col_idx[i] = col[l];
        row_start[row[l]]++;
    }

    /* shift back row_start */
    for (i = row_length; i > 0; i--)
        row_start[i] = row_start[i - 1];

    row_start[0] = 0;

    for (i = 0; i < row_length; i++) {
        sort(col_idx, csr_values, row_start[i], row_start[i + 1]);
    }
}

void sort(int* col_idx, float* a, int start, int end) {
    int i, j, it;
    int dt;

    for (i = end - 1; i > start; i--)
        for (j = start; j < i; j++)
            if (col_idx[j] > col_idx[j + 1]) {
                if (a) {
                    dt = a[j];
                    a[j] = a[j + 1];
                    a[j + 1] = dt;
                }
                it = col_idx[j];
                col_idx[j] = col_idx[j + 1];
                col_idx[j + 1] = it;
            }
}