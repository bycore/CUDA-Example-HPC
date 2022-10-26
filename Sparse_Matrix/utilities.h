#include <stdio.h>
#include <stdlib.h>
float* sparse_gen(int n, int& nnz, int& nnz_row, int& nnz_max) {
    float* arr = new float[n];

    nnz_row = 0;

    for (int j = 0; j < n; j++) {
        float prob = (rand() % 10) / 10.0;

        // If randomly generated no (b/w 0 and 1) is not greater than 0.7 then matrix cell value is 0.
        if (prob >= 0.7) {
            arr[j] = rand() % 100 + 1;
            nnz_row++;
        } else {
            arr[j] = 0;
        }
    }

    if (nnz_row > nnz_max) {
        nnz_max = nnz_row;
    }

    nnz += nnz_row;

    return arr;
}

// Generates the dense vector required for multiplication
float* vect_gen(int n, bool isSerial = false) {
    float* vect = new float[n];

    for (int i = 0; i < n; i++) {
        if (!isSerial)
            vect[i] = rand() % 10;
        else
            vect[i] = i + 1;
    }

    return vect;
}

// Prints a matrix
void display_matrix(float** mat, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", mat[i][j]);
        }
        printf("\n");
    }
}

void display_vector(float* vect, int n) {
    for (int i = 0; i < n; i++) {
        fprintf(stdout,"%f ",vect[i]);
    }
    fprintf(stdout,"\n");
}
