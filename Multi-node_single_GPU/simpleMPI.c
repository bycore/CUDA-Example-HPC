#include "simpleMPI.h"
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    int blockSize = 5;
    int gridSize = 1;
    int dataSizePerNode = gridSize * blockSize;

    MPI_Init(&argc, &argv);

    int commSize, commRank;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);

    int dataSizeTotal = dataSizePerNode * commSize;
    float* dataRoot = NULL;

    if (commRank == 0) {
        printf("Running on %d nodes\n", commSize);
        dataRoot = (float*)malloc(dataSizeTotal * sizeof(float));
        initData(dataRoot, dataSizeTotal);
        printTotalData("Initial:", dataRoot, dataSizeTotal);
    }

    float* dataNode = (float*)malloc(dataSizePerNode * sizeof(float));
    MPI_Scatter(dataRoot, dataSizePerNode, MPI_FLOAT, dataNode, dataSizePerNode, MPI_FLOAT, 0, MPI_COMM_WORLD);
    computeGPU(dataNode, blockSize, gridSize);
    printNodeData(commRank, dataNode, dataSizePerNode);
    MPI_Gather(dataNode, dataSizePerNode, MPI_FLOAT, dataRoot, dataSizePerNode, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    if (commRank == 0) {
        printTotalData("Result:", dataRoot, dataSizeTotal);
        free(dataRoot);
    }

    free(dataNode);

    if (commRank == 0) {
        printf("PASSED\n");
    }

    MPI_Finalize();

    return 0;
}
