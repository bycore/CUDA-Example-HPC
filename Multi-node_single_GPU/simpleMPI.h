#ifndef _SIMPLEMPI_H
#define _SIMPLEMPI_H

void initData(float *data, int dataSize);
void printTotalData(const char *name,float *data, int dataSize);
void printNodeData(int commRank,float *data, int dataSize);
void computeGPU(float *hostData, int blockSize, int gridSize);

#endif

