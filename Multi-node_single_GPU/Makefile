MPICC=mpicxx
NVCC=nvcc

CUDA_LIBS= -L /usr/local/cuda-11.7/lib64 
CUDA_ARC=-arch sm_50
CUDA_OPT=-lcudart 

CFILES=simpleMPI.c
CUFILES=simpleCu.cu
OBJECTS=simpleMPI.o simpleCu.o

all:
	$(MPICC) -c $(CFILES)  -o simpleMPI.o

	$(NVCC) $(CUDA_ARC) -c $(CUFILES) -o simpleCu.o 

	$(MPICC)  $(CUDA_LIBS)  $(OBJECTS)  -o simpleMPI.exe $(CUDA_OPT)
#	$(NVCC)  $(MPI_LIBS) $(OBJECTS) -o simpleMPI 
run: 
	 mpirun -n  2 ./simpleMPI.exe
clean:
	rm -f *.exe *.o

