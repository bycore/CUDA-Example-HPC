ASSOCIATE_DIR=../Complex/

NVCC = nvcc -arch sm_50

TARGETBIN:= FFT_example.out
TEST_SOURCE = FFT_example.cu


$(TARGETBIN):$(TEST_SOURCE) Complex.o 
	$(NVCC) -I$(ASSOCIATE_DIR) Complex.o $(TEST_SOURCE) -o $(TARGETBIN) -rdc=true

Complex.o:$(ASSOCIATE_DIR)Complex.cu
	$(NVCC) --device-c $(ASSOCIATE_DIR)Complex.cu -o Complex.o

.PHONY:clean
clean:
	-rm -rf $(TARGETBIN)
	-rm -rf *.o
    

