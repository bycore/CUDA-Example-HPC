ASSOCIATE_DIR=../Complex/

NVCC = nvcc -arch sm_50

TARGETBIN:= demo
TEST_SOURCE = demo.cu


$(TARGETBIN):$(TEST_SOURCE) Complex.o
	$(NVCC) -I$(ASSOCIATE_DIR) Complex.o $(TEST_SOURCE) -o $(TARGETBIN)

Complex.o:$(ASSOCIATE_DIR)Complex.cu
	$(NVCC) --device-c $(ASSOCIATE_DIR)Complex.cu -o Complex.o

.PHONY:clean
clean:
	-rm -rf $(TARGETBIN)
	-rm -rf *.o
    

