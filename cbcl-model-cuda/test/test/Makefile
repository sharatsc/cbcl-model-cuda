CUDA  = /Developer/CUDA
CFLAGS=  --host-compilation c++ -I $(CUDA)/common/inc \
	    -L $(CUDA)/lib
SOURCES= main.cpp cbcl_model.cu 
LIBS   = -lcutil
cbcl_model: 
	nvcc $(CFLAGS) $(SOURCES) $(LIBS)
clean:
	rm *.o
