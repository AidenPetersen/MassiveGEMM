NVCC=nvcc
MPICC=mpicxx

CUDASRC=gemm.cu
CUDAOBJ=$(CUDASRC:.cu=.o)
CUDAFLAGS=

SRC=main.cpp matrix.cpp
OBJ=$(SRC:.cpp=.o)
CFLAGS=-Wall -Wextra 

LDFLAGS=-lcudart

EXEC=mmult

all: $(EXEC)

$(EXEC): $(CUDAOBJ) $(OBJ)
	$(MPICC) $(CFLAGS) $(LDFLAGS) -o$@ $^

%.o: %.cu
	$(NVCC) $(CUDAFLAGS) -c $^ -o $@

%.o: %.cpp
	$(MPICC) $(CFLAGS) -c $^ -o$@

clean:
	rm -f *.o $(EXEC)



