CC    =   mpicc
FLAGS = -fopenmp

all: program

program: parallel_project.o cudaChecker.o
	$(CC) $(FLAGS) -o program parallel_project.o  cudaChecker.o /usr/local/cuda-9.1/lib64/libcudart_static.a -ldl -lrt -lstdc++

parallel_project.o: main.c
	$(CC) $(FLAGS) -c main.c -o parallel_project.o

cudaChecker.o: cudaChecker.cu cudaChecker.h
	nvcc  -I /usr/local/cuda-9.1/samples/common/inc  -c cudaChecker.cu -o cudaChecker.o 

clean:
	rm -f parallel_project.o cudaChecker.o
