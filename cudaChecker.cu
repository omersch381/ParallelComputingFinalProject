#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdlib.h>
#include <string.h>
extern "C" {
  #include "cudaChecker.h"
}
#define GROUP_STRING_SIZE_LIMIT 7

// Declarations

__global__ void
areTheCharsInGroup(char mainChar, char checkedChar,
                   char *groupToCheck,
                   int arraySize, int *areThey);

// Implementations

__global__ void
areTheCharsInGroup(char mainChar, char checkedChar,
                   char* groupToCheck,
                   int arraySize, int *areThey) {
  int isMainCharInTheGroup = 0;
  int isCheckedCharInTheGroup = 0;

  for (int i = 0; i < arraySize; i++) {
    for (int j = 0; j < GROUP_STRING_SIZE_LIMIT; j++) {
      if (groupToCheck[i * GROUP_STRING_SIZE_LIMIT + j]) {
        if (mainChar == groupToCheck[i * GROUP_STRING_SIZE_LIMIT + j])
          isMainCharInTheGroup = 1;
        if (checkedChar == groupToCheck[i * GROUP_STRING_SIZE_LIMIT + j])
          isCheckedCharInTheGroup = 1;
      }
    }
  }
  if (isMainCharInTheGroup && isCheckedCharInTheGroup)
    *areThey = 1;
  else {
    *areThey = 0;
    isMainCharInTheGroup = 0;
    isCheckedCharInTheGroup = 0;
  }
}

extern "C" int areTheCharsInGroupGPU(char mainChar, char checkedChar,
                          char groupToCheck[][GROUP_STRING_SIZE_LIMIT],
                          int arraySize) {
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

  size_t groupToCheckSize = arraySize * GROUP_STRING_SIZE_LIMIT * sizeof(char);

  // Allocate memory on GPU to copy the mainSequence from the host
  // char groupToCheckDevicePointer[][GROUP_STRING_SIZE_LIMIT] = {};
  char* groupToCheckDevicePointer = 0;

  err = cudaMalloc((void **)&groupToCheckDevicePointer, groupToCheckSize);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device memory - %s\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  char* tester = (char*) malloc(groupToCheckSize);
  memcpy(tester,groupToCheck,groupToCheckSize);
  // Copy mainSequence from host to the GPU memory
  err = cudaMemcpy(groupToCheckDevicePointer, tester, groupToCheckSize,
    cudaMemcpyHostToDevice);


  // // Copy mainSequence from host to the GPU memory
  // err = cudaMemcpy(groupToCheckDevicePointer, groupToCheck, groupToCheckSize,
  //                  cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy data from host to device - %s\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Are they in the same group
  int *areThey = (int *)malloc(sizeof(int));
  *areThey = 0;

  // Launch the Kernel
  int threadsPerBlock = 64;
  int blocksPerGrid =
      (groupToCheckSize + threadsPerBlock - 1) / threadsPerBlock;
  areTheCharsInGroup<<<blocksPerGrid, threadsPerBlock>>>(
      mainChar, checkedChar, groupToCheckDevicePointer, arraySize, areThey);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel -  %s\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // // Copy the  result from GPU to the host memory.
  // err = cudaMemcpy(groupToCheck, groupToCheckDevicePointer, groupToCheckSize,
  //                  cudaMemcpyDeviceToHost);
  // if (err != cudaSuccess) {
  //   fprintf(stderr, "Failed to copy result array from device to host -%s\n",
  //           cudaGetErrorString(err));
  //   exit(EXIT_FAILURE);
  // }

  // Free allocated memory on GPU - mainSequenceDevicePointer

  // cudaFree(groupToCheckDevicePointer);

  if (cudaFree(groupToCheckDevicePointer) != cudaSuccess) {
    fprintf(stderr, "Failed to free device data - %s\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  free(tester);

  return *areThey;
}
