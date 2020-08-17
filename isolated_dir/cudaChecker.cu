#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdlib.h>
#include <string.h>
extern "C" {
  #include "cudaChecker.h"
}

#define NUMBER_OF_CONSERVATIVE_STRINGS 9
#define NUMBER_OF_SEMI_CONSERVATIVE_STRINGS 11
#define GROUP_STRING_SIZE_LIMIT 7

// Declarations

__global__ void
areTheCharsInGroup(char mainChar, char checkedChar,
                   int arraySize, int *areThey);

// Implementations

__global__ void
areTheCharsInGroup(char mainChar, char checkedChar,
                   int arraySize, int *areThey) {
  char conservativeGroup[NUMBER_OF_CONSERVATIVE_STRINGS][GROUP_STRING_SIZE_LIMIT] = {"NDEQ", "NEQK", "STA", "MILV", "QHRK", "NHQK", "FYW", "HY", "MILF"};
  char semiConservativeGroup[NUMBER_OF_SEMI_CONSERVATIVE_STRINGS][GROUP_STRING_SIZE_LIMIT] = {"SAG", "ATV", "CSA", "SGND", "STPA", "STNK", "NEQHRK", "NDEQHK", "SNDEQK", "HFY", "FVLIM"};

  int isMainCharInTheGroup = 0;
  int isCheckedCharInTheGroup = 0;

  if (arraySize == NUMBER_OF_CONSERVATIVE_STRINGS){
    for (int i = 0; i < arraySize; i++) {
      for (int j = 0; j < GROUP_STRING_SIZE_LIMIT; j++) {
        if (conservativeGroup[i][j]) {
          if (mainChar == conservativeGroup[i][j])
            isMainCharInTheGroup = 1;
          if (checkedChar == conservativeGroup[i][j])
            isCheckedCharInTheGroup = 1;
        }
      }
    }
  }
  else {
      for (int i = 0; i < arraySize; i++) {
        for (int j = 0; j < GROUP_STRING_SIZE_LIMIT; j++) {
          if (semiConservativeGroup[i][j]) {
            if (mainChar == semiConservativeGroup[i][j])
              isMainCharInTheGroup = 1;
            if (checkedChar == semiConservativeGroup[i][j])
              isCheckedCharInTheGroup = 1;
          }
        }
      }
  }
  
  // for (int i = 0; i < arraySize; i++) {
  //   for (int j = 0; j < GROUP_STRING_SIZE_LIMIT; j++) {
  //     if (groupToCheck[i * GROUP_STRING_SIZE_LIMIT + j]) {
  //       if (mainChar == groupToCheck[i * GROUP_STRING_SIZE_LIMIT + j])
  //         isMainCharInTheGroup = 1;
  //       if (checkedChar == groupToCheck[i * GROUP_STRING_SIZE_LIMIT + j])
  //         isCheckedCharInTheGroup = 1;
  //     }
  //   }
  // }
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
  int* areThey;

  err = cudaMalloc((void **)&areThey, sizeof(int));
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device memory - %s\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // char* tester = (char*) malloc(groupToCheckSize);
  // memcpy(tester,groupToCheck,groupToCheckSize);
  // // Copy mainSequence from host to the GPU memory
  // err = cudaMemcpy(groupToCheckDevicePointer, tester, groupToCheckSize,
  //   cudaMemcpyHostToDevice);


  // // Copy mainSequence from host to the GPU memory
  // err = cudaMemcpy(groupToCheckDevicePointer, groupToCheck, groupToCheckSize,
  //                  cudaMemcpyHostToDevice);
  // if (err != cudaSuccess) {
  //   fprintf(stderr, "Failed to copy data from host to device - %s\n",
  //           cudaGetErrorString(err));
  //   exit(EXIT_FAILURE);
  // }

  // Are they in the same group
  // int *areThey = (int *)malloc(sizeof(int));
  // *areThey = 0;

  // Launch the Kernel
  int threadsPerBlock = 64;
  int blocksPerGrid =
      (groupToCheckSize + threadsPerBlock - 1) / threadsPerBlock;
  areTheCharsInGroup<<<blocksPerGrid, threadsPerBlock>>>(
      mainChar, checkedChar, arraySize, areThey);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch areTheCharsInGroupGPU kernel -  %s\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  int result = 0;

  // Copy the  result from GPU to the host memory.
  err = cudaMemcpy(&result, areThey, sizeof(int), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy result array from device to host -%s\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Free allocated memory on GPU - mainSequenceDevicePointer

  // cudaFree(groupToCheckDevicePointer);

  if (cudaFree(areThey) != cudaSuccess) {
    fprintf(stderr, "Failed to free device data - %s\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // free(tester);
  
  return result;
}
