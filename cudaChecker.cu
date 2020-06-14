#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <string.h>
#define GROUP_STRING_SIZE_LIMIT 6

__global__ void generateSignsForCurrentOffsetAndCurrentHyphenIndex(char *mainSequence,
                                                        char *checkedSequence,
                                                        int offset,
                                                        int hyphenIndex,
                                                        char *currentSigns);
char checkAndSetProximity(char mainChar, char checkedChar);
int areConservative(char mainChar, char checkedChar);
int areSemiConservative(char mainChar, char checkedChar);
int areTheCharsInGroup(char mainChar, char checkedChar,
                       const char groupToCheck[][GROUP_STRING_SIZE_LIMIT],
                       int arraySize);
int generateSignsOnGPU(char *mainSequence, char *checkedSequence, int offset,
                       int hyphenIndex, char *currentSigns);

__global__ void generateSignsForCurrentOffsetAndCurrentHyphenIndex(
    char *mainSequence, char *checkedSequence, int offset, int hyphenIndex,
    char *currentSigns) {
  int j = 0;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= offset && i < offset + strlen(checkedSequence)) {

    if (i == offset + hyphenIndex)
      currentSigns[i - offset] = checkAndSetProximity(mainSequence[i], '-');
    else
      currentSigns[i - offset] =
          checkAndSetProximity(mainSequence[i], checkedSequence[i - offset]);
  }
  // if (j != hyphenIndex)
  //     currentSigns[j] =
  //     checkAndSetProximity(mainSequence[mainSequenceIndex],
  //     checkedSequence[j]);
  // else
  //     currentSigns[j] =
  //     checkAndSetProximity(mainSequence[mainSequenceIndex], '-');
  // j++;
}

char checkAndSetProximity(char mainChar, char checkedChar) {
  if (mainChar == checkedChar)
    return '*';
  else if (areConservative(mainChar, checkedChar))
    return ':';
  else if (areSemiConservative(mainChar, checkedChar))
    return '.';
  else
    return ' ';
}

int areConservative(char mainChar, char checkedChar) {
  const char conservativeGroup[9][GROUP_STRING_SIZE_LIMIT] = {
      "NDEQ", "NEQK", "STA", "MILV", "QHRK", "NHQK", "FYW", "HY", "MILF"};
  return areTheCharsInGroup(mainChar, checkedChar, conservativeGroup, 9);
}

int areSemiConservative(char mainChar, char checkedChar) {
  const char semiConservativeGroup[11][GROUP_STRING_SIZE_LIMIT] = {
      "SAG",    "ATV",    "CSA",    "SGND", "STPA", "STNK",
      "NEQHRK", "NDEQHK", "SNDEQK", "HFY",  "FVLIM"};
  return areTheCharsInGroup(mainChar, checkedChar, semiConservativeGroup, 11);
}

int areTheCharsInGroup(char mainChar, char checkedChar,
                       const char groupToCheck[][GROUP_STRING_SIZE_LIMIT],
                       int arraySize) {
  int isMainCharInTheGroup = 0;
  int isCheckedCharInTheGroup = 0;

  for (int i = 0; i < arraySize; i++) {
    for (int j = 0; j < strlen(groupToCheck[i]); j++) {
      if (mainChar == groupToCheck[i][j])
        isMainCharInTheGroup = 1;
      if (checkedChar == groupToCheck[i][j])
        isCheckedCharInTheGroup = 1;
    }
    if (isMainCharInTheGroup && isCheckedCharInTheGroup)
      return 1;
    else {
      isMainCharInTheGroup = 0;
      isCheckedCharInTheGroup = 0;
    }
  }
  return 0;
}

int generateSignsOnGPU(char *mainSequence, char *checkedSequence, int offset,
                       int hyphenIndex, char *currentSigns) {
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

  size_t mainSequenceSize = strlen(mainSequence) * sizeof(char);

  // Allocate memory on GPU to copy the mainSequence from the host
  int *mainSequenceDevicePointer;

  err = cudaMalloc((void **)&mainSequenceDevicePointer, mainSequenceSize);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device memory - %s\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy mainSequence from host to the GPU memory
  err = cudaMemcpy(mainSequenceDevicePointer, mainSequence, mainSequenceSize,
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy data from host to device - %s\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate memory on GPU to copy the checkedSequence from the host
  int *checkedSequenceDevicePointer;
  size_t checkedSequenceSize = strlen(checkedSequence) * sizeof(char);
  err = cudaMalloc((void **)&checkedSequenceDevicePointer, checkedSequenceSize);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device memory - %s\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy checkedSequence from host to the GPU memory
  err = cudaMemcpy(checkedSequenceDevicePointer, checkedSequence,
                   checkedSequenceSize, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy data from host to device - %s\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate memory on GPU to copy the currentSigns from the host
  int *currentSignsDevicePointer;
  size_t currentSignsSize = strlen(currentSigns) * sizeof(char);
  err = cudaMalloc((void **)&currentSignsDevicePointer, currentSignsSize);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device memory - %s\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Launch the Kernel
  int threadsPerBlock = 64;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  generateSignsForCurrentOffsetAndCurrentHyphenIndex<<<blocksPerGrid,
                                                       threadsPerBlock>>>(
      mainSequenceDevicePointer, checkedSequenceDevicePointer, offset,
      hyphenIndex, currentSignsDevicePointer);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel -  %s\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy the  result from GPU to the host memory.
  err = cudaMemcpy(currentSigns, currentSignsDevicePointer, currentSignsSize,
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy result array from device to host -%s\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Free allocated memory on GPU - mainSequenceDevicePointer
  if (cudaFree(mainSequenceDevicePointer) != cudaSuccess) {
    fprintf(stderr, "Failed to free device data - %s\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  // Free allocated memory on GPU - checkedSequenceDevicePointer
  if (cudaFree(checkedSequenceDevicePointer) != cudaSuccess) {
    fprintf(stderr, "Failed to free device data - %s\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  // Free allocated memory on GPU - currentSignsDevicePointer
  if (cudaFree(currentSignsDevicePointer) != cudaSuccess) {
    fprintf(stderr, "Failed to free device data - %s\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  return 0;
}
