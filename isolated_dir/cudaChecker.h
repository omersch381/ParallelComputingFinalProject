#ifndef _CUDACHECKER_H_
#define _CUDACHECKER_H_

#define GROUP_STRING_SIZE_LIMIT 7

int areTheCharsInGroupGPU(char mainChar, char checkedChar,
                          char groupToCheck[][GROUP_STRING_SIZE_LIMIT],
                          int arraySize);
#endif