#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define INPUT_FILE "input.txt"
#define MODE_READ "r"
#define SEQUENCE_1_LIMIT 3000
#define SEQUENCE_2_LIMIT 2000
#define GROUP_STRING_SIZE_LIMIT 6
#define OUTPUT_FILE "output.txt"
#define MODE_WRITE "w+"

int areConservative(char mainChar, char checkedChar);
int areSemiConservative(char mainChar, char checkedChar);
int areTheCharsInGroup(char mainChar, char checkedChar, const char groupToCheck[][GROUP_STRING_SIZE_LIMIT], int arraySize);
int main()
{
    // Test semi-conservative only
    char checker1 = 'C';
    char checker2 = 'A';
    printf("%d\n", areConservative(checker1, checker2));
    printf("%d\n", areSemiConservative(checker1, checker2));

    printf("\n");

    // Test both
    checker1 = 'Q';
    checker2 = 'K';
    printf("%d\n", areConservative(checker1, checker2));
    printf("%d\n", areSemiConservative(checker1, checker2));

    printf("\n");

    // Test conservative only
    checker1 = 'F';
    checker2 = 'W';
    printf("%d\n", areConservative(checker1, checker2));
    printf("%d\n", areSemiConservative(checker1, checker2));

    printf("\n");

    // Test none
    checker1 = 'Z';
    checker2 = 'A';
    printf("%d\n", areConservative(checker1, checker2));
    printf("%d\n", areSemiConservative(checker1, checker2));

    return 0;
}

int areConservative(char mainChar, char checkedChar)
{
    const char conservativeGroup[9][GROUP_STRING_SIZE_LIMIT] = {"NDEQ", "NEQK", "STA", "MILV", "QHRK", "NHQK", "FYW", "HY", "MILF"};
    return areTheCharsInGroup(mainChar, checkedChar, conservativeGroup, 9);
}

int areSemiConservative(char mainChar, char checkedChar)
{
    const char semiConservativeGroup[11][GROUP_STRING_SIZE_LIMIT] = {"SAG", "ATV", "CSA", "SGND", "STPA", "STNK", "NEQHRK", "NDEQHK", "SNDEQK", "HFY", "FVLIM"};

    return areTheCharsInGroup(mainChar, checkedChar, semiConservativeGroup, 11);
}

int areTheCharsInGroup(char mainChar, char checkedChar, const char groupToCheck[][GROUP_STRING_SIZE_LIMIT], int arraySize)
{
    int isMainCharInTheGroup = 0;
    int isCheckedCharInTheGroup = 0;

    for (int i = 0; i < arraySize; i++)
    {
        for (int j = 0; j < strlen(groupToCheck[i]); j++)
        {
            if (mainChar == groupToCheck[i][j])
                isMainCharInTheGroup = 1;
            if (checkedChar == groupToCheck[i][j])
                isCheckedCharInTheGroup = 1;
        }
        if (isMainCharInTheGroup && isCheckedCharInTheGroup)
            return 1;
        else
        {
            isMainCharInTheGroup = 0;
            isCheckedCharInTheGroup = 0;
        }
    }
    return 0;
}