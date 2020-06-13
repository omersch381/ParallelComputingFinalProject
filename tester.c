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
void generateSignsForCurrentOffsetAndCurrentHyphenIndex(char *mainSequence, char *checkedSequence, int offset, int hyphenIndex, char *currentSigns);
char checkAndSetProximity(char mainChar, char checkedChar);
int main()
{
    char *mainSeq = "FFFFFFFF";
    char *checkSeq = "WWWWWFW";
    int offset = 0;
    int hyphenIndex = 2;
    char *currectSigns = (char *)malloc(strlen(checkSeq) * sizeof(char));
    generateSignsForCurrentOffsetAndCurrentHyphenIndex(mainSeq, checkSeq, offset, hyphenIndex, currectSigns);
    int compare = strcmp(currectSigns, ":: ::*:");
    if (compare == 0)
        printf("The Test passed\n");
    else
        printf("The Test did NOT passed\n");


    mainSeq = "FFFFFFFF";
    checkSeq = "WZVWWFW";
    offset = 0;
    hyphenIndex = 4;
    generateSignsForCurrentOffsetAndCurrentHyphenIndex(mainSeq, checkSeq, offset, hyphenIndex, currectSigns);
    int compare2 = strcmp(currectSigns, ": .: *:");
    if (compare2 == 0)
        printf("The Test passed\n");
    else
        printf("The Test did NOT passed\n");

    return 0;
}

void generateSignsForCurrentOffsetAndCurrentHyphenIndex(char *mainSequence, char *checkedSequence, int offset, int hyphenIndex, char *currentSigns)
{
    int j = 0;
    for (int mainSequenceIndex = offset; mainSequenceIndex < offset + strlen(checkedSequence); mainSequenceIndex++)
    {
        if (j != hyphenIndex)
            currentSigns[j] = checkAndSetProximity(mainSequence[mainSequenceIndex], checkedSequence[j]);
        else
            currentSigns[j] = checkAndSetProximity(mainSequence[mainSequenceIndex], '-');
        j++;
    }
}

char checkAndSetProximity(char mainChar, char checkedChar)
{
    if (mainChar == checkedChar)
        return '*';
    else if (areConservative(mainChar, checkedChar))
        return ':';
    else if (areSemiConservative(mainChar, checkedChar))
        return '.';
    else
        return ' ';
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