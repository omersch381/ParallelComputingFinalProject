#include "mpi.h"
#include <omp.h>
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
#define SLAVE_ID 1
#define MASTER_ID 0
#define TAG 0
#define THRESHOLD 4

void readVarsFromFile(float *w1, float *w2, float *w3, float *w4, char *mainSequence, int *numOfSequences, FILE *reader);
void checkSequence(char *mainSequence, char *checkedSequence, FILE *writer, float w1, float w2, float w3, float w4);
void getclosestOffsetAndHyphen(char *mainSequence, char *checkedSequence, int *n, int *k, float w1, float w2, float w3, float w4);
float getAlignmentForClosestHypenAndCurrentOffset(char *mainSequence, char *checkedSequence, int offset, int *k, float w1, float w2, float w3, float w4);
void generateSignsForCurrentOffsetAndCurrentHyphenIndex(char *mainSequence, char *checkedSequence, int offset, int hyphenIndex, char *currentSigns);
char checkAndSetProximity(char mainChar, char checkedChar);
int areConservative(char mainChar, char checkedChar);
int areSemiConservative(char mainChar, char checkedChar);
int areTheCharsInGroup(char mainChar, char checkedChar, const char groupToCheck[][GROUP_STRING_SIZE_LIMIT], int arraySize);
float getAlignmentSum(char *signs, float w1, float w2, float w3, float w4);

void decideWhichOnesToSendToTheSlave(char mySequences[][SEQUENCE_2_LIMIT], int numOfSequences);
int getLongestSequenceIndex(int array[], int numOfElements);
void getTheSequence();

int main(int argc, char *argv[])
{
    int size, rank;
    FILE *reader, *writer;
    char *mainSequence;
    char *checkedSequence;
    float w1, w2, w3, w4;
    int numOfSequences;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    mainSequence = (char *)malloc(SEQUENCE_1_LIMIT * sizeof(char));
    // checkedSequence = (char *)malloc(SEQUENCE_2_LIMIT * sizeof(char));

    if (rank == MASTER_ID)
    {
        reader = fopen(INPUT_FILE, MODE_READ);
        writer = fopen(OUTPUT_FILE, MODE_WRITE);
        readVarsFromFile(&w1, &w2, &w3, &w4, mainSequence, &numOfSequences, reader);
        char mySequences[numOfSequences][SEQUENCE_2_LIMIT];
        for (int i = 0; i < numOfSequences; i++)
            fscanf(reader, "%s", mySequences[i]);
        decideWhichOnesToSendToTheSlave(mySequences, numOfSequences);
    }
    else
    {
        getTheSequence();
    }

    // for (int i = 0; i < numOfSequences; i++)
    // {
    //     double start = MPI_Wtime();

    //     fscanf(reader, "%s", checkedSequence);
    //     checkSequence(mainSequence, checkedSequence, writer, w1, w2, w3, w4);
    //     double end = MPI_Wtime();

    //     printf("\nThe time it took is: %lf\n", end - start);
    // }
    if (rank == MASTER_ID)
    {
        fclose(writer);
        fclose(reader);
    }

    // free(checkedSequence);
    free(mainSequence);

    MPI_Finalize();
    return 0;
}

void decideWhichOnesToSendToTheSlave(char mySequences[][SEQUENCE_2_LIMIT], int numOfSequences)
{
    int numOfSequencesToSend = numOfSequences / THRESHOLD;
    if (numOfSequencesToSend > 1)
    {
        char arrayOfSequencesToSend[numOfSequencesToSend][SEQUENCE_2_LIMIT];
        int index = 0;
        for (int j = 0; j < numOfSequencesToSend; j++)
        {
            int array[THRESHOLD];
            for (int i = 0; i < THRESHOLD; i++)
                array[i] = strlen(mySequences[index++]);
            int longestSequenceIndex = getLongestSequenceIndex(array, THRESHOLD);
            memcpy(arrayOfSequencesToSend[j], mySequences[longestSequenceIndex], strlen(mySequences[longestSequenceIndex]) + 1);
        }
        MPI_Send(&numOfSequencesToSend, 1, MPI_INT, SLAVE_ID, TAG, MPI_COMM_WORLD);
        for (int i = 0; i < numOfSequencesToSend; i++)
        {
            int length = strlen(arrayOfSequencesToSend[i]);
            MPI_Send(&length, 1, MPI_INT, SLAVE_ID, TAG, MPI_COMM_WORLD);
            MPI_Send(arrayOfSequencesToSend[i],length, MPI_CHAR, SLAVE_ID, TAG, MPI_COMM_WORLD);
        }
    }
    else
    {
        int array[numOfSequences];
        for (int i = 0; i < numOfSequences; i++)
            array[i] = strlen(mySequences[i]);
        int longestSeqIndex = getLongestSequenceIndex(array, numOfSequences);
        MPI_Send(&numOfSequencesToSend, 1, MPI_INT, SLAVE_ID, TAG, MPI_COMM_WORLD);
        int length = strlen(mySequences[longestSeqIndex]);
        MPI_Send(&length, 1, MPI_INT, SLAVE_ID, TAG, MPI_COMM_WORLD);
        MPI_Send(mySequences[longestSeqIndex], length, MPI_CHAR, SLAVE_ID, TAG, MPI_COMM_WORLD);
    }
}

int getLongestSequenceIndex(int array[], int numOfElements)
{
    int maxLength = -1;
    int maxIndex = -1;
    for (int i = 0; i < numOfElements; i++)
        if (array[i] > maxLength)
        {
            maxIndex = i;
            maxLength = array[i];
        }
    return maxIndex;
}

void getTheSequence()
{
    MPI_Status status;
    int numOfSequancesToAllocate;
    MPI_Recv(&numOfSequancesToAllocate, 1, MPI_INT, MASTER_ID, TAG, MPI_COMM_WORLD, &status);
    char sequencesToCheck[numOfSequancesToAllocate][SEQUENCE_2_LIMIT];
    for (int i = 0; i < numOfSequancesToAllocate; i++)
    {
        int length;
        printf("checker\n");
        MPI_Recv(&length, 1, MPI_INT, MASTER_ID, TAG, MPI_COMM_WORLD, &status);
        printf("length from slave is: %d\n", length);
        MPI_Recv(sequencesToCheck[i], length, MPI_CHAR, MASTER_ID, TAG, MPI_COMM_WORLD, &status);
        printf("sequencesToCheck from slave is: %s\n", sequencesToCheck[i]);
    }
    printf("\n\nAs The Slave, I received %s \n\n", sequencesToCheck[0]);
    printf("\n\nAs The Slave, I received %s \n\n", sequencesToCheck[1]);
}

void readVarsFromFile(float *w1, float *w2, float *w3, float *w4, char *mainSequence, int *numOfSequences, FILE *reader)
{
    fscanf(reader, "%f", w1);
    fscanf(reader, "%f", w2);
    fscanf(reader, "%f", w3);
    fscanf(reader, "%f", w4);
    fscanf(reader, "%s", mainSequence);
    fscanf(reader, "%d", numOfSequences);
}

void checkSequence(char *mainSequence, char *checkedSequence, FILE *writer, float w1, float w2, float w3, float w4)
{
    int *k = (int *)malloc(sizeof(int));
    int *n = (int *)malloc(sizeof(int));
    getclosestOffsetAndHyphen(mainSequence, checkedSequence, n, k, w1, w2, w3, w4);
    printf("The %s\nSequence is done and it closest offset is %d and its closest hyphen is %d\n", checkedSequence, *n, *k);
    fprintf(writer, "%s", checkedSequence);
    fprintf(writer, " %d ", *n);
    fprintf(writer, " %d\n\n", *k);
}

void getclosestOffsetAndHyphen(char *mainSequence, char *checkedSequence, int *n, int *k, float w1, float w2, float w3, float w4)
// This function is the MAJOR one to take care the *n and the *k
{
    printf("strlen(mainSequence) is %ld\n", strlen(mainSequence));
    printf("strlen(checkedSequence) is %ld\n", strlen(checkedSequence));
    int offsetsRangeSize = strlen(mainSequence) - strlen(checkedSequence);
    printf("offsetsRangeSize is %d\n", offsetsRangeSize);
    float tempNAlignment;
    float closestOffset = -1;
    // Trying all possible offsets
    // #pragma omp parallel for
    for (int offset = 0; offset < offsetsRangeSize - 1; offset++) // The -1 because I added the '-'
    {
        tempNAlignment = getAlignmentForClosestHypenAndCurrentOffset(mainSequence, checkedSequence, offset, k, w1, w2, w3, w4);
        if (tempNAlignment > closestOffset)
        {
            closestOffset = tempNAlignment;
            *n = offset;
        }
        printf("%c", mainSequence[offset]);
    }
    printf("\nThat's it for the current String\n");
}

float getAlignmentForClosestHypenAndCurrentOffset(char *mainSequence, char *checkedSequence, int offset, int *k, float w1, float w2, float w3, float w4)
{
    char *currentSigns = (char *)malloc(strlen(checkedSequence + 1) * sizeof(char));
    float closestHyphenIndexSum = -1;
    int tempSum;
    // For each offset, trying all possible hyphens (n options)
    // w-ord, wo-rd, wor-d, word-
    for (int hyphenIndex = 1; hyphenIndex < strlen(checkedSequence) + 1; hyphenIndex++)
    {
        generateSignsForCurrentOffsetAndCurrentHyphenIndex(mainSequence, checkedSequence, offset, hyphenIndex, currentSigns);
        tempSum = getAlignmentSum(currentSigns, w1, w2, w3, w4) - w4; // because of the hyphen
        if (tempSum > closestHyphenIndexSum)
        {
            closestHyphenIndexSum = tempSum;
            *k = hyphenIndex;
        }
    }
    // printf("For the first string, the hyphen index %d is the best\n", *k);
    free(currentSigns);
    return closestHyphenIndexSum;
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

float getAlignmentSum(char *signs, float w1, float w2, float w3, float w4)
{
    float sum = 0;
    for (int i = 0; i < strlen(signs); i++)
    {
        switch (signs[i])
        {
        case '*':
            sum += w1;
            break;
        case ':':
            sum += w2;
            break;
        case '.':
            sum += w3;
            break;
        default:
            sum += w4;
            break;
        }
    }
    return sum;
}
