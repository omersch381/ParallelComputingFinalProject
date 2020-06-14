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

typedef struct CheckedSequences
{
    char sequence[SEQUENCE_2_LIMIT];
    int n;
    int k;
} CheckedSequence;

void readVarsFromFile(float *w1, float *w2, float *w3, float *w4, char *mainSequence, int *numOfSequences, FILE *reader);
void checkSequence(char *mainSequence, char *checkedSequence /*, FILE *writer*/, float w1, float w2, float w3, float w4, int *n, int *k);
void getclosestOffsetAndHyphen(char *mainSequence, char *checkedSequence, int *n, int *k, float w1, float w2, float w3, float w4);
float getAlignmentForClosestHypenAndCurrentOffset(char *mainSequence, char *checkedSequence, int offset, int *k, float w1, float w2, float w3, float w4);
void generateSignsForCurrentOffsetAndCurrentHyphenIndex(char *mainSequence, char *checkedSequence, int offset, int hyphenIndex, char *currentSigns);
char checkAndSetProximity(char mainChar, char checkedChar);
int areConservative(char mainChar, char checkedChar);
int areSemiConservative(char mainChar, char checkedChar);
int areTheCharsInGroup(char mainChar, char checkedChar, const char groupToCheck[][GROUP_STRING_SIZE_LIMIT], int arraySize);
float getAlignmentSum(char *signs, float w1, float w2, float w3, float w4);

void handleMultipleRoundsOfSequences(CheckedSequence mySequences[], int numOfSequencesToSend, int *sequencesToIgnore);
void handleOneRoundOfSequences(CheckedSequence mySequences[], int numOfSequences, int numOfSequencesToSend, int *sequencesToIgnore);
void sendTheSlaveItsPartOfTheSequences(char *mainSequence, CheckedSequence mySequences[], int numOfSequences, int *sequencesToIgnore, int rank, float w1, float w2, float w3, float w4);
int getLongestSequenceIndex(int array[], int numOfElements);
void slaveJob(int rank);
void mpiSendReceiveInitialVariables(int *mainSequenceLength, int *numOfSequancesToSendReceive, char *mainSequence, float *w1, float *w2, float *w3, float *w4, int rank);
void checkTheSequences(CheckedSequence mySequences[], int numOfSequences, int *sequencesToIgnore, char *mainSequence, float w1, float w2, float w3, float w4);
void receiveTheSequencesFromTheSlave(int numOfSequences, CheckedSequence sequencesToReceive[]);

int generateSignsOnGPU(char *mainSequence, char *checkedSequence, int offset,
                       int hyphenIndex, char *currentSigns);

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

    if (rank == MASTER_ID)
    {
        mainSequence = (char *)malloc(SEQUENCE_1_LIMIT * sizeof(char));
        reader = fopen(INPUT_FILE, MODE_READ);
        writer = fopen(OUTPUT_FILE, MODE_WRITE);
        readVarsFromFile(&w1, &w2, &w3, &w4, mainSequence, &numOfSequences, reader);
        CheckedSequence mySequences[numOfSequences];

        for (int i = 0; i < numOfSequences; i++)
            fscanf(reader, "%s", mySequences[i].sequence);

        // sequencesToIgnore are the ones that the MASTER sends to the SLAVE
        int *sequencesToIgnore = (int *)malloc(numOfSequences / THRESHOLD * sizeof(int));
        sendTheSlaveItsPartOfTheSequences(mainSequence, mySequences, numOfSequences, sequencesToIgnore, rank, w1, w2, w3, w4);
        checkTheSequences(mySequences, numOfSequences, sequencesToIgnore, mainSequence, w1, w2, w3, w4);
        CheckedSequence sequencesToReceive[numOfSequences / THRESHOLD];
        receiveTheSequencesFromTheSlave(numOfSequences, sequencesToReceive);

        // writeTheSequances();
    }
    else
    {
        slaveJob(rank);
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
        free(mainSequence);
    }

    // free(checkedSequence);

    MPI_Finalize();
    return 0;
}

// void writeTheSequances()
// {
// }
void receiveTheSequencesFromTheSlave(int numOfSequences, CheckedSequence sequencesToReceive[])
{
    MPI_Status status;
    int numOfSequencesToReceive = numOfSequences / THRESHOLD;
    for (int i = 0; i < numOfSequencesToReceive; i++)
    {
        int lengthOfEachSequence;
        MPI_Recv(&lengthOfEachSequence, 1, MPI_INT, SLAVE_ID, TAG, MPI_COMM_WORLD, &status);
        // //TODO: check it out
        // char *initializer = sequencesToReceive[i].sequence;
        // for (size_t i = 0; i < lengthOfEachSequence; i++)
        // {
        //     sequencesToReceive[i].sequence[i] = ' ';
        // }

        MPI_Recv(sequencesToReceive[i].sequence, lengthOfEachSequence, MPI_CHAR, SLAVE_ID, TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&sequencesToReceive[i].n, 1, MPI_INT, SLAVE_ID, TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&sequencesToReceive[i].k, 1, MPI_INT, SLAVE_ID, TAG, MPI_COMM_WORLD, &status);
        printf("\nI received the lengthOfEachSequence %d\n", lengthOfEachSequence);
        int actualLength = strlen(sequencesToReceive[0].sequence);
        printf("actual length is %d\n", actualLength);
        printf("\nI received the sequence %s\n", sequencesToReceive[0].sequence);
        printf("I received the offset %d\n", sequencesToReceive[0].n);
        printf("I received the hyphen %d\n", sequencesToReceive[0].k);
    }
}

void checkTheSequences(CheckedSequence mySequences[], int numOfSequences, int *sequencesToIgnore, char *mainSequence, float w1, float w2, float w3, float w4)
{
    int counterToRevert = 0;
    int *k = (int *)malloc(sizeof(int));
    int *n = (int *)malloc(sizeof(int));
    for (int i = 0; i < numOfSequences; i++)
    {
        if (i != *sequencesToIgnore)
        {
            // printf("\nI am sending %s sequence to check\n", mySequences[i].sequence);
            checkSequence(mainSequence, mySequences[i].sequence, w1, w2, w3, w4, n, k);
            mySequences[i].n = *n;
            mySequences[i].k = *k;
        }
        else
        {
            sequencesToIgnore++;
            counterToRevert++;
        }
    }
    for (int i = 0; i < counterToRevert; i++)
        sequencesToIgnore--; // Reverting the ignore pointer because I will use it afterwards.
    free(n);
    free(k);
}

void sendTheSlaveItsPartOfTheSequences(char *mainSequence, CheckedSequence mySequences[], int numOfSequences, int *sequencesToIgnore, int rank, float w1, float w2, float w3, float w4)
{
    int mainSequenceLength = strlen(mainSequence);
    int numOfSequencesToSend = numOfSequences / THRESHOLD;
    mpiSendReceiveInitialVariables(&mainSequenceLength, &numOfSequencesToSend, mainSequence, &w1, &w2, &w3, &w4, rank);

    if (numOfSequencesToSend > 1)
        handleMultipleRoundsOfSequences(mySequences, numOfSequencesToSend, sequencesToIgnore);
    else
        handleOneRoundOfSequences(mySequences, numOfSequences, numOfSequencesToSend, sequencesToIgnore);
}

void handleOneRoundOfSequences(CheckedSequence mySequences[], int numOfSequences, int numOfSequencesToSend, int *sequencesToIgnore)
// This method happens when the numOfSequences is equals or less than the THRESHOLD
{
    int array[numOfSequences];
    for (int i = 0; i < numOfSequences; i++)
        array[i] = strlen(mySequences[i].sequence);
    int longestSeqIndex = sequencesToIgnore[0] = getLongestSequenceIndex(array, numOfSequences);
    int lengthOfEachSequence = strlen(mySequences[longestSeqIndex].sequence);
    MPI_Send(&lengthOfEachSequence, 1, MPI_INT, SLAVE_ID, TAG, MPI_COMM_WORLD);
    MPI_Send(mySequences[longestSeqIndex].sequence, lengthOfEachSequence, MPI_CHAR, SLAVE_ID, TAG, MPI_COMM_WORLD);
}

void handleMultipleRoundsOfSequences(CheckedSequence mySequences[], int numOfSequencesToSend, int *sequencesToIgnore)
// This method happens when the numOfSequences is more than the THRESHOLD
{
    int index = 0;
    for (int j = 0; j < numOfSequencesToSend; j++)
    {
        int array[THRESHOLD];
        for (int i = 0; i < THRESHOLD; i++)
            array[i] = strlen(mySequences[index++].sequence);
        int longestSequenceIndex = sequencesToIgnore[j] = getLongestSequenceIndex(array, THRESHOLD) + j * THRESHOLD;
        memcpy(mySequences[j].sequence, mySequences[longestSequenceIndex].sequence, strlen(mySequences[longestSequenceIndex].sequence) + 1);
    }
    for (int i = 0; i < numOfSequencesToSend; i++)
    {
        int lengthOfEachSequence = strlen(mySequences[i].sequence);
        MPI_Send(&lengthOfEachSequence, 1, MPI_INT, SLAVE_ID, TAG, MPI_COMM_WORLD);
        MPI_Send(mySequences[i].sequence, lengthOfEachSequence, MPI_CHAR, SLAVE_ID, TAG, MPI_COMM_WORLD);
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

void slaveJob(int rank)
{
    MPI_Status status;
    int mainSequenceLength, numOfSequancesToAllocate;
    char *mainSequence = (char *)malloc(SEQUENCE_1_LIMIT * sizeof(char));
    float w1, w2, w3, w4;
    mpiSendReceiveInitialVariables(&mainSequenceLength, &numOfSequancesToAllocate, mainSequence, &w1, &w2, &w3, &w4, rank);

    CheckedSequence sequencesToReceive[numOfSequancesToAllocate];
    for (int i = 0; i < numOfSequancesToAllocate; i++)
    {
        int lengthOfEachSequence;
        MPI_Recv(&lengthOfEachSequence, 1, MPI_INT, MASTER_ID, TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(sequencesToReceive[i].sequence, lengthOfEachSequence, MPI_CHAR, MASTER_ID, TAG, MPI_COMM_WORLD, &status);
    }
    for (int i = 0; i < numOfSequancesToAllocate; i++)
        checkSequence(mainSequence, sequencesToReceive[i].sequence, w1, w2, w3, w4, &sequencesToReceive->n, &sequencesToReceive->k);
    for (int i = 0; i < numOfSequancesToAllocate; i++)
    {
        int lengthOfEachSequence = strlen(sequencesToReceive[i].sequence);
        printf("From the slave - Length after returning from method is: %d\n", lengthOfEachSequence);
        MPI_Send(&lengthOfEachSequence, 1, MPI_INT, MASTER_ID, TAG, MPI_COMM_WORLD);
        printf("From the slave - RealLength after returning from method is: %ld\n", strlen(sequencesToReceive[i].sequence));
        MPI_Send(sequencesToReceive[i].sequence, lengthOfEachSequence, MPI_CHAR, MASTER_ID, TAG, MPI_COMM_WORLD);
        printf("From the slave - the sequence is: %s\n", sequencesToReceive[i].sequence);
        MPI_Send(&sequencesToReceive[i].n, 1, MPI_INT, MASTER_ID, TAG, MPI_COMM_WORLD);
        MPI_Send(&sequencesToReceive[i].k, 1, MPI_INT, MASTER_ID, TAG, MPI_COMM_WORLD);
    }
}

void mpiSendReceiveInitialVariables(int *mainSequenceLength, int *numOfSequancesToSendReceive, char *mainSequence, float *w1, float *w2, float *w3, float *w4, int rank)
{
    MPI_Status status;
    if (rank == MASTER_ID)
    {
        MPI_Send(mainSequenceLength, 1, MPI_INT, SLAVE_ID, TAG, MPI_COMM_WORLD);
        MPI_Send(mainSequence, *mainSequenceLength, MPI_CHAR, SLAVE_ID, TAG, MPI_COMM_WORLD);

        MPI_Send(w1, 1, MPI_FLOAT, SLAVE_ID, TAG, MPI_COMM_WORLD);
        MPI_Send(w2, 1, MPI_FLOAT, SLAVE_ID, TAG, MPI_COMM_WORLD);
        MPI_Send(w3, 1, MPI_FLOAT, SLAVE_ID, TAG, MPI_COMM_WORLD);
        MPI_Send(w4, 1, MPI_FLOAT, SLAVE_ID, TAG, MPI_COMM_WORLD);

        MPI_Send(numOfSequancesToSendReceive, 1, MPI_INT, SLAVE_ID, TAG, MPI_COMM_WORLD);
    }
    else // SLAVE_ID
    {
        MPI_Recv(mainSequenceLength, 1, MPI_INT, MASTER_ID, TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(mainSequence, *mainSequenceLength, MPI_CHAR, MASTER_ID, TAG, MPI_COMM_WORLD, &status);

        MPI_Recv(w1, 1, MPI_FLOAT, MASTER_ID, TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(w2, 1, MPI_FLOAT, MASTER_ID, TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(w3, 1, MPI_FLOAT, MASTER_ID, TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(w4, 1, MPI_FLOAT, MASTER_ID, TAG, MPI_COMM_WORLD, &status);

        MPI_Recv(numOfSequancesToSendReceive, 1, MPI_INT, MASTER_ID, TAG, MPI_COMM_WORLD, &status);
    }
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

void checkSequence(char *mainSequence, char *checkedSequence /*, FILE *writer*/, float w1, float w2, float w3, float w4, int *n, int *k)
{
    // int *k = (int *)malloc(sizeof(int));
    // int *n = (int *)malloc(sizeof(int));
    getclosestOffsetAndHyphen(mainSequence, checkedSequence, n, k, w1, w2, w3, w4);
    // printf("The %s\nSequence is done and it closest offset is %d and its closest hyphen is %d\n", checkedSequence, *n, *k);
    // fprintf(writer, "%s", checkedSequence);
    // fprintf(writer, " %d ", *n);
    // fprintf(writer, " %d\n\n", *k);
}

void getclosestOffsetAndHyphen(char *mainSequence, char *checkedSequence, int *n, int *k, float w1, float w2, float w3, float w4)
// This function is the MAJOR one to take care the *n and the *k
{
    // printf("strlen(mainSequence) is %ld\n", strlen(mainSequence));
    // printf("strlen(checkedSequence) is %ld\n", strlen(checkedSequence));
    int offsetsRangeSize = strlen(mainSequence) - strlen(checkedSequence);
    // printf("offsetsRangeSize is %d\n", offsetsRangeSize);
    float tempNAlignment;
    float closestOffset = -1;
// Trying all possible offsets
#pragma omp parallel for
    for (int offset = 0; offset < offsetsRangeSize - 1; offset++) // The -1 because I added the '-'
    {
        tempNAlignment = getAlignmentForClosestHypenAndCurrentOffset(mainSequence, checkedSequence, offset, k, w1, w2, w3, w4);
        if (tempNAlignment > closestOffset)
        {
            closestOffset = tempNAlignment;
            *n = offset;
        }
        // printf("%c", mainSequence[offset]);
    }
    // printf("\nThat's it for the current String\n");
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
        // generateSignsForCurrentOffsetAndCurrentHyphenIndex(mainSequence, checkedSequence, offset, hyphenIndex, currentSigns);
        generateSignsOnGPU(mainSequence, checkedSequence, offset, hyphenIndex, currentSigns);
        tempSum = getAlignmentSum(currentSigns, w1, w2, w3, w4) - w4; // because of the hyphen
        if (tempSum > closestHyphenIndexSum)
        {
            closestHyphenIndexSum = tempSum;
            *k = hyphenIndex;
        }
    }
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
