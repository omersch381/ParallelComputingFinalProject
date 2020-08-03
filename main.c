#include "mpi.h"
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define INPUT_FILE "input.txt"
#define MODE_READ "r"
#define SEQUENCE_1_LIMIT 3000
#define SEQUENCE_2_LIMIT 2000
#define GROUP_STRING_SIZE_LIMIT 7
#define OUTPUT_FILE "output.txt"
#define MODE_WRITE "w+"
#define SLAVE_ID 1
#define MASTER_ID 0
#define TAG 0
#define THRESHOLD 4
#define EXIT_MESSAGE "Please run the program with 2 processes! Exiting....\n"
#define NUMBER_OF_CONSERVATIVE_STRINGS 9
#define NUMBER_OF_SEMI_CONSERVATIVE_STRINGS 11

typedef struct CheckedSequences
{
    char sequence[SEQUENCE_2_LIMIT];
    int n;
    int k;
} CheckedSequence;

void sanityCheck(int numOfProcesses);
void masterJob(int rank);
void readVarsFromFile(float *w1, float *w2, float *w3, float *w4, char *mainSequence, int *numOfSequences, FILE *reader);
void sendTheSlaveItsPartOfTheSequences(char *mainSequence, CheckedSequence mySequences[], int numOfSequences, int *sequencesToIgnore, int rank, float w1, float w2, float w3, float w4);
void handleMultipleRoundsOfSequences(CheckedSequence mySequences[], int numOfSequencesToSend, int *sequencesToIgnore);
void handleOneRoundOfSequences(CheckedSequence mySequences[], int numOfSequences, int numOfSequencesToSend, int *sequencesToIgnore);
int getLongestSequenceIndex(int array[], int numOfElements);
void checkTheSequences(CheckedSequence mySequences[], int numOfSequences, int *sequencesToIgnore, char *mainSequence, float w1, float w2, float w3, float w4);
void receiveTheSequencesFromTheSlave(int numOfSequences, CheckedSequence sequencesToReceive[], FILE *writer);
void writeTheSequances(CheckedSequence mySequences[], int *sequencesToIgnore, int numOfSequences, FILE *writer);
void writeSequenceToFile(CheckedSequence sequenceToWrite, FILE *writer);
void slaveJob(int rank);
void sendToMasterTheResults(CheckedSequence sequencesToReceive[], int numOfSequancesToCheck);
void mpiSendReceiveInitialVariables(int *mainSequenceLength, int *numOfSequancesToSendReceive, char *mainSequence, float *w1, float *w2, float *w3, float *w4, int rank);
void checkSequence(char *mainSequence, char *checkedSequence, float w1, float w2, float w3, float w4, int *n, int *k);
float getAlignmentForClosestHypenAndCurrentOffset(char *mainSequence, char *checkedSequence, int offset, int *k, float w1, float w2, float w3, float w4);
void generateSignsForCurrentOffsetAndCurrentHyphenIndex(char *mainSequence, char *checkedSequence, int offset, int hyphenIndex, char *currentSigns);
char checkAndSetProximity(char mainChar, char checkedChar);
int areConservative(char mainChar, char checkedChar);
int areSemiConservative(char mainChar, char checkedChar);
int areTheCharsInGroup(char mainChar, char checkedChar, const char groupToCheck[][GROUP_STRING_SIZE_LIMIT], int arraySize);
// int areTheCharsInGroupGPU(char mainChar, char checkedChar,
//                           const char groupToCheck[][GROUP_STRING_SIZE_LIMIT],
//                           int arraySize);
float getAlignmentSum(char *signs, float w1, float w2, float w3, float w4);
void checkIfNotNull(void *allocation);

int main(int argc, char *argv[])
{
    int numOfProcesses, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numOfProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    sanityCheck(numOfProcesses);

    if (rank == MASTER_ID)
        masterJob(rank);
    else
        slaveJob(rank);

    MPI_Finalize();

    return 0;
}

void sanityCheck(int numOfProcesses)
{
    FILE *errorWriter;
    if (numOfProcesses != 2)
    {
        printf(EXIT_MESSAGE);
        checkIfNotNull(errorWriter = fopen(OUTPUT_FILE, MODE_WRITE));
        fprintf(errorWriter, EXIT_MESSAGE);
        fclose(errorWriter);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void masterJob(int rank)
{
    FILE *reader, *writer;
    float w1, w2, w3, w4;
    int numOfSequences;
    char *mainSequence;
    int *sequencesToIgnore;

    checkIfNotNull(mainSequence = (char *)malloc(SEQUENCE_1_LIMIT * sizeof(char)));
    checkIfNotNull(reader = fopen(INPUT_FILE, MODE_READ));
    checkIfNotNull(writer = fopen(OUTPUT_FILE, MODE_WRITE));
    readVarsFromFile(&w1, &w2, &w3, &w4, mainSequence, &numOfSequences, reader);
    CheckedSequence mySequences[numOfSequences];
    CheckedSequence sequencesToReceive[numOfSequences / THRESHOLD];

    for (int i = 0; i < numOfSequences; i++)
        fscanf(reader, "%s", mySequences[i].sequence);

    // sequencesToIgnore are the ones that the MASTER sends to the SLAVE
    checkIfNotNull(sequencesToIgnore = (int *)malloc(numOfSequences / THRESHOLD * sizeof(int)));

    printf("1\n");
    sendTheSlaveItsPartOfTheSequences(mainSequence, mySequences, numOfSequences, sequencesToIgnore, rank, w1, w2, w3, w4);
    printf("2\n");
    checkTheSequences(mySequences, numOfSequences, sequencesToIgnore, mainSequence, w1, w2, w3, w4);
    printf("3\n");
    receiveTheSequencesFromTheSlave(numOfSequences, sequencesToReceive, writer);
    printf("4\n");
    writeTheSequances(mySequences, sequencesToIgnore, numOfSequences, writer);
    printf("5\n");

    free(mainSequence);
    free(sequencesToIgnore);
    fclose(writer);
    fclose(reader);
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

void checkTheSequences(CheckedSequence mySequences[], int numOfSequences, int *sequencesToIgnore, char *mainSequence, float w1, float w2, float w3, float w4)
{
    int counterToRevert = 0;
    for (int i = 0; i < numOfSequences; i++)
    {
        if (i != *sequencesToIgnore)
            checkSequence(mainSequence, mySequences[i].sequence, w1, w2, w3, w4, &mySequences[i].n, &mySequences[i].k);
        else
        {
            sequencesToIgnore++;
            counterToRevert++;
        }
    }
    for (int i = 0; i < counterToRevert; i++)
        sequencesToIgnore--; // Reverting the ignore pointer because I will use it afterwards.
}

void receiveTheSequencesFromTheSlave(int numOfSequences, CheckedSequence sequencesToReceive[], FILE *writer)
{
    MPI_Status status;
    int numOfSequencesToReceive = numOfSequences / THRESHOLD;
    for (int i = 0; i < numOfSequencesToReceive; i++)
    {
        int lengthOfEachSequence;
        MPI_Recv(&lengthOfEachSequence, 1, MPI_INT, SLAVE_ID, TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(sequencesToReceive[i].sequence, lengthOfEachSequence, MPI_CHAR, SLAVE_ID, TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&sequencesToReceive[i].n, 1, MPI_INT, SLAVE_ID, TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&sequencesToReceive[i].k, 1, MPI_INT, SLAVE_ID, TAG, MPI_COMM_WORLD, &status);
        writeSequenceToFile(sequencesToReceive[i], writer);
    }
}

void writeTheSequances(CheckedSequence mySequences[], int *sequencesToIgnore, int numOfSequences, FILE *writer)
{
    for (int i = 0; i < numOfSequences; i++)
    {
        if (i == *sequencesToIgnore)
        {
            sequencesToIgnore++;
            continue;
        }
        else
            writeSequenceToFile(mySequences[i], writer);
    }
}

void writeSequenceToFile(CheckedSequence sequenceToWrite, FILE *writer)
{
    fprintf(writer, "%s", sequenceToWrite.sequence);
    fprintf(writer, " %d ", sequenceToWrite.n);
    fprintf(writer, " %d\n\n", sequenceToWrite.k);
}

void slaveJob(int rank)
{
    MPI_Status status;
    int mainSequenceLength, numOfSequancesToCheck;
    char *mainSequence;
    float w1, w2, w3, w4;
    checkIfNotNull(mainSequence = (char *)malloc(SEQUENCE_1_LIMIT * sizeof(char)));
    mpiSendReceiveInitialVariables(&mainSequenceLength, &numOfSequancesToCheck, mainSequence, &w1, &w2, &w3, &w4, rank);

    CheckedSequence sequencesToReceive[numOfSequancesToCheck];
    for (int i = 0; i < numOfSequancesToCheck; i++)
    {
        int lengthOfEachSequence;
        MPI_Recv(&lengthOfEachSequence, 1, MPI_INT, MASTER_ID, TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(sequencesToReceive[i].sequence, lengthOfEachSequence, MPI_CHAR, MASTER_ID, TAG, MPI_COMM_WORLD, &status);
    }

    for (int i = 0; i < numOfSequancesToCheck; i++)
        checkSequence(mainSequence, sequencesToReceive[i].sequence, w1, w2, w3, w4, &sequencesToReceive->n, &sequencesToReceive->k);

    sendToMasterTheResults(sequencesToReceive, numOfSequancesToCheck);
    free(mainSequence);
}

void sendToMasterTheResults(CheckedSequence sequencesToReceive[], int numOfSequancesToCheck)
{
    for (int i = 0; i < numOfSequancesToCheck; i++)
    {
        int lengthOfEachSequence = strlen(sequencesToReceive[i].sequence);
        MPI_Send(&lengthOfEachSequence, 1, MPI_INT, MASTER_ID, TAG, MPI_COMM_WORLD);
        MPI_Send(sequencesToReceive[i].sequence, lengthOfEachSequence, MPI_CHAR, MASTER_ID, TAG, MPI_COMM_WORLD);
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

void checkSequence(char *mainSequence, char *checkedSequence, float w1, float w2, float w3, float w4, int *n, int *k)
{
    int offsetsRangeSize = strlen(mainSequence) - strlen(checkedSequence);
    float tempNAlignment;
    float closestOffsetSum = -1;

// Trying all possible offsets
// #pragma omp parallel for
    for (int offset = 0; offset < offsetsRangeSize - 1; offset++) // The -1 because I added the '-'
    {
        tempNAlignment = getAlignmentForClosestHypenAndCurrentOffset(mainSequence, checkedSequence, offset, k, w1, w2, w3, w4);
        if (tempNAlignment > closestOffsetSum)
        {
            closestOffsetSum = tempNAlignment;
            *n = offset;
        }
    }
    printf("The biggest sum is: %f, the biggest offset is %d and hyphen %d\n", closestOffsetSum, *n, *k);
}

float getAlignmentForClosestHypenAndCurrentOffset(char *mainSequence, char *checkedSequence, int offset, int *k, float w1, float w2, float w3, float w4)
{
    char *currentSigns;
    float closestHyphenIndexSum = -1;
    int tempSum;

    checkIfNotNull(currentSigns = (char *)malloc(strlen(checkedSequence + 1) * sizeof(char)));
    // For each offset, trying all possible hyphens (n options), since index = 1
    // w-ord, wo-rd, wor-d, word-
    for (int hyphenIndex = 1; hyphenIndex < strlen(checkedSequence) + 1; hyphenIndex++)
    {
        printf("hyphenIndex checked is: %d\n", hyphenIndex);
        generateSignsForCurrentOffsetAndCurrentHyphenIndex(mainSequence, checkedSequence, offset, hyphenIndex, currentSigns);
        tempSum = getAlignmentSum(currentSigns, w1, w2, w3, w4) - w4; // because of the hyphen
        printf("alignment sum is: %d\n", tempSum);
        if (tempSum > closestHyphenIndexSum)
        {
            closestHyphenIndexSum = tempSum;
            *k = hyphenIndex;
            printf("NEW HIGHEST ALIGNMENT SUM!! %d\n", tempSum);
        }
    }
    free(currentSigns);
    // printf("biggest k is: %d\n", *k);
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
    const char conservativeGroup[NUMBER_OF_CONSERVATIVE_STRINGS][GROUP_STRING_SIZE_LIMIT] = {"NDEQ", "NEQK", "STA", "MILV", "QHRK", "NHQK", "FYW", "HY", "MILF"};
    return areTheCharsInGroup(mainChar, checkedChar, conservativeGroup, 9);
}

int areSemiConservative(char mainChar, char checkedChar)
{
    const char semiConservativeGroup[NUMBER_OF_SEMI_CONSERVATIVE_STRINGS][GROUP_STRING_SIZE_LIMIT] = {"SAG", "ATV", "CSA", "SGND", "STPA", "STNK", "NEQHRK", "NDEQHK", "SNDEQK", "HFY", "FVLIM"};
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

void checkIfNotNull(void *allocation)
{
    if (!allocation)
    {
        printf("Malloc failed!! Exiting...\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}