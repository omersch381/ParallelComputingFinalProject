#include "mpi.h"
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "cudaChecker.h"

#define GROUP_STRING_SIZE_LIMIT 7
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
#define MATCH '*'
#define CONSERVATIVE ':'
#define SEMI_CONSERVATIVE '.'
#define NO_MATCH ' '
#define HYPHEN '-'
#define PRINT_TO_TERMINAL 1

typedef struct CheckedSequences
{
    char *sequence;
    int n;
    int k;
} CheckedSequence;

void sanityCheck(int numOfProcesses);
void masterJob(int rank);
void readVarsFromFile(float *w1, float *w2, float *w3, float *w4, char *mainSequence, int *numOfSequences, FILE *reader);
void sendTheSlaveItsPartOfTheSequences(char *mainSequence, CheckedSequence mySequences[], int numOfSequences, int *sequencesToIgnore, int rank, float w1, float w2, float w3, float w4);
void sendMultipleRoundsOfSequences(CheckedSequence mySequences[], int numOfSequencesToSend, int *sequencesToIgnore);
void sendOneRoundOfSequences(CheckedSequence mySequences[], int numOfSequences, int numOfSequencesToSend, int *sequencesToIgnore);
int getLongestSequenceIndex(int array[], int numOfElements);
void checkTheSequences(CheckedSequence mySequences[], int numOfSequences, int *sequencesToIgnore, char *mainSequence, float w1, float w2, float w3, float w4);
void receiveTheSequencesFromTheSlave(int numOfSequences, CheckedSequence sequencesToReceive[], FILE *writer);
void writeThesequences(CheckedSequence mySequences[], int *sequencesToIgnore, int numOfSequences, FILE *writer);
void writeSequenceToFile(CheckedSequence sequenceToWrite, FILE *writer);
void slaveJob(int rank);
void sendToMasterTheResults(CheckedSequence sequencesToReceive[], int numOfsequencesToCheck);
void mpiSendReceiveInitialVariables(int *mainSequenceLength, int *numOfsequencesToSendReceive, char *mainSequence, float *w1, float *w2, float *w3, float *w4, int rank);
void checkSequence(char *mainSequence, char *checkedSequence, float w1, float w2, float w3, float w4, int *n, int *k);
float getAlignmentForClosestHypenAndCurrentOffset(char *mainSequence, char *checkedSequence, int offset, int *k, float w1, float w2, float w3, float w4, char *currentSigns);
void generateSignsForCurrentOffsetAndCurrentHyphenIndex(char *mainSequence, char *checkedSequence, int offset, int hyphenIndex, char *currentSigns);
char checkAndSetProximity(char mainChar, char checkedChar);
int areConservative(char mainChar, char checkedChar);
int areSemiConservative(char mainChar, char checkedChar);
// int areTheCharsInGroup(char mainChar, char checkedChar, const char groupToCheck[][GROUP_STRING_SIZE_LIMIT], int arraySize);
// int areTheCharsInGroupGPU(char mainChar, char checkedChar,
//                           char groupToCheck[][GROUP_STRING_SIZE_LIMIT],
//                           int arraySize);
float getAlignmentSum(char *signs, float w1, float w2, float w3, float w4, int offset, int size);
void checkIfNotNull(void *allocation);
void moveTheHyphenInOneIndexInSignsChain(char *currentSigns, int hyphenIndex, int offset, char *checkedSequence, char *mainSequence);
void addHyphenAt(char *checkedSequence, int index);

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
    {
        checkIfNotNull(mySequences[i].sequence = (char *)malloc(sizeof(char) * SEQUENCE_2_LIMIT));
        fscanf(reader, "%s", mySequences[i].sequence);
    }

    // sequencesToIgnore are the ones that the MASTER sends to the SLAVE
    checkIfNotNull(sequencesToIgnore = (int *)malloc(numOfSequences / THRESHOLD * sizeof(int)));

    double start = MPI_Wtime();
    sendTheSlaveItsPartOfTheSequences(mainSequence, mySequences, numOfSequences, sequencesToIgnore, rank, w1, w2, w3, w4);
    checkTheSequences(mySequences, numOfSequences, sequencesToIgnore, mainSequence, w1, w2, w3, w4);
    receiveTheSequencesFromTheSlave(numOfSequences, sequencesToReceive, writer);
    double end = MPI_Wtime();

    writeThesequences(mySequences, sequencesToIgnore, numOfSequences, writer);

    for (int i = 0; i < numOfSequences; i++)
        free(mySequences[i].sequence);

    free(mainSequence);
    free(sequencesToIgnore);

    fclose(writer);
    fclose(reader);

    if (PRINT_TO_TERMINAL)
        printf("The time it took is %lf\n", end - start);
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
        sendMultipleRoundsOfSequences(mySequences, numOfSequencesToSend, sequencesToIgnore);
    else
        sendOneRoundOfSequences(mySequences, numOfSequences, numOfSequencesToSend, sequencesToIgnore);
}

void sendMultipleRoundsOfSequences(CheckedSequence mySequences[], int numOfSequencesToSend, int *sequencesToIgnore)
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

void sendOneRoundOfSequences(CheckedSequence mySequences[], int numOfSequences, int numOfSequencesToSend, int *sequencesToIgnore)
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
#pragma omp parallel for
    for (int i = 0; i < numOfSequences; i++)
    {
        if (i != *sequencesToIgnore)
        {
            checkSequence(mainSequence, mySequences[i].sequence, w1, w2, w3, w4, &mySequences[i].n, &mySequences[i].k);
        }
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
        checkIfNotNull(sequencesToReceive[i].sequence = (char *)malloc(sizeof(char) * SEQUENCE_2_LIMIT));
        MPI_Recv(sequencesToReceive[i].sequence, lengthOfEachSequence, MPI_CHAR, SLAVE_ID, TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&sequencesToReceive[i].n, 1, MPI_INT, SLAVE_ID, TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&sequencesToReceive[i].k, 1, MPI_INT, SLAVE_ID, TAG, MPI_COMM_WORLD, &status);
        writeSequenceToFile(sequencesToReceive[i], writer);
        free(sequencesToReceive[i].sequence);
    }
}

void writeThesequences(CheckedSequence mySequences[], int *sequencesToIgnore, int numOfSequences, FILE *writer)
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
    int mainSequenceLength, numOfsequencesToCheck;
    char *mainSequence;
    float w1, w2, w3, w4;
    checkIfNotNull(mainSequence = (char *)malloc(SEQUENCE_1_LIMIT * sizeof(char)));

    mpiSendReceiveInitialVariables(&mainSequenceLength, &numOfsequencesToCheck, mainSequence, &w1, &w2, &w3, &w4, rank);

    CheckedSequence sequencesToReceive[numOfsequencesToCheck];
    for (int i = 0; i < numOfsequencesToCheck; i++)
        checkIfNotNull(sequencesToReceive[i].sequence = (char *)malloc(sizeof(char) * SEQUENCE_2_LIMIT));

    for (int i = 0; i < numOfsequencesToCheck; i++)
    {
        int lengthOfEachSequence;
        MPI_Recv(&lengthOfEachSequence, 1, MPI_INT, MASTER_ID, TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(sequencesToReceive[i].sequence, lengthOfEachSequence, MPI_CHAR, MASTER_ID, TAG, MPI_COMM_WORLD, &status);
    }

    // Trying all possible offsets
    for (int i = 0; i < numOfsequencesToCheck; i++)
        checkSequence(mainSequence, sequencesToReceive[i].sequence, w1, w2, w3, w4, &sequencesToReceive->n, &sequencesToReceive->k);

    sendToMasterTheResults(sequencesToReceive, numOfsequencesToCheck);
    free(mainSequence);
    for (int i = 0; i < numOfsequencesToCheck; i++)
        free(sequencesToReceive[i].sequence);
}

void sendToMasterTheResults(CheckedSequence sequencesToReceive[], int numOfsequencesToCheck)
{
    for (int i = 0; i < numOfsequencesToCheck; i++)
    {
        int lengthOfEachSequence = strlen(sequencesToReceive[i].sequence);
        MPI_Send(&lengthOfEachSequence, 1, MPI_INT, MASTER_ID, TAG, MPI_COMM_WORLD);
        MPI_Send(sequencesToReceive[i].sequence, lengthOfEachSequence, MPI_CHAR, MASTER_ID, TAG, MPI_COMM_WORLD);
        MPI_Send(&sequencesToReceive[i].n, 1, MPI_INT, MASTER_ID, TAG, MPI_COMM_WORLD);
        MPI_Send(&sequencesToReceive[i].k, 1, MPI_INT, MASTER_ID, TAG, MPI_COMM_WORLD);
    }
}

void mpiSendReceiveInitialVariables(int *mainSequenceLength, int *numOfsequencesToSendReceive, char *mainSequence, float *w1, float *w2, float *w3, float *w4, int rank)
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

        MPI_Send(numOfsequencesToSendReceive, 1, MPI_INT, SLAVE_ID, TAG, MPI_COMM_WORLD);
    }
    else // SLAVE_ID
    {
        MPI_Recv(mainSequenceLength, 1, MPI_INT, MASTER_ID, TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(mainSequence, *mainSequenceLength, MPI_CHAR, MASTER_ID, TAG, MPI_COMM_WORLD, &status);

        MPI_Recv(w1, 1, MPI_FLOAT, MASTER_ID, TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(w2, 1, MPI_FLOAT, MASTER_ID, TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(w3, 1, MPI_FLOAT, MASTER_ID, TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(w4, 1, MPI_FLOAT, MASTER_ID, TAG, MPI_COMM_WORLD, &status);

        MPI_Recv(numOfsequencesToSendReceive, 1, MPI_INT, MASTER_ID, TAG, MPI_COMM_WORLD, &status);
    }
}

void checkSequence(char *mainSequence, char *checkedSequence, float w1, float w2, float w3, float w4, int *n, int *k)
{
    int offsetsRangeSize = strlen(mainSequence) - strlen(checkedSequence);
    float tempNAlignment;
    float closestOffsetSum = -1;
    int hyphenHolder = -1;
    char *currentSigns;

    checkIfNotNull(currentSigns = (char *)malloc(strlen(checkedSequence) + 1));
    currentSigns[strlen(checkedSequence)] = 0;

    for (int offset = 0; offset < offsetsRangeSize; offset++)
    {
        char *backup = strdup(checkedSequence); // I would like to mention that I could have just rewinded the hyphen index
        // and avoid this allocation.

        tempNAlignment = getAlignmentForClosestHypenAndCurrentOffset(mainSequence, checkedSequence, offset, k, w1, w2, w3, w4, currentSigns);
        if (tempNAlignment > closestOffsetSum)
        {
            closestOffsetSum = tempNAlignment;
            *n = offset;
            hyphenHolder = *k; // the k might be overriden on next iteration, I would like to keep the highest (n,k) tuple.
        }
        strcpy(checkedSequence, backup);
        free(backup);
    }
    free(currentSigns);
    *k = hyphenHolder;

    if (PRINT_TO_TERMINAL)
        printf("The biggest sum is: %f, the biggest offset is %d and hyphen %d\n", closestOffsetSum, *n, *k);
}

float getAlignmentForClosestHypenAndCurrentOffset(char *mainSequence, char *checkedSequence, int offset, int *k, float w1, float w2, float w3, float w4, char *currentSigns)
{
    float closestHyphenIndexSum = -1;
    float tempSum = -1;

    // For each offset, trying all possible hyphens (n options), since index = 1
    // w-ord, wo-rd, wor-d, word-

    for (int hyphenIndex = 1; hyphenIndex < strlen(checkedSequence); hyphenIndex++)
    {
        if (hyphenIndex == 1)
        {
            addHyphenAt(checkedSequence, hyphenIndex);
            generateSignsForCurrentOffsetAndCurrentHyphenIndex(mainSequence, checkedSequence, offset, hyphenIndex, currentSigns);
        }
        else
            moveTheHyphenInOneIndexInSignsChain(currentSigns, hyphenIndex, offset, checkedSequence, mainSequence);

        tempSum = getAlignmentSum(currentSigns, w1, w2, w3, w4, offset, strlen(checkedSequence));
        if (tempSum > closestHyphenIndexSum)
        {
            closestHyphenIndexSum = tempSum;
            *k = hyphenIndex;
        }
    }
    return closestHyphenIndexSum;
}

void moveTheHyphenInOneIndexInSignsChain(char *currentSigns, int hyphenIndex, int offset, char *checkedSequence, char *mainSequence)
{
    char selectedChar = checkedSequence[hyphenIndex];
    currentSigns[hyphenIndex - 1] = checkAndSetProximity(mainSequence[offset + hyphenIndex - 1], selectedChar);
    currentSigns[hyphenIndex] = checkAndSetProximity(mainSequence[offset + hyphenIndex], HYPHEN);
}

void addHyphenAt(char *checkedSequence, int index)
{
    checkIfNotNull(checkedSequence = (char *)realloc(checkedSequence, strlen(checkedSequence) + 2));
    checkedSequence[strlen(checkedSequence) + 1] = 0;
    for (int i = strlen(checkedSequence); i > index; i--)
        checkedSequence[i] = checkedSequence[i - 1];

    checkedSequence[index] = HYPHEN;
}

void generateSignsForCurrentOffsetAndCurrentHyphenIndex(char *mainSequence, char *checkedSequence, int offset, int hyphenIndex, char *currentSigns)
{
    int j = 0;
    currentSigns[strlen(currentSigns) - 1] = 0;
    for (int mainSequenceIndex = offset; mainSequenceIndex < offset + strlen(checkedSequence); mainSequenceIndex++)
    {
        currentSigns[j] = checkAndSetProximity(mainSequence[mainSequenceIndex], checkedSequence[j]);
        j++;
    }
}

char checkAndSetProximity(char mainChar, char checkedChar)
{
    if (mainChar == checkedChar)
        return MATCH;
    else if (areConservative(mainChar, checkedChar))
        return CONSERVATIVE;
    else if (areSemiConservative(mainChar, checkedChar))
        return SEMI_CONSERVATIVE;
    else
        return NO_MATCH;
}

int areConservative(char mainChar, char checkedChar)
{
    char conservativeGroup[NUMBER_OF_CONSERVATIVE_STRINGS][GROUP_STRING_SIZE_LIMIT] = {"NDEQ", "NEQK", "STA", "MILV", "QHRK", "NHQK", "FYW", "HY", "MILF"};
    return areTheCharsInGroupGPU(mainChar, checkedChar, conservativeGroup, 9);
}

int areSemiConservative(char mainChar, char checkedChar)
{
    char semiConservativeGroup[NUMBER_OF_SEMI_CONSERVATIVE_STRINGS][GROUP_STRING_SIZE_LIMIT] = {"SAG", "ATV", "CSA", "SGND", "STPA", "STNK", "NEQHRK", "NDEQHK", "SNDEQK", "HFY", "FVLIM"};
    return areTheCharsInGroupGPU(mainChar, checkedChar, semiConservativeGroup, 11);
}

// int areTheCharsInGroup(char mainChar, char checkedChar, const char groupToCheck[][GROUP_STRING_SIZE_LIMIT], int arraySize)
// {
//     int isMainCharInTheGroup = 0;
//     int isCheckedCharInTheGroup = 0;

//     for (int i = 0; i < arraySize; i++)
//     {
//         for (int j = 0; j < strlen(groupToCheck[i]); j++)
//         {
//             if (mainChar == groupToCheck[i][j])
//                 isMainCharInTheGroup = 1;
//             if (checkedChar == groupToCheck[i][j])
//                 isCheckedCharInTheGroup = 1;
//         }
//         if (isMainCharInTheGroup && isCheckedCharInTheGroup)
//             return 1;
//         else
//         {
//             isMainCharInTheGroup = 0;
//             isCheckedCharInTheGroup = 0;
//         }
//     }
//     return 0;
// }


float getAlignmentSum(char *signs, float w1, float w2, float w3, float w4, int offset, int size)
{
    float sum = 0;
    for (int i = 0; i < offset + size; i++)
    {
        switch (signs[i])
        {
        case MATCH:
            sum += w1;
            break;
        case CONSERVATIVE:
            sum -= w2;
            break;
        case SEMI_CONSERVATIVE:
            sum -= w3;
            break;
        case NO_MATCH:
            sum -= w4;
            break;
        default:
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
