#include "mpi.h"
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define INPUT_FILE "input.txt"
#define MODE_READ "r"
#define SEQUENCE_1_LIMIT 3000
#define SEQUENCE_2_LIMIT 2000 // TODO
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
#define NO_MATCH 'X'

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
float getAlignmentSum(char *signs, float w1, float w2, float w3, float w4, int offset, int size);
void checkIfNotNull(void *allocation);
void moveTheHyphenInOneIndexInSignsChain(char* currentSigns, int hyphenIndex, int offset, char *checkedSequence, char *mainSequence);
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
    // else
    //     slaveJob(rank);

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
        mySequences[i].sequence = (char *)malloc(sizeof(char) * SEQUENCE_2_LIMIT);
        fscanf(reader, "%s", mySequences[i].sequence);
    }

    // sequencesToIgnore are the ones that the MASTER sends to the SLAVE
    checkIfNotNull(sequencesToIgnore = (int *)malloc(numOfSequences / THRESHOLD * sizeof(int)));

    // printf("1\n");
    // sendTheSlaveItsPartOfTheSequences(mainSequence, mySequences, numOfSequences, sequencesToIgnore, rank, w1, w2, w3, w4);
    // printf("2\n");
    checkTheSequences(mySequences, numOfSequences, sequencesToIgnore, mainSequence, w1, w2, w3, w4);
    // printf("3\n");
    // receiveTheSequencesFromTheSlave(numOfSequences, sequencesToReceive, writer);
    // printf("4\n");
    // writeTheSequances(mySequences, sequencesToIgnore, numOfSequences, writer);
    // printf("5\n");

    for (int i = 0; i < numOfSequences; i++)
    {
        free(mySequences[i].sequence);
    }
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
        {
            if (i == 3)
            {
                // printf("Check: %s\n", mySequences[i].sequence);
                checkSequence(mainSequence, mySequences[i].sequence, w1, w2, w3, w4, &mySequences[i].n, &mySequences[i].k);
            }
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
    // char *test = (char *)malloc(sizeof(char) * 6);
    // char *test2 = (char *)malloc(sizeof(char) * 3);
    // test[0] = 'P';
    // test[1] = 'S';
    // test[2] = 'H';
    // test[3] = 'L';
    // test[4] = 'Q';
    // test[5] = 'Y';
    // test2[0] = 'S';
    // test2[1] = 'H';
    // test2[2] = 'Q';

    int offsetsRangeSize = strlen(mainSequence) - strlen(checkedSequence);
    float tempNAlignment;
    float closestOffsetSum = -1;
    int hyphenHolder = -1;

    // Trying all possible offsets
    // #pragma omp parallel for
    for (int offset = 0; offset < offsetsRangeSize; offset++)
    {
        printf("From checkSequence, offset is %d\n", offset);
        tempNAlignment = getAlignmentForClosestHypenAndCurrentOffset(mainSequence, checkedSequence, offset, k, w1, w2, w3, w4);
        if (tempNAlignment > closestOffsetSum)
        {
            closestOffsetSum = tempNAlignment;
            *n = offset;
            hyphenHolder = *k;
        }
    }
    printf("The biggest sum is: %f, the biggest offset is %d and hyphen %d\n", closestOffsetSum, *n, hyphenHolder);
}

float getAlignmentForClosestHypenAndCurrentOffset(char *mainSequence, char *checkedSequence, int offset, int *k, float w1, float w2, float w3, float w4)
{
    // printf("The Length of checkSequence is: %ld\n", strlen(checkedSequence));
    char *currentSigns;
    float closestHyphenIndexSum = -1;
    float tempSum = -1;

    // checkIfNotNull(currentSigns = (char *)malloc(strlen(checkedSequence + 1) * sizeof(char)));
    // For each offset, trying all possible hyphens (n options), since index = 1
    // w-ord, wo-rd, wor-d, word-

    // char *backup = (char *)malloc(strlen(checkedSequence) * sizeof(char));
    // strcpy(backup, checkedSequence);
    // printf("check11\n");
    // printf("size is %ld\n", strlen(checkedSequence + 1));

    // checkIfNotNull(currentSigns = (char *)malloc(strlen(checkedSequence) + 1 * sizeof(char)));
    currentSigns = (char *)malloc(strlen(checkedSequence) + 2);
    currentSigns[strlen(checkedSequence) + 1] = 0;
    // printf("The length of checkedSequence is: %ld\n", strlen(checkedSequence));
    // printf("The length of signs is: %ld\n", strlen(currentSigns));
    // printf("check12\n");
    char* backup = strdup(checkedSequence);
    for (int hyphenIndex = 1; hyphenIndex < strlen(checkedSequence); hyphenIndex++)
    {
        if (hyphenIndex == 1)
        {
            // printf("check1\n");

            addHyphenAt(checkedSequence, hyphenIndex);
            // printf("check2\n");
            generateSignsForCurrentOffsetAndCurrentHyphenIndex(mainSequence, checkedSequence, offset, hyphenIndex, currentSigns);
            // printf("check3\n");
        }
        else {
            moveTheHyphenInOneIndexInSignsChain(currentSigns, hyphenIndex, offset, checkedSequence, mainSequence);
        }




        // // printf("HyphenIndex is %d\n", hyphenIndex);
        // checkIfNotNull(currentSigns = (char *)malloc(strlen(checkedSequence + 1) * sizeof(char)));
        // char *backup = (char *)malloc(strlen(checkedSequence) * sizeof(char));
        // strcpy(backup, checkedSequence);
        // printf("strlen of backup is %ld\n", strlen(backup));
        // printf("strlen of checkedSequence is %ld\n", strlen(checkedSequence));

        // addHyphenAt(backup, hyphenIndex);
        // generateSignsForCurrentOffsetAndCurrentHyphenIndex(mainSequence, backup, offset, hyphenIndex, currentSigns);
        tempSum = getAlignmentSum(currentSigns, w1, w2, w3, w4, offset, strlen(checkedSequence));
        // printf("I called getAlignment for the %d time\n", hyphenIndex);
        // printf("offset is %d, hyphenIndex checked is: %d, alignment sum is: %d\n", offset, hyphenIndex, tempSum);
        if (tempSum > closestHyphenIndexSum)
        {
            closestHyphenIndexSum = tempSum;
            *k = hyphenIndex;
            printf("NEW HIGHEST K!! %d, with the sum of %f and offset %d\n", hyphenIndex, tempSum, offset);
        }
        // printf("1\n");
        // free(backup);
        // printf("2\n");
        // checkedSequence = strdup(backup);
        // printf("3\n");

        // printf("4\n");
        // printf("hyphenIndex is %d on iteration num #%d\n", hyphenIndex, offset);
        // free(currentSigns);
    }
    // printf("checked signs is %s\n", currentSigns);
    free(currentSigns);
    // printf("5\n");
    // free(backup);
    // printf("6\n");
    // printf("biggest k is: %d, biggest offset is %d\n", *k, offset);
    // printf("for offset %d, biggest k is %d with the sum of %f\n", offset, *k, closestHyphenIndexSum);

    strcpy(checkedSequence, backup);
    return closestHyphenIndexSum;
}

void moveTheHyphenInOneIndexInSignsChain(char* currentSigns, int hyphenIndex, int offset, char *checkedSequence, char *mainSequence) {
    char selectedChar = checkedSequence[hyphenIndex];
    currentSigns[hyphenIndex - 1] = checkAndSetProximity(mainSequence[offset + hyphenIndex - 1], selectedChar);
    currentSigns[hyphenIndex] = checkAndSetProximity(mainSequence[offset + hyphenIndex], '-');
}

void addHyphenAt(char *checkedSequence, int index)
{
    // printf("1\n");
    // printf("%s\n", checkedSequence);
    checkedSequence = (char *)realloc(checkedSequence, strlen(checkedSequence) + 2);
    // printf("2\n");
    checkedSequence[strlen(checkedSequence) + 1] = 0;
    for (int i = strlen(checkedSequence); i > index; i--)
        checkedSequence[i] = checkedSequence[i - 1];
    checkedSequence[index] = '-';
}

void generateSignsForCurrentOffsetAndCurrentHyphenIndex(char *mainSequence, char *checkedSequence, int offset, int hyphenIndex, char *currentSigns)
{
    int j = 0;
    // printf("we start printin\n");
    // printf("We Compare %s to %s at offset %d and with hyphen %d\n", mainSequence, checkedSequence, offset, hyphenIndex);
    currentSigns[strlen(currentSigns) - 1] = 0;
    for (int mainSequenceIndex = offset; mainSequenceIndex < offset + strlen(checkedSequence); mainSequenceIndex++)
    {
        // printf("we compare the char %c to the char %c\n", mainSequence[mainSequenceIndex], checkedSequence[j]);
        // printf("The result is %c\n", checkAndSetProximity(mainSequence[mainSequenceIndex], checkedSequence[j]));
        currentSigns[j] = checkAndSetProximity(mainSequence[mainSequenceIndex], checkedSequence[j]);
        j++;
        // printf("%s\n", currentSigns);
        // printf("%c", currentSigns[mainSequenceIndex]);
    }
    // printf("\n");
    // printf("The signs chain is: %s, with the kength of %ld\n", currentSigns, strlen(currentSigns));
}

char checkAndSetProximity(char mainChar, char checkedChar)
{
    // printf("We compare %c to %c\n", mainChar, checkedChar);
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
    const char conservativeGroup[NUMBER_OF_CONSERVATIVE_STRINGS][GROUP_STRING_SIZE_LIMIT] ={ "NDEQ", "NEQK", "STA", "MILV", "QHRK", "NHQK", "FYW", "HY", "MILF" };
    return areTheCharsInGroup(mainChar, checkedChar, conservativeGroup, 9);
}

int areSemiConservative(char mainChar, char checkedChar)
{
    const char semiConservativeGroup[NUMBER_OF_SEMI_CONSERVATIVE_STRINGS][GROUP_STRING_SIZE_LIMIT] ={ "SAG", "ATV", "CSA", "SGND", "STPA", "STNK", "NEQHRK", "NDEQHK", "SNDEQK", "HFY", "FVLIM" };
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

float getAlignmentSum(char *signs, float w1, float w2, float w3, float w4, int offset, int size)
{
    float sum = 0;
    for (int i = 0; i < offset + size; i++)
    {
        switch (signs[i])
        {
        case MATCH:
            sum += w1;
            // printf("current sum for the %d'th char is %f\n", i, sum);
            break;
        case CONSERVATIVE:
            sum -= w2;
            // printf("current sum for the %d'th char is %f\n", i, sum);
            break;
        case SEMI_CONSERVATIVE:
            sum -= w3;
            // printf("current sum for the %d'th char is %f\n", i, sum);
            break;
        case NO_MATCH:
            sum -= w4;
            // printf("current sum for the %d'th char is %f\n", i, sum);
            break;
        default:
            break;
        }
    }
    // printf("The sum of the current signs %s is %f\n", signs, sum);
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