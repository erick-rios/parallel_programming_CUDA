/**
 * @file vectorialSum.c
 * @brief This file contains the code to perform the addition of two vectors
 * 
 * This program performs the addition of two vectors of size 2^20 elements each.
 * The vectors are initialized with random values and the addition is performed
 * on the CPU. The time taken to perform the addition is calculated and printed.
 * 
 * @author ERICK JESUS RIOS GONZALEZ
 * @date 25/08/2024
 * @version 1.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


/**
 * @brief Function to initialize a vector with random values
 * 
 * This function initializes a vector with random values between 1 and 100.
 * 
 * @param vector The vector to be initialized
 * @param size The size of the vector
 */
void initVector(float *vector, int size) {
    for (int i = 0; i < size; i++) {
        vector[i] = (float)(rand() % 100 + 1);
    }
}

/**
 * @brief Function to perform the addition of two vectors
 * 
 * This function performs the addition of two vectors and stores the result in a third vector.
 * 
 * @param vectorA The first vector
 * @param vectorB The second vector
 * @param vectorC The resulting vector
 */
void vectorialSum(float *vectorA, float *vectorB, float *vectorC, int size) {
    for (int i = 0; i < size; i++) {
        vectorC[i] = vectorA[i] + vectorB[i];
    }
}

/**
 * @brief Function to print the elements of a vector
 * 
 * This function prints the elements of a vector.
 * 
 * @param vector The vector to be printed
 * @param size The size of the vector
 */
void showVector(float *vector, int size) {
    for (int i = 0; i < size; i++) {
        printf("V[%d] = %f\n", i, vector[i]);
    }
}

/**
 * @brief Main function
 * 
 * This is the main function of the program.
 * 
 * @return 0
 */
int main() {
    // Set the size of the vectors
    int size = 1 << 20; // 2^20 = 1048576 elements
    float *vectorA, *vectorB, *vectorC;
    struct timeval start, end;

    // Allocate memory for the vectors
    vectorA = (float *)malloc(size * sizeof(float));
    vectorB = (float *)malloc(size * sizeof(float));
    vectorC = (float *)malloc(size * sizeof(float));

    // Initialize the vectors
    initVector(vectorA, size);
    initVector(vectorB, size);

    // Start the timer
    gettimeofday(&start, NULL);

    // Perform vector addition
    vectorialSum(vectorA, vectorB, vectorC, size);

    // Stop the timer
    gettimeofday(&end, NULL);

    // Calculate and print the time taken
    long timeTaken = (end.tv_sec - start.tv_sec) * 1000000 + end.tv_usec - start.tv_usec;
    printf("Time: %ld microseconds\n", timeTaken);

    // Optional: print the vectors
    // showVector(vectorA, size);
    // showVector(vectorB, size);
    // showVector(vectorC, size);

    // Free allocated memory
    free(vectorA);
    free(vectorB);
    free(vectorC);

    return 0;
}
