/**
 * @file scalarMultiplication.cu
 * @brief CUDA program to perfom scalar multiplication on the GPU.
 * 
 * This program performs scalar multiplication on the GPU. The program takes two vectors as input and returns the sum of the two vectors.
 * Initialize the vectors with random values and print the result.
 * 
 * @author ERICK JESUS RIOS GONZALEZ
 * @date 25/08/2024
 * @version 1.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define BLOCK_SIZE 2048
#define SCALAR atoi(argv[1])


/**
 * @brief Function to initialize the vector with random values.
 * 
 * This function initializes the vector with random values between 1 and 100
 * 
 * @param vectorToInit Pointer to the vector to be initialized.
 * @param size Size of the vector.  
 */

void initVector(float *vectorToInit, int size) {
    for (int i = 0; i < size; i++) {
        vectorToInit[i] = (float)rand() / RAND_MAX;
    }
}

/**
 * @brief Function to show the vector.
 * 
 * This function prints the vector.
 * 
 * @param vectorToPrint Pointer to the vector to be printed.
 * @param size Size of the vector.  
 */
void showVector(float *vectorToPrint, int size) {
    for (int i = 0; i < size; i++) {
        printf("V[%d] = %f\n", i, vectorToPrint[i]);
    }
    printf("\n");
}

/**
 * @brief Function to perform scalar multiplication on the GPU.
 * This function performs scalar multiplication on the GPU.
 * @param vectorA Pointer to the first vector.
 * @param scalar Scalar to multiply the vector.
 * @param vectorC Pointer to the resulting vector.
 * @param size Size of the vectors. 
 */
__global__ void scalarMultiplication(float *vectorA, float scalar, float *vectorC, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        vectorC[index] = scalar * vectorA[index];
    }
}

/**
 * @brief Main function.
 * 
 * This function initializes the vectors, allocates memory on the GPU, copies the vectors to the GPU, performs the scalar multiplication, copies the result to the CPU, and prints the result.
 * @return 0 if the program ends correctly.
 */

int main(int argc, char *argv[]){

    if (argc < 2) {
    printf("Usage: %s <scalar_value>\n", argv[0]);
    exit(1);
    }

    int size = 1 << 20;

    // Allocate memory on the CPU, define pointers to the vectors, and initialize the vectors.
    float *vectorA, *vectorC;
    // Allocate memory on the GPU, define pointers to the vectors.
    float *d_vectorA, *d_vectorC;
    struct timeval start, end;

    // Allocate memory on the CPU.
    vectorA = (float *)malloc(size * sizeof(float));
    vectorC = (float *)malloc(size * sizeof(float));

    // Initialize the vectors.
    initVector(vectorA, size);

    // Allocate memory on the GPU.
    cudaMalloc(&d_vectorA, size * sizeof(float));
    cudaMalloc(&d_vectorC, size * sizeof(float));

    // Initialize the timer.
    gettimeofday(&start, NULL);

    // Copy the vectors to the GPU.
    cudaMemcpy(d_vectorA, vectorA, size * sizeof(float), cudaMemcpyHostToDevice);

    // Perform the scalar multiplication.
    scalarMultiplication<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_vectorA, SCALAR, d_vectorC, size);

    // Copy the result to the CPU.
    cudaMemcpy(vectorC, d_vectorC, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Stop the timer.
    gettimeofday(&end, NULL);
    
    // Print the time.
    printf("Time: %ld microseconds\n", (end.tv_sec - start.tv_sec) * 1000000 + end.tv_usec - start.tv_usec);

    // Print the result.
    //showVector(vectorC, size);
    //showVector(vectorA, size);
    //showVector(vectorB, size);

    // Free memory on the CPU
    free(vectorA);
    free(vectorC);

    // Free memory on the GPU
    cudaFree(d_vectorA);
    cudaFree(d_vectorC);

    return 0;
}