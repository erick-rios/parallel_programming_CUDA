/**
 * @file caesarAlgorithm.cu
 * @brief CUDA program to implement Caesar's Algorithm to cipher a message on the GPU.
 * 
 * This program implements Caesar's Algorithm to cipher a message on the GPU. The program takes a message as input and returns the ciphered message.
 * 
 * @author ERICK JESUS RIOS GONZALEZ
 * @date 25/08/2024
 * @version 1.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define MAX_BLOCK_SIZE 1024

/**
 * @brief Function to perform Caesar's Algorithm on the GPU.
 * 
 * This function performs Caesar's Algorithm on the GPU.
 * 
 * @param messageToCipher Pointer to the message to be ciphered.
 * @param cipheredMessage Pointer to the resulting ciphered message.
 * @param size Size of the message.
 * @param shift Shift to be applied to the message.
 * 
 * @see https://en.wikipedia.org/wiki/Caesar_cipher
 */
__global__ void caesarAlgorithm(char *messageToCipher, char *cipheredMessage, int size, int shift) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        char c = messageToCipher[i];
        if (c >= 'a' && c <= 'z') {
            cipheredMessage[i] = (c - 'a' + shift) % 26 + 'a';
        } else if (c >= 'A' && c <= 'Z') {
            cipheredMessage[i] = (c - 'A' + shift) % 26 + 'A';
        } else {
            cipheredMessage[i] = c; // Non-alphabetic characters remain unchanged
        }
    }
}

/**
 * @brief Main function.
 * 
 * This function is the entry point of the program.
 * 
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments.
 * @return 0 if the program exits successfully, 1 otherwise.
 */
int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <message> <shift>\n", argv[0]);
        return 1;
    }

    char *message = argv[1];
    int shift = atoi(argv[2]);
    int size = strlen(message);

    // Calculate block and grid sizes
    int blockSize = (size > MAX_BLOCK_SIZE) ? MAX_BLOCK_SIZE : size;
    int gridSize = (size + blockSize - 1) / blockSize;

    // Allocate memory on the CPU
    char *cipheredMessage = (char *)malloc((size + 1) * sizeof(char));

    // Allocate memory on the GPU
    char *d_message, *d_cipheredMessage;
    cudaMalloc((void **)&d_message, size * sizeof(char));
    cudaMalloc((void **)&d_cipheredMessage, size * sizeof(char));

    // Copy the message to the GPU
    cudaMemcpy(d_message, message, size * sizeof(char), cudaMemcpyHostToDevice);

    // Launch the kernel
    caesarAlgorithm<<<gridSize, blockSize>>>(d_message, d_cipheredMessage, size, shift);

    // Copy the ciphered message back to the CPU
    cudaMemcpy(cipheredMessage, d_cipheredMessage, size * sizeof(char), cudaMemcpyDeviceToHost);

    // Null-terminate the ciphered message
    cipheredMessage[size] = '\0';

    // Print the original and ciphered message
    printf("Message: %s\n", message);
    printf("Ciphered message: %s\n", cipheredMessage);

    // Free memory on the CPU and GPU
    free(cipheredMessage);
    cudaFree(d_message);
    cudaFree(d_cipheredMessage);

    return 0;
}
