/**
 * @file matrixMultiplication.cu
 * @brief CUDA program to multiply two matrices on the GPU.
 * 
 * This program multiplies two matrices on the GPU using CUDA. The program takes the dimensions of the matrices as input and returns the result of the multiplication.
 * 
 * @author ERICK JESUS RIOS GONZALEZ
 * @date 010/09/2024
 * @version 1.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>

// Dimension of the matrices
#define N 1024

/**
 * @brief Function to initialize the matrices A and B with random values.
 * 
 * This function initializes the matrices A and B with random values.
 */
void init_matrices(float *A, float *B, int n) {
    for (int i = 0; i < n * n; i++) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }
}

/**
 * @brief Kernel function to multiply two matrices.
 * 
 * This kernel function multiplies two matrices A and B and stores the result in matrix C.
 * 
 * @param A Pointer to the first matrix.
 * @param B Pointer to the second matrix.
 * @param C Pointer to the resulting matrix.
 * @param n Dimension of the matrices.
 */
__global__ void matMulKernel(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0;
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

/**
 * @brief Function to verify the result of the matrix multiplication.
 * 
 * This function verifies the result of the matrix multiplication by comparing it with the result obtained using the CPU.
 * 
 * @param A Pointer to the first matrix.
 * @param B Pointer to the second matrix.
 * @param C Pointer to the resulting matrix.
 * @param n Dimension of the matrices.
 */
void verify_result(float *A, float *B, float *C, int n) {
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            float sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A[row * n + k] * B[k * n + col];
            }
            if (fabs(C[row * n + col] - sum) > 1e-5) {
                printf("Error en la posición (%d, %d)\n", row, col);
                return;
            }
        }
    }
    printf("Resultados correctos\n");
}

/**
 * @brief Main function.
 * 
 * This is the main function that initializes the matrices, multiplies them on the GPU, and verifies the result.
 */
int main() {
    float *A, *B, *C;                // Host matrices
    float *d_A, *d_B, *d_C;          // Device matrices

    // Assign memory in the host
    A = (float*)malloc(N * N * sizeof(float));
    B = (float*)malloc(N * N * sizeof(float));
    C = (float*)malloc(N * N * sizeof(float));

    // Initialize matrices
    init_matrices(A, B, N);

    // Assign memory in the device
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));

    // Transfer data from host to device
    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define the number of threads per block and the number of blocks
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Measure the time of the kernel
    auto start = std::chrono::high_resolution_clock::now();

    // Launch the kernel
    matMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Synchronize threads
    cudaDeviceSynchronize();

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    printf("Tiempo de ejecución en GPU: %ld ms\n", duration.count());

    // Transfer data from device to host
    cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify the result
    verify_result(A, B, C, N);

    // Free memory in the host
    free(A);
    free(B);
    free(C);

    // Free memory in the device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
