/**
 * @file matrixSumColumns.cu
 * @brief CUDA program to sum the elements of each column of a matrix.
 * 
 * This program performs the sum of the elements of each column of a matrix. The program takes a matrix as input and returns the sum of the elements of each column.
 * 
 * @author ERICK JESUS RIOS GONZALEZ
 * @date 03/09/2024
 * @version 1.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 2048

/**
 * @brief Function to perform vectorial addition on the GPU.
 * 
 * This function performs vectorial addition on the GPU.
 * 
 * @param vectorA Pointer to the first vector.
 * @param vectorB Pointer to the second vector.
 * @param vectorC Pointer to the resulting vector.
 * @param size Size of the vectors.
 */
__global__ void vectorialSum(float *vectorA, float *vectorB, float *vectorC, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        vectorC[index] = vectorA[index] + vectorB[index];
    }
}

/**
 * @brief Function to perform vectorized matrix addition on the GPU.
 * 
 * This function performs vectorized matrix addition on the GPU.
 * 
 * @param matrixA Pointer to the first matrix.
 * @param matrixB Pointer to the second matrix.
 * @param resultC Pointer to the resulting matrix.
 * @param rows Number of rows of the matrix.
 * @param cols Number of columns of the matrix.
 * 
 * @note The number of rows and columns of the matrices must be the same.
 */
void vectorizedMatrixSum(float *matrixA, float *matrixB, float *resultC, int rows, int cols) {
    int size = rows * cols;
    
    float *d_matrixA, *d_matrixB, *d_matrixC;
    cudaError_t err;
    
    // Allocate memory on the GPU
    err = cudaMalloc((void**)&d_matrixA, size * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    
    err = cudaMalloc((void**)&d_matrixB, size * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        cudaFree(d_matrixA);
        exit(-1);
    }

    err = cudaMalloc((void**)&d_matrixC, size * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        cudaFree(d_matrixA);
        cudaFree(d_matrixB);
        exit(-1);
    }

    // Copy matrices to the GPU
    cudaMemcpy(d_matrixA, matrixA, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixB, matrixB, size * sizeof(float), cudaMemcpyHostToDevice);

    // Perform vector addition on GPU
    vectorialSum<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_matrixA, d_matrixB, d_matrixC, size);
    cudaDeviceSynchronize(); // Ensure the kernel has completed

    // Copy the result back to the host
    cudaMemcpy(resultC, d_matrixC, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_matrixA);
    cudaFree(d_matrixB);
    cudaFree(d_matrixC);
}

/**
 * @brief Function to initialize a matrix with random values.
 * 
 * This function initializes a matrix with random values.
 * 
 * @param matrix Pointer to the matrix.
 * @param rows Number of rows of the matrix.
 * @param cols Number of columns of the matrix.
 * 
 * @note The matrix must be allocated before calling this function.
 */
void initMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = (float)rand() / RAND_MAX;
        }
    }
}

/**
 * @brief Function to sum the elements of each column of a matrix.
 * 
 * This function sums the elements of each column of a matrix.
 * 
 * @param matrix Pointer to the matrix.
 * @param result Pointer to the resulting array.
 * @param rows Number of rows of the matrix.
 * @param cols Number of columns of the matrix.
 * 
 * @note The number of rows of the matrix must be the same as the size of the resulting array.
 */
void sumByColumns(float *matrix, float *result, int rows, int cols) {
    for (int j = 0; j < cols; j++) {
        result[j] = 0;
        for (int i = 0; i < rows; i++) {
            result[j] += matrix[i * cols + j];
        }
    }
}

/**
 * @brief Function to perform traditional matrix addition.
 * 
 * This function performs traditional matrix addition.
 * 
 * @param A Pointer to the first matrix.
 * @param B Pointer to the second matrix.
 * @param C Pointer to the resulting matrix.
 * @param rows Number of rows of the matrices.
 * @param cols Number of columns of the matrices.
 */
void traditionalMatrixSum(float *A, float *B, float *C, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            C[i * cols + j] = A[i * cols + j] + B[i * cols + j];
        }
    }
}

/**
 * @brief Main function.
 * 
 * This function initializes the matrices, performs the sum of the elements of each column of a matrix, performs the traditional matrix sum, and performs the vectorized matrix sum on the GPU.
 * 
 * @return 0 if the program ends correctly.
 */
int main() {
    int rows = 1024, cols = 1024;
    struct timeval start, end;

    // Allocate memory for the matrix and result array
    float *matrixA = (float *)malloc(rows * cols * sizeof(float));
    float *matrixB = (float *)malloc(rows * cols * sizeof(float));
    float *resultSum = (float *)malloc(cols * sizeof(float));
    float *resultC = (float *)malloc(rows * cols * sizeof(float)); // For matrix sum

    if (matrixA == NULL || matrixB == NULL || resultSum == NULL || resultC == NULL) {
        printf("Memory allocation failed!\n");
        exit(-1);
    }

    // Initialize the matrices
    initMatrix(matrixA, rows, cols);
    initMatrix(matrixB, rows, cols);

    // Sum by columns
    gettimeofday(&start, NULL);
    sumByColumns(matrixA, resultSum, rows, cols);
    gettimeofday(&end, NULL);
    printf("Time for sum by columns: %ld microseconds\n", (end.tv_sec - start.tv_sec) * 1000000 + end.tv_usec - start.tv_usec);

    // Traditional matrix sum
    gettimeofday(&start, NULL);
    traditionalMatrixSum(matrixA, matrixB, resultC, rows, cols);
    gettimeofday(&end, NULL);
    printf("Time for traditional matrix sum: %ld microseconds\n", (end.tv_sec - start.tv_sec) * 1000000 + end.tv_usec - start.tv_usec);

    // Perform vectorized matrix sum on GPU
    gettimeofday(&start, NULL);
    vectorizedMatrixSum(matrixA, matrixB, resultC, rows, cols);
    gettimeofday(&end, NULL);
    printf("Time for vectorized matrix sum on GPU: %ld microseconds\n", (end.tv_sec - start.tv_sec) * 1000000 + end.tv_usec - start.tv_usec);

    // Free allocated memory
    free(matrixA);
    free(matrixB);
    free(resultSum);
    free(resultC);

    return 0;
}
