/**
 * @file matrixMultiplication.c
 * @brief This file contains a simple example of matrix multiplication in C.
 * 
 * This program initializes two matrices A and B with some values, performs matrix multiplication, and prints the result.
 * 
 * @author ERICK JESUS RIOS GONZALEZ
 * @date 03/09/2024
 * @version 1.0
 */

#include <stdio.h>
#include <stdlib.h>


/**
 * @brief Function to initialize two matrices with some values
 * 
 * This function initializes two matrices A and B with some values.
 * 
 * @param A The first matrix
 * @param B The second matrix  
 * @param rowsA The number of rows of matrix A
 * @param colsA The number of columns of matrix A
 * @param rowsB The number of rows of matrix B
 * @param colsB The number of columns of matrix B
 */
void initialize_matrices(int *A, int *B, int rowsA, int colsA, int rowsB, int colsB) {
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsA; j++) {
            A[i * colsA + j] = i + j; // Example initialization
        }
    }

    for (int i = 0; i < rowsB; i++) {
        for (int j = 0; j < colsB; j++) {
            B[i * colsB + j] = i - j; // Example initialization
        }
    }
}

/**
 * @brief Function to perform matrix multiplication
 * 
 * This function performs matrix multiplication of two matrices A and B and stores the result in matrix C.
 * 
 * @param A The first matrix
 * @param B The second matrix
 * @param C The resulting matrix
 * @param rowsA The number of rows of matrix A
 * @param colsA The number of columns of matrix A
 * @param colsB The number of columns of matrix B
 */
void matrix_multiply(int *A, int *B, int *C, int rowsA, int colsA, int colsB) {
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            C[i * colsB + j] = 0;
            for (int k = 0; k < colsA; k++) {
                C[i * colsB + j] += A[i * colsA + k] * B[k * colsB + j];
            }
        }
    }
}

/**
 * @brief Function to print a matrix
 * 
 * This function prints the elements of a matrix.
 * 
 * @param M The matrix to be printed
 * @param rows The number of rows of the matrix
 * @param cols The number of columns of the matrix
 */
void print_matrix(int *M, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", M[i * cols + j]);
        }
        printf("\n");
    }
}
/**
 * Main function
 * 
 * This function initializes two matrices A and B, performs matrix multiplication, and prints the result.
 * 
 * @return 0
 */
int main() {
    // Define matrix dimensions
    int rowsA = 3, colsA = 3;
    int rowsB = 3, colsB = 3;

    // Allocate memory for matrices A, B, and C
    int *A = (int *)malloc(rowsA * colsA * sizeof(int));
    int *B = (int *)malloc(rowsB * colsB * sizeof(int));
    int *C = (int *)malloc(rowsA * colsB * sizeof(int)); // Result matrix

    if (A == NULL || B == NULL || C == NULL) {
        printf("Error allocating memory!\n");
        return 1;
    }

    // Initialize matrices A and B
    initialize_matrices(A, B, rowsA, colsA, rowsB, colsB);

    // Perform matrix multiplication
    matrix_multiply(A, B, C, rowsA, colsA, colsB);

    // Print the result
    printf("Matrix A:\n");
    print_matrix(A, rowsA, colsA);
    printf("\nMatrix B:\n");
    print_matrix(B, rowsB, colsB);
    printf("\nMatrix C (Result):\n");
    print_matrix(C, rowsA, colsB);

    // Free allocated memory
    free(A);
    free(B);
    free(C);

    return 0;
}
