/**
 * @file matrix_multiplication.cu
 * @brief Matrix multiplication using CUDA with tiling.
 * 
 * This program performs matrix multiplication using CUDA with tiling. The program generates two random matrices
 * and multiplies them using a kernel function that is executed on the GPU. The program also performs the matrix
 * multiplication on the CPU to compare the results and measure the execution time. The program validates the results
 * and calculates the speed-up obtained by using the GPU.
 * 
 * @author Erick Jesús Ríos González    
 * @date 19/11/2024
 */

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cstdlib>

#define TILE_WIDTH 16  // Define the tile size

/**
 * @brief Kernel function to perform matrix multiplication with tiling.
 * 
 * This kernel function performs matrix multiplication with tiling. The matrices are divided into tiles of size TILE_WIDTH
 * to take advantage of shared memory and improve memory access patterns. The kernel is executed on the GPU.
 * 
 * @param d_M Input matrix M.
 * @param d_N Input matrix N.
 * @param d_P Output matrix P.
 * @param Width Width of the matrices.
 * 
 * @return void
 */
__global__ void MatrixMulKernel(float* d_M, float* d_N, float* d_P, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;

    for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {
        Mds[ty][tx] = d_M[Row * Width + ph * TILE_WIDTH + tx];
        Nds[ty][tx] = d_N[(ph * TILE_WIDTH + ty) * Width + Col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    d_P[Row * Width + Col] = Pvalue;
}

/**
 * @brief Function to initialize a matrix with random values.
 * 
 * This function initializes a matrix with random values between 0 and 1.
 * 
 * @param matrix Pointer to the matrix.
 * @param size Size of the matrix.
 * 
 * @return void
 */
void initializeMatrix(float* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

/**
 * @brief Function to perform matrix multiplication on the CPU.
 * 
 * This function performs matrix multiplication on the CPU to validate the results obtained from the GPU.
 * 
 * @param M Input matrix M.
 * @param N Input matrix N.
 * @param P Output matrix P.
 * @param Width Width of the matrices.
 * 
 * @return void
 */
void cpuMatrixMultiply(float* M, float* N, float* P, int Width) {
    for (int i = 0; i < Width; ++i) {
        for (int j = 0; j < Width; ++j) {
            float value = 0;
            for (int k = 0; k < Width; ++k) {
                value += M[i * Width + k] * N[k * Width + j];
            }
            P[i * Width + j] = value;
        }
    }
}

/**
 * @brief Main function.
 * 
 * This function generates two random matrices, performs matrix multiplication using CUDA with tiling, and compares
 * the results obtained on the CPU and GPU. The program measures the execution time on both devices and calculates
 * the speed-up obtained by using the GPU.
 * 
 * @return 0
 */
int main() {
    const int Width = 128;  // Define matrix size (Width x Width)
    const int matrixSize = Width * Width;
    const int matrixBytes = matrixSize * sizeof(float);

    // Host memory allocation
    float *h_M = new float[matrixSize];
    float *h_N = new float[matrixSize];
    float *h_P = new float[matrixSize];
    float *h_P_GPU = new float[matrixSize];

    // Initialize matrices
    initializeMatrix(h_M, matrixSize);
    initializeMatrix(h_N, matrixSize);

    // Device memory allocation
    float *d_M, *d_N, *d_P;
    cudaMalloc(&d_M, matrixBytes);
    cudaMalloc(&d_N, matrixBytes);
    cudaMalloc(&d_P, matrixBytes);

    // Copy matrices to device
    cudaMemcpy(d_M, h_M, matrixBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, matrixBytes, cudaMemcpyHostToDevice);

    // Kernel configuration
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(Width / TILE_WIDTH, Width / TILE_WIDTH);

    // Launch kernel and measure GPU execution time
    auto startGPU = std::chrono::high_resolution_clock::now();
    MatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, Width);
    cudaDeviceSynchronize();
    auto endGPU = std::chrono::high_resolution_clock::now();

    // Copy result back to host
    cudaMemcpy(h_P_GPU, d_P, matrixBytes, cudaMemcpyDeviceToHost);

    // Measure CPU execution time
    auto startCPU = std::chrono::high_resolution_clock::now();
    cpuMatrixMultiply(h_M, h_N, h_P, Width);
    auto endCPU = std::chrono::high_resolution_clock::now();

    // Validate results
    bool valid = true;
    for (int i = 0; i < matrixSize; ++i) {
        if (abs(h_P[i] - h_P_GPU[i]) > 1e-5) {
            valid = false;
            break;
        }
    }
    std::cout << "Validation: " << (valid ? "PASSED" : "FAILED") << "\n";

    // Calculate and print execution times
    auto gpuTime = std::chrono::duration_cast<std::chrono::milliseconds>(endGPU - startGPU).count();
    auto cpuTime = std::chrono::duration_cast<std::chrono::milliseconds>(endCPU - startCPU).count();
    std::cout << "GPU Time: " << gpuTime << " ms\n";
    std::cout << "CPU Time: " << cpuTime << " ms\n";
    std::cout << "Speed-up: " << static_cast<float>(cpuTime) / gpuTime << "\n";

    // Free memory
    delete[] h_M;
    delete[] h_N;
    delete[] h_P;
    delete[] h_P_GPU;
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    return 0;
}
