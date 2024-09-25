/**
 * @file divergencia.cu
 * 
 * @brief Compare the execution time of two kernels with and without divergence
 * 
 * This code compares the execution time of two kernels: one with divergence and
 * another without divergence. The kernel `kernelWithDivergence` has a condition
 * that produces divergence, while the kernel `kernelWithoutDivergence` uses a
 * mask to avoid divergence. The main function initializes an array on the host,
 * allocates memory on the device, and compares the execution times of both kernels.
 * 
 * @author Erick Jesús Ríos González
 * @date 24/09/2024
 */

#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

/**
 * @brief Kernel with divergence
 * 
 * @param array Pointer to the array
 * @param n Size of the array
 * 
 * @note This kernel has a condition that produces divergence
 */
__global__ void kernelWithDivergence(int *array, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        // Condition that produces divergence
        if (array[idx] % 2 == 0) {
            array[idx] *= 2;  // Operation for even numbers
        } else {
            array[idx] *= 3;  // Operation for odd numbers
        }
    }
}

/**
 * @brief Kernel without divergence
 * 
 * @param array Pointer to the array
 * @param n Size of the array
 * 
 * @note This kernel uses a mask to avoid divergence
 */
__global__ void kernelWithoutDivergence(int *array, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        // Operation using a mask to avoid divergence
        int is_even = (array[idx] % 2 == 0);
        array[idx] = is_even * array[idx] * 2 + (!is_even) * array[idx] * 3;
    }
}

/**
 * @brief Fill an array with random numbers
 * 
 * @param array Pointer to the array
 * @param n Size of the array
 * 
 * @note This function fills the array with random numbers
 */
void fillArray(int *array, int n) {
    for (int i = 0; i < n; i++) {
        array[i] = rand() % 100;  // Fill with random numbers
    }
}

/**
 * @brief Compare the execution time of two kernels with and without divergence
 * 
 * @param h_array Pointer to the array on the host
 * @param n Size of the array
 * @param blockSize Block size for the kernels
 * 
 * @note This function allocates memory on the device, transfers the data from the host
 * to the device, calls the kernels, and compares the execution times of both kernels
 * using the high_resolution_clock from the chrono library
 */
void compareKernels(int *h_array, int n, int blockSize) {
    int *d_array;
    cudaMalloc(&d_array, n * sizeof(int));

    // Number of required blocks
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Measure execution time of the kernel with divergence
    cudaMemcpy(d_array, h_array, n * sizeof(int), cudaMemcpyHostToDevice);
    auto start = std::chrono::high_resolution_clock::now();
    kernelWithDivergence<<<numBlocks, blockSize>>>(d_array, n);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diffWithDivergence = end - start;
    std::cout << "Time with divergence: " << diffWithDivergence.count() << " seconds" << std::endl;

    // Measure execution time of the kernel without divergence
    cudaMemcpy(d_array, h_array, n * sizeof(int), cudaMemcpyHostToDevice);
    start = std::chrono::high_resolution_clock::now();
    kernelWithoutDivergence<<<numBlocks, blockSize>>>(d_array, n);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diffWithoutDivergence = end - start;
    std::cout << "Time without divergence: " << diffWithoutDivergence.count() << " seconds" << std::endl;

    // Free memory
    cudaFree(d_array);
}

/**
 * @brief Main function
 * 
 * @return 0
 */
int main() {
    int n = 1 << 20;  // 1 million elements
    int blockSize = 256;
    
    int *h_array = (int*)malloc(n * sizeof(int));

    // Initialize the array
    fillArray(h_array, n);

    // Compare execution times
    compareKernels(h_array, n, blockSize);

    // Free memory on the host
    free(h_array);

    return 0;
}

