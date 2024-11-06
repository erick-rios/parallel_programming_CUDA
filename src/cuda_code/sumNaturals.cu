/**
 * @file sumNaturals.cu
 * @brief Sum of the first N natural numbers using parallel reduction
 * @details This program calculates the sum of the first N natural numbers using a parallel reduction algorithm.
 * 
 * The program initializes an array of size N with the first N natural numbers, then it calculates the sum of the array
 * using a parallel reduction algorithm. The algorithm uses a two-step reduction in shared memory to calculate the sum
 * of the array. The program uses a block size of 512 threads and a grid size of (N + 1023) / 1024 blocks.
 * 
 * The program validates the result by comparing it with the expected result of the sum of the first N natural numbers.
 * 
 * The program uses CUDA to perform the parallel reduction on the GPU.
 * 
 * The program can be compiled with the following command:
 * 
 * nvcc sumNaturals.cu -o sumNaturals
 * 
 * The program can be executed with the following command:
 * 
 * ./sumNaturals
 * 
 * @author Erick Jesús Ríos González
 * @date 05/11/2024
 */

#include <iostream>
#include <cuda.h>
#include <chrono>

// Size of the array
#define N 1024

/**
 * @brief Parallel reduction algorithm to calculate the sum of an array
 * @details This function calculates the sum of an array using a parallel reduction algorithm.
 * 
 * The function uses a two-step reduction in shared memory to calculate the sum of the array. The function uses a block
 * size of 512 threads and a grid size of (N + 1023) / 1024 blocks.
 * 
 * @param input Array of integers to calculate the sum
 * @param result Pointer to store the result of the sum
 * @param n Size of the array
 * 
 */
__global__ void sumReduction(int *input, int *result, int n) {
    extern __shared__ int sdata[];
    
    // Index of the current thread
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    // Load data into shared memory and perform the first step of the reduction
    sdata[tid] = (i < n) ? input[i] : 0;
    if (i + blockDim.x < n) sdata[tid] += input[i + blockDim.x];
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the result of the block to global memory
    if (tid == 0) result[blockIdx.x] = sdata[0];
}

/**
 * @brief CPU function to calculate the sum of an array
 * @details This function calculates the sum of an array using the CPU.
 * 
 * The function calculates the sum of the array using a for loop to iterate over the elements of the array and add them
 * to the sum.
 * 
 * @param data Array of integers to calculate the sum
 * @param n Size of the array
 */
int cpuSum(int *data, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

/**
 * @brief Main function
 * @details This function initializes the data on the host, allocates memory on the device, transfers the data from the
 * host to the device, configures the kernel and calculates the reduction, reduces the partial results, validates the
 * result and frees the memory.
 * 
 * @return 0
 */

int main() {
    // Initialize data on the host
    int *host_data = new int[N];
    for (int i = 0; i < N; i++) {
        host_data[i] = i + 1;
    }

    // Result variables
    int host_result = 0;
    int cpu_result = 0;

    // Allocate memory on the device
    int *device_data, *device_partial_sums;
    cudaMalloc((void**)&device_data, N * sizeof(int));
    cudaMalloc((void**)&device_partial_sums, (N / 1024) * sizeof(int));

    // Transfer data from the host to the device
    cudaMemcpy(device_data, host_data, N * sizeof(int), cudaMemcpyHostToDevice);

    // Measure time on the GPU
    auto gpu_start = std::chrono::high_resolution_clock::now();
    
    // Set up the kernel configuration
    int threads = 512;
    int blocks = (N + threads * 2 - 1) / (threads * 2);
    sumReduction<<<blocks, threads, threads * sizeof(int)>>>(device_data, device_partial_sums, N);

    // Reduce the partial sums
    int *host_partial_sums = new int[blocks];
    cudaMemcpy(host_partial_sums, device_partial_sums, blocks * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < blocks; i++) {
        host_result += host_partial_sums[i];
    }

    auto gpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_duration = gpu_end - gpu_start;

    // Measure time on the CPU
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_result = cpuSum(host_data, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;

    // Validate the result
    int expected_result = (N * (N + 1)) / 2;
    if (host_result == expected_result && cpu_result == expected_result) {
        std::cout << "Resultado correcto: " << host_result << std::endl;
    } else {
        std::cout << "Resultado incorrecto: GPU: " << host_result 
                  << ", CPU: " << cpu_result 
                  << ", esperado: " << expected_result << std::endl;
    }

    // Print the time taken by the GPU and the CPU
    std::cout << "Tiempo en GPU: " << gpu_duration.count() << " ms" << std::endl;
    std::cout << "Tiempo en CPU: " << cpu_duration.count() << " ms" << std::endl;
    std::cout << "Speed-up: " << cpu_duration.count() / gpu_duration.count() << std::endl;

    // Free memory
    delete[] host_data;
    delete[] host_partial_sums;
    cudaFree(device_data);
    cudaFree(device_partial_sums);

    return 0;
}
