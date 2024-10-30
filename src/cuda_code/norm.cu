/**
 * @file norm.cu
 * 
 * @brief Calculate the norm and dot product of two vectors in the GPU
 * 
 * This code calculates the norm and dot product of two vectors using kernels executed in the GPU.
 * The main function initializes two vectors on the host, allocates memory on the device,
 * and compares the execution time of the kernels with the CPU version.
 * 
 * @author Erick Jesús Ríos González
 * @date 29/09/2024
 */
#include <iostream>
#include <cuda.h>
#include <cmath>
#include <chrono>

#define N 10 // Size of the vectors
#define THREADS_PER_BLOCK 32

/**
 * Kernel to calculate the norm of a vector
 * 
 * @param d_out Pointer to the output vector
 * @param d_in Pointer to the input vector
 * @param n Size of the vector
 * 
 * @return None
 */
__global__ void vectorNormKernel(float* d_out, const float* d_in, int n) {
    __shared__ float shared_data[THREADS_PER_BLOCK];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;

    // Each thread calculates the square of the corresponding element
    shared_data[local_tid] = (tid < n) ? d_in[tid] * d_in[tid] : 0.0f;
    __syncthreads();

    // Reduction in the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (local_tid < stride) {
            shared_data[local_tid] += shared_data[local_tid + stride];
        }
        __syncthreads();
    }

    // The first thread of each block saves the partial result
    if (local_tid == 0) {
        d_out[blockIdx.x] = shared_data[0];
    }
}

/**
 * Kernel to calculate the dot product of two vectors
 * 
 * @param d_out Pointer to the output vector
 * @param d_a Pointer to the first input vector
 * @param d_b Pointer to the second input vector
 * @param n Size of the vectors
 * 
 * @return None
 * 
 */
__global__ void dotProductKernel(float* d_out, const float* d_a, const float* d_b, int n) {
    __shared__ float shared_data[THREADS_PER_BLOCK];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;

    // Each thread calculates the product of the corresponding elements
    shared_data[local_tid] = (tid < n) ? d_a[tid] * d_b[tid] : 0.0f;
    __syncthreads();

    // Reduction in the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (local_tid < stride) {
            shared_data[local_tid] += shared_data[local_tid + stride];
        }
        __syncthreads();
    }

    // The first thread of each block saves the partial result
    if (local_tid == 0) {
        d_out[blockIdx.x] = shared_data[0];
    }
}

/**
 * Function to reduce an array of floats in the CPU
 * 
 * @param partials Pointer to the array of floats
 * @param n Size of the array
 * 
 * @return Sum of the elements in the array
 * 
 */
float reduceOnCPU(float* partials, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += partials[i];
    }
    return sum;
}

/**
 * Main function
 * 
 * @return 0
 */
int main() {
    int size = N * sizeof(float);

    // Initialize vectors on the host
    float* h_a = new float[N];
    float* h_b = new float[N];

    // Fill the input vectors with random numbers
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(rand()) / RAND_MAX;
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate memory on the device
    float *d_a, *d_b, *d_out_norm, *d_out_dot;
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_out_norm, blocks * sizeof(float));
    cudaMalloc(&d_out_dot, blocks * sizeof(float));

    // Transfer data to the device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Setup and execution of the kernel to calculate the norm in the GPU
    auto start_gpu = std::chrono::high_resolution_clock::now();
    vectorNormKernel<<<blocks, THREADS_PER_BLOCK>>>(d_out_norm, d_a, N);
    cudaDeviceSynchronize();

    // Transfer data to the host
    float* h_partial_norm = new float[blocks];
    cudaMemcpy(h_partial_norm, d_out_norm, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    float norm_a = sqrt(reduceOnCPU(h_partial_norm, blocks));

    // Setup and execution of the kernel to calculate the dot product in the GPU
    dotProductKernel<<<blocks, THREADS_PER_BLOCK>>>(d_out_dot, d_a, d_b, N);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> gpu_duration = end_gpu - start_gpu;

    // Transfer data to the host
    float* h_partial_dot = new float[blocks];
    cudaMemcpy(h_partial_dot, d_out_dot, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    float dot_product = reduceOnCPU(h_partial_dot, blocks);

    // Compute the norm and dot product in the CPU
    auto start_cpu = std::chrono::high_resolution_clock::now();
    float norm_a_cpu = 0.0f, dot_product_cpu = 0.0f, norm_b_cpu = 0.0f;
    for (int i = 0; i < N; ++i) {
        norm_a_cpu += h_a[i] * h_a[i];
        norm_b_cpu += h_b[i] * h_b[i];
        dot_product_cpu += h_a[i] * h_b[i];
    }
    norm_a_cpu = sqrt(norm_a_cpu);
    norm_b_cpu = sqrt(norm_b_cpu);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = end_cpu - start_cpu;

    // Validate the results and Cauchy-Schwarz inequality
    bool correct = fabs(dot_product - dot_product_cpu) < 1e-5;
    std::cout << "Norma GPU: " << norm_a << "\n";
    std::cout << "Producto punto GPU: " << dot_product << "\n";
    std::cout << "Norma CPU: " << norm_a_cpu << "\n";
    std::cout << "Producto punto CPU: " << dot_product_cpu << "\n";
    std::cout << "Verificación de Cauchy-Schwarz: " 
              << (dot_product * dot_product <= norm_a * norm_a * norm_b_cpu * norm_b_cpu ? "Cumple\n" : "No cumple\n");
    std::cout << "Resultado correcto: " << (correct ? "Sí\n" : "No\n");

    // Calculate speed-up
    float speedup = cpu_duration.count() / gpu_duration.count();
    std::cout << "Tiempo en GPU: " << gpu_duration.count() << " ms\n";
    std::cout << "Tiempo en CPU: " << cpu_duration.count() << " ms\n";
    std::cout << "Speed-up (CPU/GPU): " << speedup << "\n";

    // Free memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_partial_norm;
    delete[] h_partial_dot;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out_norm);
    cudaFree(d_out_dot);

    return 0;
}
