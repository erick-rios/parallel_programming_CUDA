/**
 * @file transposeMatrix.cu
 * 
 * @brief Calculate the transpose of a matrix in the GPU
 * 
 * This code calculates the transpose of a matrix using a kernel executed in the GPU.
 * The main function initializes a matrix on the host, allocates memory on the device,
 * and compares the execution time of the kernel with the CPU version.
 * 
 * @author Erick Jesús Ríos González
 * @date 29/09/2024
 */

#include <iostream>
#include <cuda.h>
#include <chrono>

#define N 1024 // Size of the matrix

/**
 * Kernel to calculate the transpose of a matrix
 * 
 * @param d_out Pointer to the output matrix
 * @param d_in Pointer to the input matrix
 * @param width Width of the matrix
 * 
 * @return None
 * 
 * @note The kernel is executed in the GPU
 */
__global__ void transposeKernel(float* d_out, float* d_in, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < width) {
        d_out[y * width + x] = d_in[x * width + y];
    }
}

/**
 * Function to calculate the transpose of a matrix in the CPU
 * 
 * @param out Pointer to the output matrix
 * @param in Pointer to the input matrix
 * @param width Width of the matrix
 * 
 * @return None
 * 
 * @note The function is executed in the CPU
 */
void transposeCPU(float* out, float* in, int width) {
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            out[j * width + i] = in[i * width + j];
        }
    }
}

/**
 * Main function
 * 
 * @return 0
 */
int main() {
    int size = N * N * sizeof(float);

    // Initialize matrices on the host
    float* h_in = new float[N * N];
    float* h_out_cpu = new float[N * N];
    float* h_out_gpu = new float[N * N];

    // Fill the input matrix with random numbers
    for (int i = 0; i < N * N; ++i) {
        h_in[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate memory on the device
    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    // Transfer data to the device
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // Define the number of threads per block and blocks per grid
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Compute the transpose in the GPU and measure the time
    auto start_gpu = std::chrono::high_resolution_clock::now();
    transposeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_in, N);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> gpu_duration = end_gpu - start_gpu;

    // Transfer data to the host
    cudaMemcpy(h_out_gpu, d_out, size, cudaMemcpyDeviceToHost);

    // Compute the transpose in the CPU and measure the time
    auto start_cpu = std::chrono::high_resolution_clock::now();
    transposeCPU(h_out_cpu, h_in, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = end_cpu - start_cpu;

    // Validate the results
    bool correct = true;
    for (int i = 0; i < N * N; ++i) {
        if (fabs(h_out_gpu[i] - h_out_cpu[i]) > 1e-5) {
            correct = false;
            break;
        }
    }

    if (correct) {
        std::cout << "Resultado correcto.\n";
    } else {
        std::cout << "Error en la transpuesta.\n";
    }

    // Speed-up calculation
    float speedup = cpu_duration.count() / gpu_duration.count();
    std::cout << "Tiempo en GPU: " << gpu_duration.count() << " ms\n";
    std::cout << "Tiempo en CPU: " << cpu_duration.count() << " ms\n";
    std::cout << "Speed-up (CPU/GPU): " << speedup << "\n";

    // Free memory
    delete[] h_in;
    delete[] h_out_cpu;
    delete[] h_out_gpu;
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
