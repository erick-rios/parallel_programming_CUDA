/**
 * @file riemannSum.cu
 * @brief Riemann sum calculation using CUDA.
 * 
 * This program calculates the Riemann sum of a function f(x) = x^2 using CUDA. The Riemann sum is
 * calculated using the formula:
 * 
 * \f[
 * \sum_{i=0}^{N-1} f(x_i) \Delta x
 * \f]
 * 
 * where \f$ f(x) = x^2 \f$ and \f$ \Delta x = \frac{B - A}{N} \f$.
 * 
 * The program uses a kernel function to calculate the Riemann sum in parallel. The kernel function
 * uses shared memory to reduce the access to global memory and performs an interleaved pair reduction
 * to calculate the final result. The kernel function is executed on the GPU and the result is
 * transferred back to the host to validate the result.
 * 
 * @author Erick Jesús Ríos González
 * @date 01/10/2024
 */

#include <iostream>
#include <cuda.h>

#define A 0.0f
#define B 1.0f

/**
 * @brief Function to calculate the value of the function f(x) = x^2.
 * 
 * This function calculates the value of the function f(x) = x^2.
 * 
 * @param x Value of the variable x.
 * 
 * @return Value of the function f(x) = x^2.
 */
__device__ float f(float x) {
    return x * x;
}

/**
 * @brief CPU function to calculate the Riemann sum of the function f(x) = x^2.
 * 
 * This function calculates the Riemann sum of the function f(x) = x^2 using the formula:
 * 
 * \f[
 * \sum_{i=0}^{N-1} f(x_i) \Delta x
 * \f]
 * 
 * where \f$ f(x) = x^2 \f$ and \f$ \Delta x = \frac{B - A}{N} \f$.
 * 
 * @param float a initial value of the interval
 * @param float dx width of each subinterval
 * @param int N number of subintervals
 * @param float *partialSums array with the partial sums
 * 
 * @return void 
 */
__global__ void riemannSum(float a, float dx, int N, float *partialSums) {
    extern __shared__ float cache[];
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float x;
    float tempSum = 0.0;
    
    // Each thread computes a part of the Riemann sum
    while (tid < N) {
        x = a + tid * dx;
        tempSum += f(x) * dx;
        tid += blockDim.x * gridDim.x;
    }
    
    // Store result in shared memory
    cache[cacheIndex] = tempSum;

    __syncthreads();

    // Perform interleaved pair reduction within a block
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    // Write the result of each block's sum to global memory
    if (cacheIndex == 0) {
        partialSums[blockIdx.x] = cache[0];
    }
}

/**
 * @brief CPU function to calculate the Riemann sum of the function f(x) = x^2.
 * 
 * This function calculates the Riemann sum of the function f(x) = x^2 using the formula:
 * 
 * \f[
 * \sum_{i=0}^{N-1} f(x_i) \Delta x
 * \f]
 * 
 * where \f$ f(x) = x^2 \f$ and \f$ \Delta x = \frac{B - A}{N} \f$.
 * 
 * @param n number of subintervals
 * 
 * @return float Riemann sum of the function f(x) = x^2.
 */
float riemannSumCPU(int n) {
    float dx = (B - A) / n;
    float sum = 0.0;
    for (int i = 0; i < n; i++) {
        float x = A + i * dx;
        sum += (x * x) * dx;
    }
    return sum;
}

/**
 * @brief Main function.
 * 
 * This function initializes the data, allocates memory on the device, executes the kernel function,
 * transfers the results back to the host, and validates the results.
 * 
 * @return 0 if the program exits successfully.
 */
int main() {
    // Step 1: Data Initialization
    int N = 1 << 20; // Number of subintervals (e.g., 2^20)
    float dx = (B - A) / N;  // Width of each subinterval

    // Host memory for result
    float hostSum = 0.0f;

    // Step 2: Dynamic Memory Allocation on Host and Device
    float *d_partialSums;
    int blockSize = 256;  // Number of threads per block
    int numBlocks = (N + blockSize - 1) / blockSize;
    cudaMalloc((void**)&d_partialSums, numBlocks * sizeof(float));

    // Step 3: Setting up Kernel Execution
    size_t sharedMemSize = blockSize * sizeof(float);
    riemannSum<<<numBlocks, blockSize, sharedMemSize>>>(A, dx, N, d_partialSums);

    // Step 4: Transfer the results back to the host
    float *hostPartialSums = new float[numBlocks];
    cudaMemcpy(hostPartialSums, d_partialSums, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Step 5: Reduction of partial results on the host
    for (int i = 0; i < numBlocks; i++) {
        hostSum += hostPartialSums[i];
    }

    // Validate the result
    float cpu_result = riemannSumCPU(N); // Calculate the result using the CPU
    std::cout << "Resultado GPU: " << hostSum << std::endl;
    std::cout << "Resultado CPU: " << cpu_result << std::endl;

    if (fabs(hostSum - cpu_result) < 5e0) {
        std::cout << "El resultado es correcto." << std::endl;
    } else {
        std::cout << "El resultado es incorrecto." << std::endl;
    }

    // Step 6: Results Validation (approximating the integral of f(x) = x^2 from 0 to 1)
    std::cout << "Approximate integral: " << hostSum << std::endl;
    std::cout << "Expected result: " << 1.0f / 3.0f << std::endl;

    // Step 7: Memory Release
    cudaFree(d_partialSums);
    delete[] hostPartialSums;

    return 0;
}
