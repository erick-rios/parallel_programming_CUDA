/**
 * @file dotProduct.cu
 * @brief Dot product of two vectors using CUDA.
 * 
 * This program calculates the dot product of two vectors using CUDA. The dot product is calculated
 * using a kernel function that uses shared memory to reduce the access to global memory. The kernel
 * function is executed on the GPU and the result is transferred back to the host to validate the
 * result.
 * 
 * The program initializes two vectors with random values and calculates the dot product using the
 * CPU. The result is then calculated using the GPU and compared with the CPU result to validate the
 * correctness of the GPU implementation.
 * 
 * The program uses a block size of 256 threads and the number of blocks is calculated based on the
 * size of the vectors. The kernel function calculates the dot product of the vectors and stores the
 * result in a shared memory array. The shared memory array is then reduced to calculate the final
 * result, which is stored in the global memory.
 * 
 * The program prints the result calculated using the GPU and the expected result calculated using
 * the CPU. It then compares the two results and prints whether the result is correct or incorrect.
 * 
 * @author Erick Jesús Ríos González
 * @date 01/10/2024
 */

#include <iostream>
#include <cuda_runtime.h>

#define N 1024  // Size of the vectors

/**
 * @brief Kernel function to calculate the dot product of two vectors.
 * 
 * This kernel function calculates the dot product of two vectors using shared memory to reduce the
 * access to global memory. The kernel function is executed on the GPU and the result is stored in
 * the global memory.
 * 
 * @param a Pointer to the first vector.
 * @param b Pointer to the second vector.
 * @param c Pointer to the result of the dot product.
 * @param n Size of the vectors.
 */
__global__ void dotProductKernel(int *a, int *b, int *c, int n) {
    __shared__ int cache[256];  // Memoria compartida para reducir el acceso a memoria global
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    int temp = 0;
    while (tid < n) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    // Store the result in shared memory
    cache[cacheIndex] = temp;

    __syncthreads(); // Thread synchronization

    // Reduction in shared memory
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    // The first thread stores the result in global memory
    if (cacheIndex == 0) {
        atomicAdd(c, cache[0]);
    }
}

/**
 * @brief Function to initialize an array with random values.
 * 
 * This function initializes an array with random values between 0 and 99.
 * 
 * @param arr Pointer to the array.
 * @param size Size of the array.
 */
void initializeArray(int *arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 100;  // Values between 0 and 99
    }
}

/**
 * @brief Main function.
 * 
 * This function initializes two vectors with random values and calculates the dot product using the
 * CPU. The result is then calculated using the GPU and compared with the CPU result to validate the
 * correctness of the GPU implementation.
 * 
 * @return 0 if the program exits successfully.
 */
int main() {
    // Initialize random seed
    int *h_a, *h_b, h_c;
    int *d_a, *d_b, *d_c;
    int size = N * sizeof(int);

    // Assign memory in the Host
    h_a = (int *)malloc(size);
    h_b = (int *)malloc(size);

    // initialize arrays with random values
    initializeArray(h_a, N);
    initializeArray(h_b, N);
    h_c = 0;

    // Assign memory in the Device (GPU)
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, sizeof(int));

    // Transfer data from Host to Device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, &h_c, sizeof(int), cudaMemcpyHostToDevice);

    // Set the block size and number of blocks
    int blockSize = 256;  // Size of the block
    int numBlocks = (N + blockSize - 1) / blockSize;  // Number of blocks

    // executes the kernel function
    dotProductKernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);

    // Transfer data from Device to Host
    cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

    // Validate the result
    int expected = 0;
    for (int i = 0; i < N; i++) {
        expected += h_a[i] * h_b[i];
    }

    // Print the result
    std::cout << "Resultado GPU: " << h_c << std::endl;
    std::cout << "Resultado esperado (CPU): " << expected << std::endl;

    if (h_c == expected) {
        std::cout << "El resultado es correcto." << std::endl;
    } else {
        std::cout << "El resultado es incorrecto." << std::endl;
    }

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);

    return 0;
}
