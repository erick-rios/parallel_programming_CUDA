/**
 * @file swapArrays.cu
 * @brief Swap elements of two arrays using CUDA
 * 
 * This code shows how to swap the elements of two arrays using CUDA. The kernel
 * `swapArrays` is defined to swap the elements of two arrays. The main function
 * initializes two arrays on the host, allocates memory on the device, transfers
 * the data from the host to the device, calls the kernel, and transfers the data
 * back to the host to validate the results.
 * 
 * @author Erick Jesús Ríos González
 * @date 28/0/2024
 */
#include <iostream>
#include <cuda_runtime.h>

/**
 * @brief Kernel to swap the elements of two arrays
 * 
 * @param array1 Pointer to the first array
 * @param array2 Pointer to the second array
 * @param n Size of the arrays
 * 
 * @note This kernel assumes that the size of the arrays is a multiple of the block size
 */
__global__ void swapArrays(int *array1, int *array2, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        // Swap elements
        int temp = array1[idx];
        array1[idx] = array2[idx];
        array2[idx] = temp;
    }
}

/**
 * @brief Main function
 * 
 * @return 0
 * 
 * @note This function initializes two arrays on the host, allocates memory on the device,
 * transfers the data from the host to the device, calls the kernel, and transfers the data
 * back to the host to validate the results.
 */
int main() {
    // Initialize arrays
    int n = 100;  // Size of the arrays
    int *h_array1 = (int*)malloc(n * sizeof(int));
    int *h_array2 = (int*)malloc(n * sizeof(int));

    // Fill arrays with values from 0 to n-1
    for (int i = 0; i < n; i++) {
        h_array1[i] = i;
        h_array2[i] = n - i;
    }

    // Allocate memory on the device
    int *d_array1, *d_array2;
    cudaMalloc((void**)&d_array1, n * sizeof(int));
    cudaMalloc((void**)&d_array2, n * sizeof(int));

    // Transfer data from host to device
    cudaMemcpy(d_array1, h_array1, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_array2, h_array2, n * sizeof(int), cudaMemcpyHostToDevice);

    // Set the block size and the number of blocks
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Call the kernel
    swapArrays<<<numBlocks, blockSize>>>(d_array1, d_array2, n);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Transfer data from device to host
    cudaMemcpy(h_array1, d_array1, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_array2, d_array2, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Validate the results
    std::cout << "Arreglo 1 después del intercambio:" << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << h_array1[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Arreglo 2 después del intercambio:" << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << h_array2[i] << " ";
    }
    std::cout << std::endl;

    // Free memory
    cudaFree(d_array1);
    cudaFree(d_array2);
    free(h_array1);
    free(h_array2);

    // Fin
    return 0;
}
