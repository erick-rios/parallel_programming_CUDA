/**
 * @file convolution2D.cu
 * @brief 2D convolution using CUDA.
 * 
 * This program applies a 2D convolution to an image using CUDA. The program loads an image in grayscale
 * and applies a 3x3 kernel to perform the convolution. The convolution is performed using a kernel function
 * that is executed on the GPU. The program compares the results obtained on the CPU and GPU to validate
 * the correctness of the implementation.
 * 
 * @author Erick Jesús Ríos González
 * @date 19/11/2024
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// Size of the block used for the kernel execution
#define BLOCK_SIZE 16

using namespace cv;
using namespace std;

/**
 * @brief Convolution kernel executed on the GPU.
 * 
 * This kernel function performs a 2D convolution on an image using a given convolution kernel.
 * The kernel is applied to each pixel in the image, and the result is stored in the output image.
 * 
 * @param d_input Input image.
 * @param d_output Output image.
 * @param d_kernel Convolution kernel.
 * @param rows Number of rows in the image.
 * @param cols Number of columns in the image.
 * @param ksize Size of the convolution kernel.
 * 
 * @return void
 */
__global__ void convolution2D(float *d_input, float *d_output, float *d_kernel, int rows, int cols, int ksize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int half_k = ksize / 2;

    if (x >= cols || y >= rows) return;

    float sum = 0.0;

    for (int i = -half_k; i <= half_k; i++) {
        for (int j = -half_k; j <= half_k; j++) {
            int r = min(max(y + i, 0), rows - 1);
            int c = min(max(x + j, 0), cols - 1);
            sum += d_input[r * cols + c] * d_kernel[(i + half_k) * ksize + (j + half_k)];
        }
    }

    d_output[y * cols + x] = sum;
}

/**
 * @brief Check for CUDA errors.
 * 
 * This function checks for CUDA errors and prints an error message if an error is found.
 * 
 * @param msg Message to print in case of an error.
 * @return void
 */
void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "CUDA error (" << msg << "): " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Main function.
 * 
 * This function loads an image in grayscale, applies a 3x3 convolution kernel to the image using CUDA,
 * and compares the results obtained on the CPU and GPU. The results are saved to disk and the maximum
 * difference between the results is printed to the console.
 * 
 * @return int Program exit status.
 */
int main() {
    // Set CUDA device to use for execution
    Mat image = imread("../../images/output_gray.jpg", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Error al cargar la imagen." << endl;
        return -1;
    }

    int rows = image.rows;
    int cols = image.cols;

    // Convert image to float
    Mat imageFloat;
    image.convertTo(imageFloat, CV_32F);

    // Define kernel
    int ksize = 3;
    float h_kernel[] = {
         0, -1,  0,
        -1,  4, -1,
         0, -1,  0
    };

    // Create output images
    Mat outputCPU(rows, cols, CV_32F);
    Mat outputGPU(rows, cols, CV_32F);

    // Allocate memory on device
    float *d_input, *d_output, *d_kernel;
    size_t imageSize = rows * cols * sizeof(float);
    size_t kernelSize = ksize * ksize * sizeof(float);

    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);
    cudaMalloc(&d_kernel, kernelSize);

    // Transfer data to device
    cudaMemcpy(d_input, imageFloat.ptr<float>(), imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelSize, cudaMemcpyHostToDevice);

    // Setup kernel execution configuration
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Execute kernel
    convolution2D<<<gridSize, blockSize>>>(d_input, d_output, d_kernel, rows, cols, ksize);
    checkCUDAError("Kernel execution");

    // Transfer data back to host
    cudaMemcpy(outputGPU.ptr<float>(), d_output, imageSize, cudaMemcpyDeviceToHost);

    // Validation using OpenCV
    Mat kernel = Mat(ksize, ksize, CV_32F, h_kernel);
    filter2D(imageFloat, outputCPU, -1, kernel, Point(-1, -1), 0, BORDER_CONSTANT);

    // Compare results
    Mat diff;
    absdiff(outputCPU, outputGPU, diff);
    double maxDiff;
    minMaxLoc(diff, nullptr, &maxDiff);
    cout << "Máxima diferencia entre CPU y GPU: " << maxDiff << endl;

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    // Show results
    imwrite("../../images/output_cpu.jpg", outputCPU);
    imwrite("../../images/output_gpu.jpg", outputGPU);

    cout << "Resultados guardados: output_cpu.jpg y output_gpu.jpg" << endl;

    return 0;
}
