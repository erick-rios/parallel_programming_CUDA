/**
 * @file imageConverter.cu
 * @brief CUDA program to convert an RGB image to grayscale.
 * 
 * This program converts an RGB image to grayscale using CUDA. The program takes an input image and returns the grayscale image.
 * 
 * @author ERICK JESUS RIOS GONZALEZ
 * @date 10/09/2024
 * @version 1.0
 */

#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <iostream>

using namespace cv;

/**
 * @brief Kernel function to convert an RGB image to grayscale.
 * 
 * This kernel function converts an RGB image to grayscale.
 * 
 * @param d_in Pointer to the input image.
 * @param d_out Pointer to the output image.
 * @param width Width of the image.
 * @param height Height of the image.
 * @param channels Number of channels in the image.
 */
__global__ void rgb2grayKernel(unsigned char *d_in, unsigned char *d_out, int width, int height, int channels) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int idx = (row * width + col) * channels;

        // Extraer los componentes RGB del píxel
        unsigned char r = d_in[idx];
        unsigned char g = d_in[idx + 1];
        unsigned char b = d_in[idx + 2];

        // Convertir a escala de grises
        d_out[row * width + col] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
    }
}

/**
 * @brief Main function to convert an RGB image to grayscale.
 * 
 * This function loads an RGB image, converts it to grayscale using CUDA, and displays the original and grayscale images.
 * 
 * @return 0 if successful.
 */
int main() {
    // Load the input image
    Mat img = imread("../../images/5c05de636f596cb157698cde7923ce19e8473211228abb1cea24a12baaaa8074.jpg", IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "No se puede cargar la imagen!" << std::endl;
        return -1;
    }

    // Get the dimensions of the image
    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();

    // Create a grayscale image
    Mat gray_img(height, width, CV_8UC1);

    // Get the pointers to the input and output images
    unsigned char *h_in = img.data;  // Input array for the RGB image
    unsigned char *h_out = gray_img.data;  // Output array for the grayscale image

    // Memory allocation on the device
    unsigned char *d_in, *d_out;
    size_t img_size = width * height * channels * sizeof(unsigned char);
    size_t gray_img_size = width * height * sizeof(unsigned char);

    // Memory allocation on the device
    cudaMalloc((void**)&d_in, img_size);
    cudaMalloc((void**)&d_out, gray_img_size);

    // Transfer the input image to the device
    cudaMemcpy(d_in, h_in, img_size, cudaMemcpyHostToDevice);

    // Configure the kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    rgb2grayKernel<<<numBlocks, threadsPerBlock>>>(d_in, d_out, width, height, channels);

    // Synchronize threads
    cudaDeviceSynchronize();

    // Transfer the output image to the host
    cudaMemcpy(h_out, d_out, gray_img_size, cudaMemcpyDeviceToHost);

    // Display the images
    imshow("Imagen en color", img);
    imshow("Imagen en escala de grises", gray_img);

    // Save the grayscale image
    imwrite("../../images/output_gray.jpg", gray_img);

    // Free memory on the device
    cudaFree(d_in);
    cudaFree(d_out);

    // Wait for a key press
    waitKey(0);

    return 0;
}
