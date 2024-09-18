#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

#define BLUR_SIZE 10  // Tamaño del filtro de desenfoque

__global__ void blurKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        for (int channel = 0; channel < 3; channel++) {
            int pixelValue = 0;
            int pixels = 0;

            for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; blurRow++) {
                for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; blurCol++) {
                    int currentRow = row + blurRow;
                    int currentCol = col + blurCol;

                    if (currentRow > -1 && currentRow < height && currentCol > -1 && currentCol < width) {
                        int offset = (currentRow * width + currentCol) * 3 + channel;
                        pixelValue += input[offset];
                        pixels++;
                    }
                }
            }

            int outputOffset = (row * width + col) * 3 + channel;
            output[outputOffset] = (unsigned char)(pixelValue / pixels);
        }
    }
}

int main() {
    Mat image = imread("../../images/5c05de636f596cb157698cde7923ce19e8473211228abb1cea24a12baaaa8074.jpg", IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Error: No se pudo cargar la imagen" << endl;
        return -1;
    }

    int width = image.cols;
    int height = image.rows;

    unsigned char* hostInputImage = image.data;
    unsigned char* hostOutputImage = (unsigned char*)malloc(width * height * 3 * sizeof(unsigned char));

    unsigned char *deviceInputImage, *deviceOutputImage;
    cudaMalloc((void**)&deviceInputImage, width * height * 3 * sizeof(unsigned char));
    cudaMalloc((void**)&deviceOutputImage, width * height * 3 * sizeof(unsigned char));

    cudaMemcpy(deviceInputImage, hostInputImage, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    blurKernel<<<gridSize, blockSize>>>(deviceInputImage, deviceOutputImage, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(hostOutputImage, deviceOutputImage, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    Mat outputImage(height, width, CV_8UC3, hostOutputImage);

    imshow("Original Image", image);
    imshow("Blurred Image", outputImage);
    waitKey(0);

    free(hostOutputImage);
    cudaFree(deviceInputImage);
    cudaFree(deviceOutputImage);

    return 0;
}
