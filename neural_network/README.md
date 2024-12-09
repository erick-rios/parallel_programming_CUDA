# Neural Network with CUDA

This project demonstrates the implementation of a neural network using CUDA for parallel programming. The goal is to leverage the power of GPU computing to accelerate the training and inference processes of neural networks.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project showcases how to use CUDA to implement a neural network. CUDA is a parallel computing platform and application programming interface (API) model created by Nvidia. It allows developers to use Nvidia GPUs for general-purpose processing.

## Features
- Parallelized neural network training
- Efficient matrix operations using CUDA
- Example implementations of common neural network layers

## Installation
To get started with this project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/erick-rios/parallel_programming_CUDA
    ```
2. Navigate to the project directory:
    ```sh
    cd neural_network_cuda
    ```
3. Install the required dependencies:
    ```sh
    # Example for a Python project
    pip install -r requirements.txt
    ```

## Usage
To run the neural network training, use the following command:
```sh
python train.py
```
Make sure you have a compatible Nvidia GPU and CUDA installed on your system.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.