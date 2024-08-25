# Parallel Programming in CUDA - Course Repository

Welcome to the official repository for the **Parallel Programming in CUDA** course. This course is designed to teach you the fundamentals and advanced concepts of parallel programming using NVIDIA's CUDA platform. Through this repository, you will find all the necessary materials, assignments, and project files to excel in the course.

## Project Structure

Below is the directory structure of this repository:

```plaintext
├── README.md                   # Course overview and repository guide
├── docs                        # Documentation and course resources
│   ├── syllabus.pdf            # Course syllabus
│   ├── assignment_guidelines.md # Guidelines for assignments
│   └── references.md           # Recommended reading and references
├── src                         # Source code for examples and assignments
│   ├── vector_addition         # Example project for vector addition in CUDA
│   │   ├── vector_addition.cu  # CUDA implementation
│   │   ├── Makefile            # Build script for vector addition
│   │   └── README.md           # Detailed explanation of the vector addition project
│   ├── matrix_multiplication   # Example project for matrix multiplication in CUDA
│   │   ├── matrix_multiplication.cu # CUDA implementation
│   │   ├── Makefile            # Build script for matrix multiplication
│   │   └── README.md           # Detailed explanation of the matrix multiplication project
│   └── ...
├── tests                       # Unit tests and performance benchmarks
│   ├── test_vector_addition.cu # Test cases for vector addition
│   ├── test_matrix_multiplication.cu # Test cases for matrix multiplication
│   └── ...
└── resources                   # Additional resources and datasets
    ├── datasets                # Example datasets for use in projects
    │   └── example_dataset.csv # Sample dataset
    └── slides                  # Lecture slides
        ├── lecture1.pptx       # Introduction to CUDA
        ├── lecture2.pptx       # Memory management in CUDA
        └── ...
