# Flash-Attention-GPU-Kernel
# Flash-Attention-GPU-Kernel

A learning project for implementing and understanding **Flash Attention** using GPU programming tools such as **Triton** and **CUDA**.

This repository is based on an open tutorial/project by `hkproj`, which implements Flash Attention 2 in Triton and includes CUDA examples.

Original source: https://github.com/hkproj/triton-flash-attention

## Overview

Flash Attention is an optimized attention algorithm designed to make Transformer attention faster and more memory-efficient on GPUs.

Standard attention can be expensive because it materializes large attention matrices, where memory usage grows quadratically with sequence length. Flash Attention improves this by using GPU-aware tiling and reducing memory reads and writes between high-bandwidth memory and on-chip SRAM.

## Project Structure

```text
Flash-Attention-GPU-Kernel/
├── triton/
│   ├── flash_attention.py
│   └── requirements.txt
├── cuda/
│   ├── Makefile
│   ├── cuda_common.cuh
│   ├── matrix_add.cu
│   ├── vector_add.cu
│   └── vector_add_simple.cu
├── .gitignore
└── README.md
Contents
triton/

Contains the main Flash Attention implementation using Triton.

flash_attention.py
Triton-based Flash Attention implementation and comparison with a naive attention implementation.
requirements.txt
Python dependencies needed to run the Triton implementation.
cuda/

Contains basic CUDA examples used for learning GPU programming fundamentals.

vector_add_simple.cu
vector_add.cu
matrix_add.cu
cuda_common.cuh
Makefile

These files are useful for practicing CUDA compilation, kernels, memory allocation, and simple GPU computations.

Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/Flash-Attention-GPU-Kernel.git
cd Flash-Attention-GPU-Kernel

Install the Python dependencies:

pip install -r triton/requirements.txt
Running the Triton Flash Attention Code
cd triton
python flash_attention.py

You may need to reduce the values of:

BATCH_SIZE
NUM_HEADS
SEQ_LEN
HEAD_DIM

Large sequence lengths can require a lot of GPU memory, especially when comparing against the naive attention implementation.

Running the CUDA Examples
cd cuda
make
make run

To clean the generated output files:

make clean
Learning Goals

The goal of this project is to understand:

GPU programming basics
CUDA kernels
Triton kernels
Transformer attention
Flash Attention memory optimization
Differences between naive attention and optimized attention
Attribution

This project is based on the open tutorial repository:

Original repo: https://github.com/hkproj/triton-flash-attention
Instructor/author: hkproj / Umar Jamil

This repository is used for educational purposes and personal learning.

References
FlashAttention paper: https://arxiv.org/abs/2205.14135
FlashAttention-2 paper: https://arxiv.org/abs/2307.08691
Original tutorial repo: https://github.com/hkproj/triton-flash-attention

Important: replace this part with your actual GitHub username:

```bash
git clone https://github.com/YOUR_USERNAME/Flash-Attention-GPU-Kernel.git