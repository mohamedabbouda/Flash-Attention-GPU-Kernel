# Flash-Attention-GPU-Kernel

An implementation **Flash Attention** using **Triton** and **CUDA**.



---

## Overview

Flash Attention is an optimized attention algorithm designed to make Transformer attention faster and more memory-efficient on GPUs.

Standard attention materializes a large attention matrix, which can become expensive for long sequences. Flash Attention improves this by using GPU-aware tiling and reducing memory reads and writes between high-bandwidth memory and on-chip SRAM.

---

## Repository Structure

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
```

---

## Main Components

### Triton Implementation

The `triton/` directory contains the main Flash Attention implementation.

| File | Description |
|---|---|
| `flash_attention.py` | Triton-based Flash Attention implementation and comparison with naive attention |
| `requirements.txt` | Python dependencies required to run the Triton implementation |


---

## Installation

Clone the repository:

```bash
git clone https://github.com/mohamedabbouda/Flash-Attention-GPU-Kernel.git
cd Flash-Attention-GPU-Kernel
```

Install the Python dependencies:

```bash
pip install -r triton/requirements.txt
```

---

## Running the Triton Flash Attention Code

```bash
cd triton
python flash_attention.py
```

You may need to reduce these values inside `flash_attention.py` depending on your GPU memory:

```python
BATCH_SIZE
NUM_HEADS
SEQ_LEN
HEAD_DIM
```

Large sequence lengths can require a lot of GPU memory, especially when comparing Flash Attention with a naive attention implementation.

---



---





## References

- FlashAttention paper: https://arxiv.org/abs/2205.14135
- FlashAttention-2 paper: https://arxiv.org/abs/2307.08691
