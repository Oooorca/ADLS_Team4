# ADLS Labs Report

## Lab 1 - Task 1
### Accuracy vs Fixed Point
![Lab1 T1 Accuracy vs Fixed Point](https://github.com/Oooorca/ADLS_zy1324/blob/main/lab1_t1_accuracy_vs_fixed_point_width.png?raw=true)

## Lab 1 - Task 2
### Pruning Accuracy vs Sparsity
![Lab1 T2 Pruning Accuracy vs Sparsity](https://github.com/Oooorca/ADLS_zy1324/blob/main/lab1_t2_pruning_accuracy_vs_sparsity.png?raw=true)

## Lab 2 - Task 1
### NAS Comparison
![Lab2 T1 NAS Comparison](https://github.com/Oooorca/ADLS_zy1324/blob/main/lab2_t1_nas_comparison.png?raw=true)

## Lab 2 - Task 2
### Compression NAS Comparison
![Lab2 T2 Compression NAS Comparison](https://github.com/Oooorca/ADLS_zy1324/blob/main/lab2_t2_compression_nas_comparison.png?raw=true)

## Lab 3 - Task 1
### Accuracy vs Trials (Quantization Search)
![Lab3 T1 Accuracy vs Trials](https://github.com/Oooorca/ADLS_zy1324/blob/main/lab3_t1_accuracy_vs_trials.png?raw=true)

## Lab 3 - Task 2
### Accuracy vs Trials (Precision Search)
![Lab3 T2 Accuracy vs Trials](https://github.com/Oooorca/ADLS_zy1324/blob/main/lab3_t2_accuracy_vs_trials.png?raw=true)

## **Lab 4 - Performance Optimization and Custom Kernels**

### **Task 1: Investigating `torch.compile` Performance**
#### **(a) Modify the code and investigate why we did not observe real run-time speedups with `torch.compile`**
**Observations:**
- On **CPU**, we observed that the compiled model was initially **much slower** than the original model when no warm-up was used. This was due to **compilation overhead and initial weight loading**.
- **Using warm-up (`warmup=1`) significantly improved the compiled model's performance**, as it allowed the system to stabilize before measuring inference times.
- After increasing `n` (e.g., `n=20`), the compiled model showed a **clear acceleration**, demonstrating that the **initial compilation overhead was amortized** over multiple runs.
- **Larger batch sizes (`batch_size=128`) also improved performance**, as it helped the compiled model take better advantage of kernel fusion and efficient memory usage.
- **Subsequent runs benefit from caching**, reducing overhead from **weight loading, initialization, and compilation**, leading to more consistent and improved execution times.
- The **compiled model consistently outperforms the original model when warm-up is applied**, showing that `torch.compile` is effective when properly benchmarked.



**Experiment Results (CPU):**
| Model | Batch Size | Iterations (`n`) | Warm-up | Average Inference Time (s) |
|--------|------------|----------------|---------|----------------------|
| **Original Model** | 64 | 5 | 0 | 4.6558 s |
| **Compiled Model** | 64 | 5 | 0 | 13.045 s |
| **Original Model** | 64 | 5 | 1 | 4.5771 s |
| **Compiled Model** | 64 | 5 | 1 | 3.0778 s |
| **Original Model** | 64 | 20 | 1 | 4.7771 s |
| **Compiled Model** | 64 | 20 | 1 | 3.0158 s |
| **Original Model** | 128 | 20 | 1 | 9.5405 s |
| **Compiled Model** | 128 | 20 | 1 | 6.1587 s |

#### **(b) If you change the `device` to `cuda`, do you observe the same thing?**
- **On GPU**, the compiled model **did show improvements**, but the relative speedup was smaller than on CPU.  
- This is likely because PyTorch already optimizes tensor computations well on GPUs.
- The compilation overhead was still present but was less significant compared to the actual execution time.

**Experiment Results (GPU, Tesla T4):**
| Model | Batch Size | Iterations (`n`) | Average Inference Time (s) | Speedup |
|--------|------------|----------------|----------------------|----------|
| **Original Model** | 128 | 10 | 0.1028 s | 1.00x |
| **Compiled Model** | 128 | 10 | 0.0887 s | 1.15x |
| **Original Model** | 128 | 100 | 0.1063 s | 1.00x |
| **Compiled Model** | 128 | 100 | 0.0933 s | 1.14x |
| **Original Model** | 512 | 100 | 0.4321 s | 1.00x |
| **Compiled Model** | 512 | 100 | 0.3745 s | 1.15x |

**Key Takeaways:**
- **On CPU**, `torch.compile` becomes beneficial with larger `n` and `batch_size`.  
- **On GPU**, the improvement exists but is less pronounced due to PyTorch's already optimized CUDA kernels.  

---

### **Task 2: Profiling the SDPA Kernel (Kernel Fusion)**
#### **(a) Extend the profiling to compare the naive SDPA kernel with the fused implementation**
- The fused SDPA kernel in PyTorch 2.0+ reduces the number of memory accesses and kernel launches by merging the `QK^T`, `softmax`, and `attn @ V` computations into **one fused operation**.
- We profiled the naive and fused implementations on **CPU and GPU**.

**Experiment Results (CPU):**
| Model | Average Inference Time (s) | Speedup |
|--------|----------------------|----------|
| **Naive SDPA** | 1.5709 s | 1.00x |
| **Fused SDPA** | 0.0359 s | **43.72x** |

**Experiment Results (GPU, Tesla T4):**
| Model | Average Inference Time (s) | Speedup |
|--------|----------------------|----------|
| **Naive SDPA** | 0.000511 s | 1.00x |
| **Fused SDPA** | 0.000272 s | **1.88x** |

#### **(b) If you change the `device` to `cuda`, do you observe the same thing?**
- Yes, on GPU, the **fused SDPA kernel** is **still faster** than the naive implementation.
- However, the absolute execution time for both naive and fused versions is **already very low** on GPU, making the relative speedup smaller compared to CPU.

**Key Takeaways:**
- **On CPU**, SDPA fusion provides a **huge performance boost** (~43x speedup).
- **On GPU**, the benefit is **smaller** (~1.88x speedup) since GPUs are already optimized for matrix multiplications.
- Kernel fusion helps to **reduce kernel launch overhead**, especially in memory-bound operations.

---

### **Task 3: Implementing and Profiling MXINT8 Custom Kernel**
#### **(a) How does MXINT8 benefit custom hardware if both the activation and weights in a linear layer are quantized to MXINT8?**
- **Reduced memory footprint**: Since MXINT8 uses **only 8 bits** instead of FP32 (**32 bits**), it reduces the storage requirements by **4x**.
- **Faster computation**: Many AI accelerators (NVIDIA Tensor Cores, Google TPUs) have **specialized INT8 execution units** that can process **4 times more operations per cycle** compared to FP32.
- **Lower power consumption**: INT8 operations consume **less power** than FP32, making them **ideal for mobile and edge AI applications**.

#### **(b) What is the purpose of the variables `dont_need_abs` and `bias` in the C++ kernel?**
- **`dont_need_abs`**: Determines whether an absolute value computation is required during **dequantization**. If `true`, the kernel **skips unnecessary absolute calculations**, improving efficiency.
- **`bias`**: Used to **adjust rounding errors** when converting from **quantized values back to floating-point values**. Ensures better numerical accuracy.

#### **(c) How does `cta_tiler` partition data for copying to shared memory in the CUDA kernel? How does `layout_sX` partition threads in a thread block?**
- **`cta_tiler` (CTA-based tiling strategy)**:
  - **Breaks large data chunks into smaller tiles** and **loads them into shared memory**.
  - Each **CUDA thread block (CTA)** processes a **subset** of the data to maximize memory locality.
  - **Reduces global memory accesses**, leading to better efficiency.

- **`layout_sX` (Thread mapping strategy)**:
  - Defines how threads **map to different sections** of a matrix computation.
  - Determines whether threads operate on **rows, columns, or blocks** to optimize parallel processing.
  - Optimized for **warp-level execution** (32 threads per warp).

#### **(d) Why the saved GPU memory is not exactly (32 - (4+8/32))/32 = 86.7% of the FP32 model?**
- **Theoretical Calculation**:  
  \[
  \left(32 - \frac{4+8}{32} \right) / 32 = 86.7\% \text{ memory savings}
  \]
- **Experimental Results (GPU, Tesla T4):**
  | Model | Peak Memory Usage (MB) | Saved Memory |
  |--------|----------------------|----------------|
  | **FP32 Model** | 2906.2 MB | - |
  | **MXINT8 Model** | 976.2 MB | **66.4% (not 86.7%)** |

**Why actual savings < theoretical savings?**
1. **CUDA memory fragmentation**: PyTorch **reserves memory pools** and does not immediately release unused memory.
2. **Alignment & Padding Overhead**: CUDA requires **128-bit/256-bit** memory alignment, leading to **wasted memory**.
3. **Temporary buffers and workspace allocations**: Some GPU operations **still require FP16/FP32 workspaces**, which take up additional memory.