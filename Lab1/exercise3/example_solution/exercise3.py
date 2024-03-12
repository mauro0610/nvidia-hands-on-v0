import cupy as cp
import numpy as np
import math

N = 10000 #initialize array size

mul_array = cp.RawKernel(r'''
   extern "C" __global__
   void mulFunc(const int *d_A, const int *d_B, int *d_C) {
      int tid = threadIdx.x + blockDim.x * blockIdx.x;
      d_C[tid] = d_A[tid] * d_B[tid];
   }
''', 'mulFunc')


h_A = np.arange(N, dtype=np.int32)
h_B = np.arange(N, dtype=np.int32)

d_A = cp.asarray(h_A)
d_B = cp.asarray(h_B)
d_C = cp.zeros(N, dtype=cp.int32)

num_threads_per_block = 256
num_blocks_per_grid = math.ceil(N / num_threads_per_block)

mul_array((num_blocks_per_grid,), (num_threads_per_block,), (d_A, d_B, d_C))
h_C = cp.asnumpy(d_C)
print(h_C)

#expected output[       0        1        4 ... 99940009 99960004 99980001]
