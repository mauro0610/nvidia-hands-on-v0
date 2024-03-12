import cupy as cp
from cupyx import jit
import math

@jit.rawkernel()
def mulFunc(d_A, d_B, d_C):
    tid = jit.threadIdx.x + jit.blockDim.x * jit.blockIdx.x
    d_C[tid] = d_A[tid] * d_B[tid]


N = 10000 #initialize array size

num_threads_per_block = 128
num_blocks_per_grid = math.ceil(N / num_threads_per_block)

d_A = cp.arange(N, dtype=cp.int32)
d_B = cp.arange(N, dtype=cp.int32)
d_C = cp.zeros(N, dtype=cp.int32)

mulFunc( (num_blocks_per_grid,), (num_threads_per_block,), (d_A, d_B, d_C) )
print("d_C:", d_C)
#print("d_A:", d_A)
#print("d_B:", d_B)


#expected output: [       0        1        4 ... 99940009 99960004 99980001]
