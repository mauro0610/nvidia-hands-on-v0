import cupy as cp

N = 500000

with cp.cuda.Device(0):
    d_A = cp.arange(N, dtype=cp.int32)
    d_B = cp.arange(N, dtype=cp.int32)
    d_C = cp.zeros(N, dtype=cp.int32)

    d_C = d_A + d_B

h_C = cp.asnumpy(d_C)
print(h_C)
