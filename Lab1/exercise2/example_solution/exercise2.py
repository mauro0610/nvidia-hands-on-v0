import cupy as cp

N = 225

#generate matrix 
d_A = cp.random.random((N, N), dtype=cp.float32)
d_B = cp.random.random((N, N), dtype=cp.float32)
#d_B = cp.random.random(N*N, dtype=cp.float32).reshape(N, N)

#apply matrix operator @ or cp.dot()
d_C = d_A @ d_B
d_C2 = cp.dot(d_A, d_B)
print(d_C)
print(d_C2)

