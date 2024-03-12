import cupy as cp

N = 225

#generate matrix 
mA=cp.random.random((N,N), dtype=cp.float32)
mB=cp.random.random(N*N, dtype=cp.float32).reshape(N,N)

matrixproduct=mA@mB

print(matrixproduct)


#apply matrix operator @ or cp.dot()



