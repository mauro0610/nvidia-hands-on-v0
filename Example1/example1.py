import cupy as cp

N = 10000 

#select Device with index 0.
with cp.cuda.Device(0):
    #input data initialzed
    d_A = cp.arange(N, dtype=cp.int32)
    d_B = cp.arange(N, dtype=cp.int32)
    d_C = cp.zeros(N, dtype=cp.int32) # initialize zero filled array
    d_C = d_A + d_B

#optional: copy result from Device to Host
h_C = cp.asnumpy(d_C)
print(h_C)

