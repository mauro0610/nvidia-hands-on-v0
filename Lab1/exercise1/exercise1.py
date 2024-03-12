import cupy as cp
import numpy as np

#defining array size
N=500000

#selecting Device with index 0
with cp.cuda.Device(0):
	x_gpu=cp.arange(N, dtype=cp.float32)
	y_gpu=cp.arange(N, dtype=cp.float32)
	array_sum=cp.zeros(N,dtype=cp.float32)
	array_sum=x_gpu+y_gpu

#optional
result_cpu=cp.asnumpy(array_sum)
print(result_cpu)



#expected output: [     0      2      4 ... 999994 999996 999998]
