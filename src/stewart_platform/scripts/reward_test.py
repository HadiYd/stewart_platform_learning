import numpy as np


S = np.array([0, 0, 0, 0])
Q = np.diagflat([[10,10], [10,10]])

print(" Q = ", S@Q@np.transpose(S))

test = [1,2,4,5,4,5]

print("test: ",test[:4])