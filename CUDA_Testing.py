import numpy as np
from numpy import sin
from timeit import default_timer as timer
from numba import vectorize, cuda, types
import math
## from pyculib import rand FIGURE OUT HOW TO GET PYCULIB WORKING


@vectorize(["float32(float32, float32)"], target='cuda')
def add(a, b):
    return a + b


@vectorize(["float32(float32, float32)"], target='cuda')
def testfunc(a, b):
    return (a*b)**(a/100)


@vectorize(["float32(float32, float32)", "float32(float32, float32)"], nopython=True)
def sintest(a, b):
    return (sin(a))+(sin(b))


@vectorize(['f8(f8, f8)', 'u1(u1, u1)'])
def sinc(a, b):
    return math.sin(a) + math.sin(b)


def sintest2(a, b):
    return math.sin(a) + math.sin(b)


N = 10000000


def main(function):
    T = np.ones(N, dtype=np.uint8)
    A = cuda.device_array_like(T)
    B = cuda.device_array_like(T)
    A3 = np.random.poisson(lam=1, size=N)
    B3 = np.random.poisson(lam=1, size=N)
    X = cuda.device_array(N, dtype=np.float16)
    start = timer()
    X = function(A3, B3)
    print(X)
    add_time1 = timer() - start
    free = cuda.current_context().get_memory_info()[0]/1e9
    total = cuda.current_context().get_memory_info()[1]/1e9
    print('----------------------------------------')
    print(" CUDA Addition took {} seconds".format(add_time1))
    print(" Free memory: {} GB, Total memory: {} GB".format(free, total))
    print('----------------------------------------\n')
    return add_time1


def testfunc2(function):
    A2 = np.ones(N, dtype=np.uint8)
    B2 = np.ones(N, dtype=np.uint8)
    X2 = np.zeros(N, dtype=np.float32)
    A4 = np.random.poisson(lam=1, size=N)
    B4 = np.random.poisson(lam=1, size=N)
    start = timer()
    for i in range(0, N):
        X2[i] = function(A4[i], B4[i])
    add_time2 = timer() - start
    print(X2)
    print('----------------------------------------')
    print(" Python Addition took {} seconds ".format(add_time2))
    print('----------------------------------------\n')
    return add_time2


t1 = main(sinc)
t2 = testfunc2(sintest2)

print('Speed ratio: {}'.format(t2/t1))
