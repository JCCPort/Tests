import numpy as np
from numpy import sin
from timeit import default_timer as timer
from numba import vectorize, cuda, types
import math


@vectorize(["float32(float32, float32)"], target='cuda')
def add(a, b):
    return a + b


@vectorize(["float32(float32, float32)"], target='cuda')
def testfunc(a, b):
    return (a*b)**(a/100)


@vectorize(["float32(float32, float32)", "float32(float32, float32)"], nopython=True)
def sintest(a, b):
    return (sin(a))+(sin(b))


@vectorize(['f8(f8, f8)', 'f4(f4, f4)'])
def sinc(a, b):
    return math.sin(a*math.pi) + math.sin(b*math.pi)


def sintest2(a, b):
    return math.sin(a*math.pi) + math.sin(b*math.pi)


N = 60000


def main(function):
    T = np.ones(N, dtype=np.float32)
    A = cuda.device_array_like(T)
    B = cuda.device_array_like(T)
    X = cuda.device_array(N, dtype=np.float32)

    start = timer()
    X = function(A, B)
    add_time = timer() - start
    free = cuda.current_context().get_memory_info()[0]/1e9
    total = cuda.current_context().get_memory_info()[1]/1e9
    print('----------------------------------------')
    print(" CUDA Addition took {} seconds".format(add_time))
    print(" Free memory: {} GB, Total memory: {} GB".format(free, total))
    print('----------------------------------------\n')


def testfunc2(function):
    A2 = np.ones(N, dtype=np.float32)
    B2 = np.ones(N, dtype=np.float32)
    X2 = np.zeros(N, dtype=np.float32)
    start = timer()
    for i in range(0, N):
        X2[i] = function(A2[i], B2[i])
    add_time = timer() - start
    print('----------------------------------------')
    print(" Python Addition took {} seconds ".format(add_time))
    print('----------------------------------------\n')


main(sinc)
testfunc2(sintest2)
