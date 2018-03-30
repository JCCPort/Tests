import numpy as np
from timeit import default_timer as timer
from numba import vectorize, cuda

@vectorize(["float32(float32, float32)"], target='cuda')
def add(a, b):
    return a + b

@vectorize(["float32(float32, float32)"], target='cuda')
def testfunc(a, b):
    return (a*b)**(a/100)


N = 60000000


def main():
    T = np.ones(N, dtype=np.float32)
    A = cuda.device_array_like(T)
    B = cuda.device_array_like(T)
    C = cuda.device_array_like(T)
    D = cuda.device_array_like(T)
    E = cuda.device_array_like(T)
    X = cuda.device_array(N, dtype=np.float32)

    start = timer()
    for i in range(0, 10):
        X = testfunc(A, B)
    add_time = timer() - start
    free = cuda.current_context().get_memory_info()[0]/1e9
    total = cuda.current_context().get_memory_info()[1]/1e9
    print('----------------------------------------')
    print(" CUDA Addition took {} seconds".format(add_time))
    print(" Free memory: {} GB, Total memory: {} GB".format(free, total))
    print('----------------------------------------\n')

def add2():
    A2 = np.ones(N, dtype=np.float32)
    B2 = np.ones(N, dtype=np.float32)
    C2 = np.zeros(N, dtype=np.float32)

    start = timer()
    C2 = A2+B2
    add_time = timer() - start
    print("\nAddition-two took {} seconds \n".format(add_time))

def testfunc2():
    A2 = np.ones(N, dtype=np.float32)
    B2 = np.ones(N, dtype=np.float32)
    C2 = np.ones(N, dtype=np.float32)
    D2 = np.ones(N, dtype=np.float32)
    E2 = np.ones(N, dtype=np.float32)
    X2 = np.zeros(N, dtype=np.float32)

    start = timer()
    for i in range(0, 10):
        X2 = (A2*B2)**(A2/100)
    add_time = timer() - start
    print('----------------------------------------')
    print(" Python Addition took {} seconds ".format(add_time))
    print('----------------------------------------\n')

main()
testfunc2()