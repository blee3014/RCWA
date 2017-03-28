import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftn, fftshift
import time
from functools import wraps


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ("Total time running %s: %s seconds" % (function.func_name, str(t1-t0)))
        return result
    return function_timer

# @fn_timer     # to time the function
# @profile      # to see the memory usage
def convmat2D(A, P, Q=1):
    """
    Returns a convolution matrix from a real space grid of permittivity and permeability to solve Maxwell's equation in
    Fourier space.
    :param A: matrix of the unit cell in real space
    :param P: positive integer of spatial harmonics in x axis
    :param Q: positive integer of spatial harmonics in y axis
    :return: convolution matrix (calculate size)

    C = convmat(A, P)           for 1D
    C = convmat(A, P, Q)        for 2D
    """
    # -----------------------------------------------------------------------------------------------------------------
    # Handle input and output arguments
    if Q == 1:
        Nx = int(A.shape[0])
    else:
        Nx, Ny = A.shape

    # Compute indices of spatial harmonics
    NH = P*Q
    p = np.array(range(-np.int(P/2), np.int(P/2) + 1))
    q = np.array(range(-np.int(Q/2), np.int(Q/2) + 1))

    # Compute Fourier coefficients of A
    Af = fftshift(fftn(A)) / (Nx*Ny)

    # Coordinate of the zero-order harmonic
    p0 = int(Nx/2)
    q0 = int(Ny/2)

    C = []

    # Fill in the convolution matrix
    # loop through the rows
    for qrow in range(1, Q+1):
        for prow in range(1, P+1):
            # loop through the columns
            for qcol in range(1, Q+1):
                for pcol in range(1, P+1):
                    pfft = p[prow-1] - p[pcol-1]
                    qfft = q[qrow-1] - q[qcol-1]
                    C.append(Af[p0 + pfft, q0+qfft])

    return np.array(C).reshape((NH, NH))


# example script
if __name__ == '__main__':

    eps_ref = 2.0
    eps_r = 6.0

    num_x = 1500
    num_y = num_x
    Lx = 1.75
    Ly = 1.50
    w = 0.8*Ly
    lambda_0 = 2.0

    dx = Lx / num_x
    dy = Ly / num_y
    xa = np.arange(0, num_x) * dx
    ya = np.arange(0, num_y) * dy
    xa = xa - np.average(xa)
    ya = ya - np.average(ya)
    X, Y = np.meshgrid(xa, ya)

    # Grid indices of the triangle
    h = w * np.sqrt(3.0) / 2
    ny = int(np.round(h / dy))
    ny1 = int(np.floor((num_y - ny) / 2.0))
    ny2 = ny1 + ny - 1

    EPS_DEV_1 = np.ones((num_x, num_y)) * eps_r

    for ny in range(ny1, ny2 + 1):
        f = 1 - float((ny - ny1 + 1)) / (ny2 - ny1 + 1)
        nx = int(np.round(f * w / dx))
        nx1 = int(np.floor((num_x - nx) / 2.0))
        nx2 = nx1 + nx - 1
        EPS_DEV_1[nx1:nx2, ny - 1] = eps_ref

    # rule of thumb: num_ harmonics = 7 * K / wavelength
    num_harmonics_x = int(7 * (Lx / lambda_0))
    num_harmonics_y = int(7 * (Ly / lambda_0))

    # compute the convolution matrices
    EPS_DEV_1_C = convmat2D(EPS_DEV_1, P=num_harmonics_x, Q=num_harmonics_y)

    plt.subplot(121)
    plt.pcolormesh(xa, ya, EPS_DEV_1.T, cmap='RdBu_r')
    plt.axis('equal')
    plt.title('UNIT CELL ($\epsilon_r$)')
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(np.real(EPS_DEV_1_C.T), interpolation='none')
    plt.axis('equal')
    plt.title('$\epsilon_r$ CONVOLUTION MATRIX')
    plt.colorbar()
    plt.show()
