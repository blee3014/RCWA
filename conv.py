import numpy as np
# import matplotlib.pyplot as plt
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
        Nx = A.shape[0]
    else:
        Nx, Ny = A.shape

    # Compute indices of spatial harmonics
    NH = P*Q
    p = np.array(range(-np.int(np.floor(P/2.0)), np.int(np.floor(P/2.0) + 1)))
    q = np.array(range(-np.int(np.floor(Q/2.0)), np.int(np.floor(Q/2.0) + 1)))

    # Compute Fourier coefficients of A
    Af = fftshift(fftn(A)) / (Nx*Ny)

    # Coordinate of the zero-order harmonic
    p0 = int(np.floor(Nx/2.0))
    q0 = int(np.floor(Ny/2.0))

    C = []

    # Fill in the convolution matrix
    # loop through the rows
    for qrow in range(Q+1):
        for prow in range(P+1):
            row = qrow*P + prow -1
            # loop through the columns
            for qcol in range(Q+1):
                for pcol in range(P+1):
                    col = qcol*P + pcol - 1
                    pfft = p[prow] - p[pcol]
                    qfft = q[qrow] - q[qcol]
                    C.append(Af[p0 + pfft - 1, q0+qfft - 1])

    return np.array(C).reshape((NH, NH))


# example script
if __name__ == '__main__':

    eps_ref = 1.0
    eps_r = 9.0

    num_x = 512
    num_y = num_x
    K_x = 50.0
    K_y = K_x

    x_grid = np.linspace(-K_x/2.0, K_x/2.0, num_x)
    y_grid = np.linspace(-K_y/2.0, K_y/2.0, num_y)

    X, Y = np.meshgrid(x_grid, y_grid)

    device = X**2 + Y**2 <= (0.5*K_x*0.6)**2

    ER = np.ones((num_x, num_y))
    ER_0 = np.ones((num_x, num_y))*eps_r
    for x in range(num_x):
        for y in range(num_y):
            if device[x][y] == 0:
                ER[x][y] = eps_r
            else:
                ER[x][y] = eps_ref
    w = 0.9*K_x
    dx = K_x / num_x
    dy = K_y / num_y
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

    ER_T = np.ones((num_x, num_y))*eps_ref
    for ny in range(ny1, ny2 + 1):
        f = 1 - float((ny - ny1 + 1)) / (ny2 - ny1 + 1)
        nx = int(np.round(f * w / dx))
        nx1 = int(np.floor((num_x - nx) / 2.0))
        nx2 = nx1 + nx - 1
        ER_T[nx1:nx2, ny - 1] = eps_r

    # rule of thumb: num_ harmonics = 10 * K / wavelength
    num_harmonics = 10

    # compute the convolution matrices
    ER_C = convmat2D(ER, P=num_harmonics, Q=num_harmonics)
    ER_0_C = convmat2D(ER_0, P=num_harmonics, Q=num_harmonics)
    ER_T_C = convmat2D(ER_T, P=num_harmonics, Q=num_harmonics)
    x = np.array(range(num_harmonics**2))
    y = x
    X_c, Y_c = np.meshgrid(x, y)

    # plt.subplot(2,3,1)
    # plt.pcolormesh(X, Y, ER.T, cmap='RdBu_r')
    # plt.axis('equal')
    # plt.colorbar()
    #
    # plt.subplot(2,3,2)
    # plt.pcolormesh(X, Y, ER_0.T, cmap='RdBu_r')
    # plt.axis('equal')
    # plt.colorbar()
    #
    # plt.subplot(2,3,3)
    # plt.pcolormesh(X, Y, ER_T.T, cmap='RdBu_r')
    # plt.axis('equal')
    # plt.colorbar()
    #
    # plt.subplot(2,3,4)
    # plt.imshow(np.real(ER_C.T),interpolation='none')
    # plt.axis('equal')
    # plt.colorbar()
    #
    # plt.subplot(2,3,5)
    # plt.imshow(np.real(ER_0_C.T),interpolation='none')
    # plt.axis('equal')
    # plt.colorbar()
    #
    # plt.subplot(2,3,6)
    # plt.imshow(np.real(ER_T_C.T),interpolation='none')
    # plt.axis('equal')
    # plt.colorbar()
    # plt.show()
