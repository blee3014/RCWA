import scipy.constants as cte
from scipy.sparse import dia_matrix
import numpy as np
from conv import convmat2D
import rcwa

if __name__ == '__main__':
    # define constants
    c0 = cte.speed_of_light
    u0 = cte.mu_0
    eps0 = cte.epsilon_0
    eta0 = np.sqrt(u0/eps0)

    # Initial parameters
    u_ref, eps_ref = (1.0, 2.0)
    u_trn, eps_trn = (1.0, 9.0)
    theta, psi = (30, 20)
    lambda_0 = 2.0
    k0 = 2 * np.pi / lambda_0
    unit = 1e-2
    pol = (0, 1)

    # device region
    u_r = 1.0
    eps_r = 6.0

    # Unit cell size
    Lx = 1.75   # cm
    Ly = 1.50   # cm

    # layer thickness
    d1 = 0.50   # cm
    d2 = 0.30   # cm

    # Num of harmonics
    num_harmonics_x = int(7 * (Lx / lambda_0))
    num_harmonics_y = int(7 * (Ly / lambda_0))
    MN = num_harmonics_x*num_harmonics_y

    # triangle size in layer 1
    w = 0.8*Ly

    # Compute Wave Vector Expansion, and normalize with k0
    k_inc_norm = np.sqrt(u_ref * eps_ref) * np.array([np.sin(theta)*np.cos(psi),
                                                      np.sin(theta)*np.sin(psi),
                                                      np.cos(theta)
                                                      ]
                                                     )
    # in python 2, 3/2 = 1, in python 3, 3/2 = 1.5 (automatic type conversion)
    kx_norm = k_inc_norm[0] - 2*np.pi*np.arange(-int(np.floor(num_harmonics_x/2)),
                                                int(np.floor(num_harmonics_x/2) + 1))/(k0*Lx)
    ky_norm = k_inc_norm[1] - 2*np.pi*np.arange(-int(np.floor(num_harmonics_y/2)),
                                                int(np.floor(num_harmonics_y/2) + 1))/(k0*Ly)

    # KX = Kx1 kx1 kx1 ... N times ... kx2, kx2, ... N times ... total M*N diagonal
    KX = dia_matrix((kx_norm.repeat(num_harmonics_y), [0]), shape=(MN, MN))

    # KY = ky1, ... , kyn, ky1, ... , kyn, ky1, ... , kyn repeated M times
    KY = dia_matrix((np.array(ky_norm.tolist() * num_harmonics_x), [0]), shape=(MN, MN))

    # KX_MESH, KY_MESH = np.meshgrid(kx_norm, ky_norm)
    # # Todo: Calculate Longitudinal wave vector components in Region I and Region II
    # # kz_ref = np.conj(np.sqrt(k_inc**2 * u_ref * eps_ref - KX_MESH**2 - KY_MESH**2))
    # # kz_trn = np.conj(np.sqrt(k_inc**2 * u_trn * eps_trn - KX_MESH**2 - KY_MESH**2))

    # Build device on grid
    num_x = 10
    num_y = num_x

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

    EPS_DEV_1 = np.ones((num_x, num_y))*eps_r
    U_DEV_1 = np.ones((num_x, num_y))*u_r
    EPS_DEV_2 = EPS_DEV_1
    U_DEV_2 = U_DEV_1

    for ny in range(ny1, ny2 + 1):
        f = 1 - float((ny - ny1 + 1)) / (ny2 - ny1 + 1)
        nx = int(np.round(f * w / dx))
        nx1 = int(np.floor((num_x - nx) / 2.0))
        nx2 = nx1 + nx - 1
        EPS_DEV_1[nx1:nx2, ny - 1] = eps_ref
        U_DEV_1[nx1:nx2, ny - 1] = u_ref

    # Step 3: Compute Convolution Matrices
    EPS_DEV_1_C = convmat2D(EPS_DEV_1, *(num_harmonics_x, num_harmonics_y))
    U_DEV_1_C = convmat2D(U_DEV_1, *(num_harmonics_x, num_harmonics_y))

    EPS_DEV_2_C = convmat2D(EPS_DEV_2, *(num_harmonics_x, num_harmonics_y))
    U_DEV_2_C = convmat2D(U_DEV_2, *(num_harmonics_x, num_harmonics_y))

    InitialVal = rcwa.Initial(u_ref, eps_ref,
                              u_trn, eps_ref,
                              theta, psi,
                              lambda_0,
                              unit,
                              pol
                              )

    source_field = InitialVal.get_source_field(num_harmonics_x, num_harmonics_y)

    FreeSpace = rcwa.HomoLayer(num_harmonics_x, num_harmonics_y,
                               Lx, Ly, 0,
                               1, 1,
                               units=unit)

    W0 = FreeSpace.cal_W()
    V0 = FreeSpace.cal_V()

    LayerOne = rcwa.Layer(num_harmonics_x, num_harmonics_y,
                          Lx, Ly, d1,
                          U_DEV_1_C, EPS_DEV_1_C,
                          units=unit)

    LayerTwo = rcwa.Layer(num_harmonics_x, num_harmonics_y,
                          Lx, Ly, d2,
                          U_DEV_2_C, EPS_DEV_2_C,
                          units=unit)

    # Calculate Eigenvectors, eigenvalues
    W1, V1, LAM1 = LayerOne.cal_eig(KX, KY)
    W2, V2, LAM2 = LayerTwo.cal_eig(KX, KY)

    # # plot the device's eps_r, u_r and convolution matrices
    # plt.subplot(221)
    # plt.pcolormesh(xa, ya, EPS_DEV.T, cmap='RdBu_r')
    # plt.axis('equal')
    # plt.title('UNIT CELL ($\epsilon_r$)')
    # plt.colorbar()
    #
    # plt.subplot(222)
    # plt.pcolormesh(xa, ya, U_DEV.T, cmap='RdBu_r')
    # plt.axis('equal')
    # plt.title('UNIT CELL ($\mu_r$)')
    # plt.colorbar()
    #
    # plt.subplot(223)
    # plt.imshow(np.real(EPS_DEV_C.T), interpolation='none')
    # plt.axis('equal')
    # plt.title('$\epsilon_r$ CONVOLUTION MATRIX')
    # plt.colorbar()
    #
    # plt.subplot(224)
    # plt.imshow(np.real(U_DEV_C.T), interpolation='none')
    # plt.axis('equal')
    # plt.title('$\mu_r$ CONVOLUTION MATRIX')
    # plt.colorbar()
    # plt.show()




