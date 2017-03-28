import scipy.constants as cte
from scipy.sparse import dia_matrix
import scipy as sc
import numpy as np
from conv import convmat2D
import rcwa
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # ------ Step 1: Initialize Program ---------------------------------------------------
    # define constants
    c0 = cte.speed_of_light     # speed of light
    u0 = cte.mu_0               # permeability of free space
    eps0 = cte.epsilon_0        # permittivity of free space
    eta0 = np.sqrt(u0/eps0)     # impedance of free space

    # Initial parameters
    u_ref, eps_ref = (1.0, 2.0)     # material property of reflection region
    u_trn, eps_trn = (1.0, 9.0)     # material property of transmission region
    theta, psi = (30, 20)           # incident angles
    lambda_0 = 2.0                  # wavelength
    k0 = 2 * np.pi / lambda_0       # wavenumber
    unit = 1e-2                     # units = cm
    pol = (0, 1)                    # polarization = [TE, TM]

    # device region
    u_r = 1.0                   # permeability of device triangle
    eps_r = 6.0                 # permittivity of device triangle

    # Unit cell size
    Lx = 1.75                   # unit cell length in x (cm)
    Ly = 1.50                   # unit cell length in y (cm)

    # layer thickness
    d1 = 0.50                   # first layer thickness (cm)
    d2 = 0.30                   # second layer thickness (cm)

    # Num of harmonics
    num_harmonics_x = int(7 * (Lx / lambda_0))
    num_harmonics_y = int(7 * (Ly / lambda_0))
    MN = num_harmonics_x*num_harmonics_y

    # ----- Step 2: Build device on grid --------------------------------------------------
    # triangle size in layer 1
    w = 0.8*Ly

    num_x = 1000    # x, y resolution: 1000 pts
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

    # Layer 1
    EPS_DEV_1 = np.ones((num_x, num_y)) * eps_r
    U_DEV_1 = np.ones((num_x, num_y)) * u_r

    # Layer 2
    EPS_DEV_2 = EPS_DEV_1
    U_DEV_2 = U_DEV_1

    for ny in range(ny1, ny2 + 1):
        f = 1 - float((ny - ny1 + 1)) / (ny2 - ny1 + 1)
        nx = int(np.round(f * w / dx))
        nx1 = int(np.floor((num_x - nx) / 2.0))
        nx2 = nx1 + nx - 1
        EPS_DEV_1[nx1:nx2, ny - 1] = eps_ref
        U_DEV_1[nx1:nx2, ny - 1] = u_ref

    # ----- Step 3: Compute Convolution Matrices ------------------------------------------
    EPS_DEV_1_C = convmat2D(EPS_DEV_1, *(num_harmonics_x, num_harmonics_y))
    U_DEV_1_C = convmat2D(U_DEV_1, *(num_harmonics_x, num_harmonics_y))

    EPS_DEV_2_C = convmat2D(EPS_DEV_2, *(num_harmonics_x, num_harmonics_y))
    U_DEV_2_C = convmat2D(U_DEV_2, *(num_harmonics_x, num_harmonics_y))

    # ----- Step 4: Compute Wave Vector Expansion, and normalize with respect to k0 -------
    k_inc_norm = np.sqrt(u_ref * eps_ref) * np.array([np.sin(theta)*np.cos(psi),
                                                      np.sin(theta)*np.sin(psi),
                                                      np.cos(theta)
                                                      ]
                                                     )
    # Note that in python 2, 3/2 = 1, but in python 3, 3/2 = 1.5 (automatic type conversion)
    kx_norm = k_inc_norm[0] - 2*np.pi*np.arange(-int(num_harmonics_x/2),
                                                int(num_harmonics_x/2) + 1)/(k0*Lx)
    ky_norm = k_inc_norm[1] - 2*np.pi*np.arange(-int(num_harmonics_y/2),
                                                int(num_harmonics_y/2) + 1)/(k0*Ly)

    # KX = Kx1 kx1 kx1 ... N times ... kx2, kx2, ... N times ... kxm, kxm, ... N times, total MN diagonal
    KX = dia_matrix((kx_norm.repeat(num_harmonics_y), [0]), shape=(MN, MN))

    # KY = ky1, ... , kyn, ky1, ... , kyn, ky1, ... , kyn repeated M times, total MN diagonal
    KY = dia_matrix((np.array(ky_norm.tolist() * num_harmonics_x), [0]), shape=(MN, MN))

    kz_ref = []
    kz_trn = []
    for kx in kx_norm:
        for ky in ky_norm:
            kz_ref.append(np.conj(sc.sqrt(u_ref*eps_ref - kx**2 - ky**2)))
            kz_trn.append(np.conj(sc.sqrt(u_trn*eps_trn - kx**2 - ky**2)))

    # KZ_REF, KZ_TRN = kz_ref,trn(m,n) in the diagonal MN square matrix
    KZ_REF = dia_matrix((np.array(kz_ref), [0]), shape=(MN, MN))
    KZ_TRN = dia_matrix((np.array(kz_trn), [0]), shape=(MN, MN))

    # ----- Step 5: Compute Eigen-modes of free space ----------------------------------------------
    FreeSpace = rcwa.HomoLayer(num_harmonics_x, num_harmonics_y,
                               Lx, Ly, d=0,
                               u_r=1, eps_r=1, KX=KX, KY=KY,
                               units=unit)

    W0 = FreeSpace.cal_W()
    V0 = FreeSpace.cal_V()

    # Todo:
    # ----- Step 6: Initialize Device Scattering Matrix --------------------------------------------
    S11_dev = []
    S12_dev = []
    S21_dev = S12_dev
    S22_dev = S11_dev

    # ----- Step 7: Main loop iterates through layers ----------------------------------------------
    # Initialize parameters
    InitialVal = rcwa.Initial(u_ref, eps_ref,
                              u_trn, eps_ref,
                              theta, psi,
                              lambda_0,
                              unit,
                              pol
                              )

    source_field = InitialVal.get_source_field(num_harmonics_x, num_harmonics_y)

    # Layers
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

    # Todo:
    # Compute layer scattering matrix for the i-th layer
    # Update device scattering matrix

    # ----- Step 8: Compute Reflection side connection S-matrix
    Region_ref = rcwa.RegionLayer(num_harmonics_x, num_harmonics_y,
                                  Lx, Ly, d=0,
                                  u_r=u_ref, eps_r=eps_ref,
                                  KX=KX, KY=KY, KZ=KZ_REF,
                                  units=unit)
    S11_ref, S12_ref, S21_ref, S22_ref = Region_ref.cal_S(W0, V0)

    # ----- Step 9: Compute Transmission side connection S-matrix
    Region_trn = rcwa.RegionLayer(num_harmonics_x, num_harmonics_y,
                                  Lx, Ly, d=0,
                                  u_r=u_trn, eps_r=eps_trn,
                                  KX=KX, KY=KY, KZ=KZ_TRN,
                                  units=unit)
    S11_trn, S12_trn, S21_trn, S22_trn = Region_trn.cal_S(W0, V0)

    # Todo:
    # ----- Step 10: Compute global Scattering Matrix

    # ----- Step 11: Compute reflected and transmitted fields

    # ----- Step 12: Compute diffraction efficiencies

    # ----- Step 13: Verify conservation of energy

    # ----- Step Appendix: plot the device's eps_r, u_r and convolution matrices -------------------
    plt.subplot(221)
    plt.pcolormesh(xa, ya, EPS_DEV_1.T, cmap='RdBu_r')
    plt.axis('equal')
    plt.title('UNIT CELL ($\epsilon_r$)')
    plt.colorbar()

    plt.subplot(222)
    plt.pcolormesh(xa, ya, U_DEV_1.T, cmap='RdBu_r')
    plt.axis('equal')
    plt.title('UNIT CELL ($\mu_r$)')
    plt.colorbar()

    plt.subplot(223)
    plt.imshow(np.real(EPS_DEV_1_C.T), interpolation='none')
    plt.axis('equal')
    plt.title('$\epsilon_r$ CONVOLUTION MATRIX')
    plt.colorbar()

    plt.subplot(224)
    plt.imshow(np.real(U_DEV_1_C.T), interpolation='none')
    plt.axis('equal')
    plt.title('$\mu_r$ CONVOLUTION MATRIX')
    plt.colorbar()
    plt.show()
